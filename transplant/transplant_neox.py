"""Cross-model circuit transplant on Pythia (GPT-NeoX), no training.

A = Pythia-160M (final, d=768, 12 layers x 12 heads of 64) has induction heads.
B = Pythia-70M at step 512 (d=512, 6 layers x 8 heads of 64), BEFORE its induction phase change (step 512 -> 1000).
Head size is 64 in both and the rotary layout is identical, so a head can be moved without compressing it.

Method (closed form, forward passes only):
  1. In A, score heads: previous-token attention on text, induction attention on repeated random tokens.
  2. In B, the overwritten slot is the head whose ablation hurts text loss least, at the same relative depth.
  3. Q/K/V of the moved head: ridge regression from B's own LayerNorm output at that layer to A's pre-rotary
     q,k,v of that head (same inputs run through both models). Written into B's fused query_key_value rows.
  4. Output of the moved head, two modes:
       S       : Wo_B = S^T Wo_A, S = ridge map A residual -> B residual after that layer.
       channel : the previous-token head writes its value into B's 64 lowest-variance residual directions,
                 so the induction head (fitted afterwards, on the edited B) can read it.
     The induction head always uses S.
  Heads are moved in order (previous-token first), and each later fit uses the already edited B.
Fit data and eval data are disjoint (different text windows and different random-token seeds).
Controls: no edit; naive copy (A's weights cropped to B's width); random weights; the same pipeline applied to
random non-circuit A heads; B's own later checkpoints (what training produced).
"""
import argparse, json, os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--bstep", type=int, default=512); ap.add_argument("--k", type=int, default=1)
ap.add_argument("--lam", type=float, default=1e-3); ap.add_argument("--tag", default=""); ap.add_argument("--quick", action="store_true"); ap.add_argument("--indout", default="S", choices=["S", "logit"]); ap.add_argument("--bmode", default="early", choices=["early", "ablate"])
AR = ap.parse_args(); torch.set_grad_enabled(False); torch.manual_seed(0)
CD = os.environ["HF_HOME"]
def load(name, step): return GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{name}", revision=f"step{step}", cache_dir=CD, attn_implementation="eager", dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", cache_dir=CD)

# ---------------- data (fit and eval disjoint)
rows = pq.read_table(os.path.join(HERE, "..", "data", "agnews_train.parquet")).column("text").to_pylist()
ids = np.array(tok("\n".join(rows[:20000]))["input_ids"]); T = 128; R = 50
rng = np.random.default_rng(0); starts = rng.permutation(len(ids) // T - 1)
TEXT_FIT = torch.tensor(np.stack([ids[s * T:(s + 1) * T] for s in starts[:64]]))
TEXT_EV = torch.tensor(np.stack([ids[s * T:(s + 1) * T] for s in starts[64:192]]))
def rep(seed, n): g = torch.Generator().manual_seed(seed); x = torch.randint(1000, 20000, (n, R), generator=g); return torch.cat([x, x], 1)
IND_FIT, IND_EV = rep(1, 64), rep(2, 64)
FIT = [TEXT_FIT, IND_FIT]

def scores(m):
    lt = m(TEXT_EV).logits[:, :-1]; text = torch.nn.functional.cross_entropy(lt.reshape(-1, lt.shape[-1]), TEXT_EV[:, 1:].reshape(-1)).item()
    li = m(IND_EV).logits[:, :-1]; l = torch.nn.functional.cross_entropy(li.reshape(-1, li.shape[-1]), IND_EV[:, 1:].reshape(-1), reduction="none").view(IND_EV.shape[0], -1)
    f, s = l[:, 1:R - 1].mean().item(), l[:, R:].mean().item()
    return dict(text_loss=round(text, 4), induction_score=round(f - s, 3), first_copy_loss=round(f, 3), second_copy_loss=round(s, 3))
def roles(m):
    pt = m(TEXT_EV[:32], output_attentions=True).attentions; pi = m(IND_EV[:32], output_attentions=True).attentions
    prev = torch.stack([a.diagonal(offset=-1, dim1=2, dim2=3).mean((0, 2)) for a in pt])          # (L,H)
    t = torch.arange(R, 2 * R - 1); ind = torch.stack([a[:, :, t, t - R + 1].mean((0, 2)) for a in pi])
    return prev.numpy(), ind.numpy()

SLOTS = []                      # ablate mode: B's own circuit heads, zeroed in every copy of B
def loadB(step):
    m = load("pythia-70m", step)
    for l, h in SLOTS:
        q = layer(m, l).attention.query_key_value; q.weight[qkv_rows(h)] = 0; q.bias[qkv_rows(h)] = 0
        layer(m, l).attention.dense.weight[:, h * 64:(h + 1) * 64] = 0
    return m
def H(m): return m.config.num_attention_heads
def layer(m, l): return m.gpt_neox.layers[l]
def collect(m, l, xs):
    """LN input of layer l and residual stream after layer l, over the fit batches."""
    out = {"ln": [], "res": []}; hk = layer(m, l).input_layernorm.register_forward_hook(lambda mod, i, o: out["ln"].append(o.reshape(-1, o.shape[-1])))
    for x in xs: out["res"].append(m(x, output_hidden_states=True).hidden_states[l + 1].reshape(-1, m.config.hidden_size))
    hk.remove(); return {k: torch.cat(v).double().numpy() for k, v in out.items()}
def ridge(X, Y, lam):
    Xm, Ym = X.mean(0), Y.mean(0); Xc, Yc = X - Xm, Y - Ym; C = Xc.T @ Xc
    W = np.linalg.solve(C + lam * np.trace(C) / X.shape[1] * np.eye(X.shape[1]), Xc.T @ Yc); return W, Ym - Xm @ W
def r2(X, Y, W, c): P = X @ W + c; return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

def qkv_rows(h, hd=64): return slice(h * 3 * hd, (h + 1) * 3 * hd)
def move_head(mA, mB, la, ha, lb, hb, out_mode, lam, info):
    hd = 64; cA, cB = collect(mA, la, FIT), collect(mB, lb, FIT)
    qa = layer(mA, la).attention.query_key_value; Wa = qa.weight[qkv_rows(ha)].double().numpy(); ba = qa.bias[qkv_rows(ha)].double().numpy()
    target = cA["ln"] @ Wa.T + ba                                                   # A's pre-rotary q|k|v of head ha
    W, c = ridge(cB["ln"], target, lam); info["qkv_fit_r2"] = round(r2(cB["ln"], target, W, c), 4)
    qb = layer(mB, lb).attention.query_key_value; qb.weight[qkv_rows(hb)] = torch.tensor(W.T, dtype=qb.weight.dtype); qb.bias[qkv_rows(hb)] = torch.tensor(c, dtype=qb.bias.dtype)
    WoA = layer(mA, la).attention.dense.weight[:, ha * hd:(ha + 1) * hd].double().numpy()      # (768,64)
    dB = layer(mB, lb).attention.dense
    if out_mode == "S":
        S, _ = ridge(cA["res"], cB["res"], lam); info["S_fit_r2"] = round(r2(cA["res"], cB["res"], S, _), 4); WoB = S.T @ WoA
    elif out_mode == "logit":                                                        # vocabulary-anchored: same direct logit effect
        def U(m): fn = m.gpt_neox.final_layer_norm; return (m.get_output_embeddings().weight.double() * fn.weight.double()).numpy()   # (V, d)
        UA, UB = U(mA), U(mB); Mt = np.linalg.lstsq(UB, UA, rcond=None)[0]           # UB Mt ~ UA  ->  w_B = w_A Mt^T... (d_B x d_A)
        def fsd(m): return float(np.median(torch.cat([m(x, output_hidden_states=True).hidden_states[-1].reshape(-1, m.config.hidden_size) for x in FIT]).std(-1).numpy()))
        ratio = fsd(mB) / fsd(mA); WoB = (Mt @ WoA) * ratio; info["logit_map_r2"] = round(r2(UB, UA, Mt, 0), 4); info["norm_ratio"] = round(ratio, 4)
    else:                                                                            # private low-variance channel
        Xc = cB["res"] - cB["res"].mean(0); _, sv, Vt = np.linalg.svd(Xc, full_matrices=False); Pch = Vt[-hd:].T   # (512,64)
        v = target[:, 2 * hd:]; alpha = np.median(np.linalg.norm(v @ WoA.T, axis=1)) / np.median(np.linalg.norm(v, axis=1))
        WoB = Pch * alpha; info["channel_sv_ratio"] = round(float(sv[-hd:].mean() / sv.mean()), 4)
    dB.weight[:, hb * hd:(hb + 1) * hd] = torch.tensor(WoB, dtype=dB.weight.dtype)

def importance(m):
    base = scores(m)["text_loss"]; imp = np.zeros((m.config.num_hidden_layers, H(m)))
    for l in range(m.config.num_hidden_layers):
        for h in range(H(m)):
            d = layer(m, l).attention.dense; hk = d.register_forward_pre_hook(lambda mod, a, h=h: (a[0].index_fill(-1, torch.arange(h * 64, h * 64 + 64), 0.0),))
            lt = m(TEXT_FIT).logits[:, :-1]; imp[l, h] = torch.nn.functional.cross_entropy(lt.reshape(-1, lt.shape[-1]), TEXT_FIT[:, 1:].reshape(-1)).item() - base; hk.remove()
    return imp

def plan(pairs_A, LA, LB, imp):
    """map A (layer, head) -> B (layer, least-important free head) at the same relative depth, keeping order."""
    used, out, last = set(), [], -1
    for la, ha in pairs_A:
        lb = max(min(int(round(la * (LB - 1) / (LA - 1))), LB - 1), last if last >= 0 else 0)
        free = [h for h in np.argsort(imp[lb]) if (lb, h) not in used]; hb = int(free[0]); used.add((lb, hb)); out.append((la, ha, lb, hb)); last = lb
    return out

def edit(mA, steps, modes, lam, bstep):
    mB = loadB(bstep); infos = []
    for (la, ha, lb, hb), md in zip(steps, modes):
        i = dict(A=[int(la), int(ha)], B=[int(lb), int(hb)], out_mode=md); move_head(mA, mB, la, ha, lb, hb, md, lam, i); infos.append(i)
    return mB, infos

res = {}
mA = load("pythia-160m", 143000); res["A"] = scores(mA); prevA, indA = roles(mA); print("A", res["A"], flush=True)
if AR.bmode == "ablate":
    AR.bstep = 143000; mF = load("pythia-70m", 143000); res["B_intact"] = scores(mF); pF, iF = roles(mF)
    bi = [divmod(int(i), 8) for i in np.argsort(-iF.ravel())]; bi = [p for p in bi if p[0] >= 1][:AR.k]
    bp = [divmod(int(i), 8) for i in np.argsort(-pF.ravel())]; bp = [p for p in bp if p[0] < min(l for l, _ in bi)][:AR.k]
    SLOTS += sorted(bp) + sorted(bi); res["B_own_circuit"] = dict(prev=[[l, h, round(float(pF[l, h]), 3)] for l, h in bp], ind=[[l, h, round(float(iF[l, h]), 3)] for l, h in bi])
    print("B intact", res["B_intact"], res["B_own_circuit"], flush=True); del mF
mB0 = loadB(AR.bstep); res["B_before"] = scores(mB0); prevB, indB = roles(mB0); print("B before", res["B_before"], flush=True)
LA, LB = mA.config.num_hidden_layers, mB0.config.num_hidden_layers
# A's circuit: top-k induction heads (layer >= 1) and top-k previous-token heads in earlier layers than the first induction head
ind_order = [divmod(int(i), H(mA)) for i in np.argsort(-indA.ravel())]; ind_heads = [p for p in ind_order if p[0] >= 1][:AR.k]
lmin = min(l for l, _ in ind_heads); prev_order = [divmod(int(i), H(mA)) for i in np.argsort(-prevA.ravel())]
prev_heads = [p for p in prev_order if p[0] < lmin][:AR.k]
res["A_circuit"] = dict(prev_token=[[l, h, round(float(prevA[l, h]), 3)] for l, h in prev_heads], induction=[[l, h, round(float(indA[l, h]), 3)] for l, h in ind_heads])
res["B_before_roles"] = dict(max_prev=round(float(prevB.max()), 3), max_induction=round(float(indB.max()), 3))
print("A circuit", res["A_circuit"], "B roles", res["B_before_roles"], flush=True)
imp = importance(mB0) if AR.bmode == "early" else None; del mB0
steps = plan(sorted(prev_heads) + sorted(ind_heads), LA, LB, imp) if AR.bmode == "early" else [(la, ha, lb, hb) for (la, ha), (lb, hb) in zip(sorted(prev_heads) + sorted(ind_heads), SLOTS)]
res["plan"] = steps
kp = len(prev_heads)
for pm in ("S", "channel"):
    mT, info = edit(mA, steps, [pm] * kp + [AR.indout] * len(ind_heads), AR.lam, AR.bstep)
    pr, ir = roles(mT); res[f"transplant_{pm}"] = dict(scores=scores(mT), fits=info, max_induction_attn=round(float(ir.max()), 3)); print("transplant", pm, res[f"transplant_{pm}"], flush=True); del mT
    if AR.quick: continue
    mT, _ = edit(mA, steps[kp:], [AR.indout] * len(ind_heads), AR.lam, AR.bstep); res[f"induction_only_{pm}"] = scores(mT); del mT
    mT, _ = edit(mA, steps[:kp], [pm] * kp, AR.lam, AR.bstep); res[f"prev_only_{pm}"] = scores(mT); del mT
    # control: same pipeline, random A heads (not in the circuit) at the same layers
    ctl = []
    for sd in range(3):
        r = np.random.default_rng(sd); fake = []
        for la, ha, lb, hb in steps:
            cand = [h for h in range(H(mA)) if (la, h) not in prev_heads + ind_heads]; fake.append((la, int(r.choice(cand)), lb, hb))
        mT, _ = edit(mA, fake, [pm] * kp + [AR.indout] * len(ind_heads), AR.lam, AR.bstep); ctl.append(scores(mT)); del mT
    res[f"random_A_heads_{pm}"] = ctl; print("random A heads", pm, ctl, flush=True)
if not AR.quick:
    mN = loadB(AR.bstep)                                               # naive copy: crop A's weights to B's width
    for la, ha, lb, hb in steps:
        qa, qb = layer(mA, la).attention.query_key_value, layer(mN, lb).attention.query_key_value
        qb.weight[qkv_rows(hb)] = qa.weight[qkv_rows(ha), :512]; qb.bias[qkv_rows(hb)] = qa.bias[qkv_rows(ha)]
        layer(mN, lb).attention.dense.weight[:, hb * 64:(hb + 1) * 64] = layer(mA, la).attention.dense.weight[:512, ha * 64:(ha + 1) * 64]
    res["naive_copy"] = scores(mN); del mN
    rnd = []
    for sd in range(3):
        mR = loadB(AR.bstep); g = torch.Generator().manual_seed(sd)
        for la, ha, lb, hb in steps:
            q = layer(mR, lb).attention.query_key_value; w = q.weight[qkv_rows(hb)]; q.weight[qkv_rows(hb)] = torch.randn(w.shape, generator=g) * w.std()
            d = layer(mR, lb).attention.dense; w = d.weight[:, hb * 64:(hb + 1) * 64]; d.weight[:, hb * 64:(hb + 1) * 64] = torch.randn(w.shape, generator=g) * w.std()
        rnd.append(scores(mR)); del mR
    res["random_weights"] = rnd
    for s in (1000, 2000, 143000): res[f"B_own_step{s}"] = scores(load("pythia-70m", s))
    print({k: v for k, v in res.items() if k in ("naive_copy", "random_weights") or k.startswith("B_own") or k.endswith("only_S") or k.endswith("only_channel")}, flush=True)
json.dump(res, open(os.path.join(HERE, f"res_neox{AR.tag}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved", flush=True)
