"""Readout patching: install the teacher's first-token decision by editing a few rows of the tied embedding
(the output head) in place, closed form, no labels, exactly removable.

Teacher = model WITH the AG News system prompt; student = same model WITHOUT it.
For train queries q we collect the student's final normalised hidden vector nu_q (2048) and the teacher's
first-token logits on a small token set S.  The tied embedding rows E_S are changed by
    dE = argmin sum_q || E_S nu_q + dE nu_q - L^T_q ||^2 + lam ||dE||^2
i.e. a ridge regression of the teacher-student logit gap on the student's readout features.  The final patch
changes only quantised codes in those rows, so the file stays a stock 6-bit embedding table and the
uninstall restores the original codes exactly (verified on the packed uint32 words).
usage: readout.py [--ntr 300] [--nev 500] [--eval dev|test] [--suppress 0] [--lam -1(auto)]
"""
import argparse, copy, json, time, zlib, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM

_ap = argparse.ArgumentParser()
_ap.add_argument("--ntr", type=int, default=300); _ap.add_argument("--nev", type=int, default=500)
_ap.add_argument("--dev-start", type=int, default=1500); _ap.add_argument("--eval", choices=["dev", "test"], default="dev")
_ap.add_argument("--lam", type=float, default=-1.0); _ap.add_argument("--suppress", type=int, default=0)
_ap.add_argument("--prepatch-flip", default="", help="optional .npz flip patch applied to w2 codes before fitting the readout rows")
_ap.add_argument("--prepatch-gain", default="", help="optional .npz gain patch (as saved by gainpatch.py) applied first")
_ap.add_argument("--tag", default="")
_A = _ap.parse_args()
NTR, NEV, DEVSTART, EVALMODE = _A.ntr, _A.nev, _A.dev_start, _A.eval
TAG = _A.tag or f"n{NTR}_{EVALMODE}_sup{_A.suppress}"

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); VOC, HID = int(emb.weight.shape[0]), int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
ex, ey = (po_x[DEVSTART:DEVSTART + NEV], po_y[DEVSTART:DEVSTART + NEV]) if EVALMODE == "dev" else (ev_x[:NEV], ev_y[:NEV])
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_a = mx.array(lid); lid_np = np.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
tea_ids = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
stu_ids = lambda t: enc(tmpl([{"role": "user", "content": t}]))
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)
def teacher_run(ids, cache):
    h = core.embed_tokens(mx.array([ids]))
    for layer, c in zip(core.layers, cache): h = layer(h, mask_for(layer, h, c), c, None)
    return h
def student_run(ids):
    h = core.embed_tokens(mx.array([ids]))
    for layer in core.layers:
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o); ff = layer.forward if False else layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
    return h
logits = lambda h: core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)
nu_of = lambda text: core.embedding_norm(student_run(stu_ids(text))[:, -1:])[0, 0].astype(mx.float32)

# ---- bit-level code access for the 6-bit embedding rows (little-endian bit stream)
def unpack_row(wrow):
    bits = EBITS; Wp = np.array(wrow); nw = Wp.shape[0]
    b = Wp.view(np.uint8).reshape(1, nw, 4)
    g = np.unpackbits(b, axis=-1).reshape(1, nw, 4, 8)[..., ::-1].reshape(1, nw * 32)
    n = (nw * 32) // bits; g = g[:, :n * bits].reshape(1, n, bits)
    c = np.zeros((1, n), np.int64)
    for i in range(bits): c |= g[..., i].astype(np.int64) << i
    return c[0].astype(np.int64)
def pack_row(codes):
    bits = EBITS; c = np.asarray(codes, np.int64).reshape(1, -1); rows, n = c.shape
    g = np.zeros((rows, n, bits), np.uint8)
    for i in range(bits): g[..., i] = (c >> i) & 1
    flat = g.reshape(rows, -1); pad = (-flat.shape[1]) % 32
    if pad: flat = np.concatenate([flat, np.zeros((rows, pad), np.uint8)], axis=1)
    nb = flat.shape[1] // 32
    b = np.packbits(flat.reshape(rows, nb, 4, 8)[..., ::-1], axis=-1).reshape(rows, nb * 4)
    return b.view(np.uint32).reshape(nb)
def row_dequant(v):
    w = mx.array(np.array(emb.weight[v]).copy())[None]
    s = mx.array(np.array(emb.scales[v].astype(mx.float32)).copy())[None]
    b = mx.array(np.array(emb.biases[v].astype(mx.float32)).copy())[None]
    return np.array(mx.dequantize(w, s, b, group_size=EGS, bits=EBITS)[0].astype(mx.float32))
def set_rows(rows, values):
    for v, val in zip(rows, values):
        q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
        emb.weight[int(v)] = q[0]; emb.scales[int(v)] = s[0].astype(emb.scales.dtype); emb.biases[int(v)] = b[0].astype(emb.biases.dtype)
    mx.eval(emb.parameters())

# ---- teacher prefix cache and alignment
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1]); TAIL = next(i for i in range(1, 50) if a0[-i] != b0[-i]) - 1
def align(t):
    s, te = stu_ids(t), tea_ids(t); n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return s, te, len(s) - n, len(te) - n
s, te, NB, P = align(po_x[0]); PREFIX = te[:P]
cache0 = E.lm.make_cache(); mx.eval(teacher_run(PREFIX, cache0))
def teacher_q(t):
    s, te, nb, p = align(t); assert te[:p] == PREFIX and nb == NB
    return teacher_run(te[p:], copy.deepcopy(cache0))
# ---- optional stacking of an already-fitted in-file patch (w2 integer codes and/or bf16 norm gains)
def load_flip_patch(path):
    z = np.load(path); n = 0
    for k in z.files:
        if not k.startswith("idx_"): continue
        L = int(k.split("_")[1]); idxs = z[k].astype(np.int64); signs = z[k.replace("idx_", "sign_")].astype(np.int64)
        shape = tuple(int(x) for x in z[k.replace("idx_", "shape_")]); bits = int(z[k.replace("idx_", "bits_")][0])
        w = core.layers[L - 1].feed_forward.w2
        Wp = np.array(w.weight); rows, nw = Wp.shape
        b = Wp.view(np.uint8).reshape(rows, nw, 4)
        g = np.unpackbits(b, axis=-1).reshape(rows, nw, 4, 8)[..., ::-1].reshape(rows, nw * 32)
        nn = (nw * 32) // bits; g = g[:, :nn * bits].reshape(rows, nn, bits)
        c = np.zeros((rows, nn), np.int64)
        for i in range(bits): c |= g[..., i].astype(np.int64) << i
        r = idxs // shape[1]; col = idxs % shape[1]
        c[r, col] += signs
        assert c.min() >= 0 and c.max() <= (1 << bits) - 1
        gg = np.zeros((rows, nn, bits), np.uint8)
        for i in range(bits): gg[..., i] = (c >> i) & 1
        flat = gg.reshape(rows, -1); pad = (-flat.shape[1]) % 32
        if pad: flat = np.concatenate([flat, np.zeros((rows, pad), np.uint8)], axis=1)
        nb = flat.shape[1] // 32
        q = np.packbits(flat.reshape(rows, nb, 4, 8)[..., ::-1], axis=-1).reshape(rows, nb * 4).view(np.uint32)
        w.weight = mx.array(q); mx.eval(w.weight); n += len(idxs)
    return n
def load_gain_patch(path):
    z = np.load(path); n = 0
    for k in z.files:
        if not k.startswith("new_"): continue
        _, L, which = k.split("_"); mod = core.layers[int(L) - 1].post_operator_norm if which == "op" else core.layers[int(L) - 1].post_ffn_norm
        mod.weight = mx.array(z[k].astype(np.float32)).astype(mod.weight.dtype); mx.eval(mod.weight); n += 1
    return n
if _A.prepatch_flip: print("applied flip-patch codes:", load_flip_patch(_A.prepatch_flip), flush=True)
if _A.prepatch_gain: print("applied gain-patch tensors:", load_gain_patch(_A.prepatch_gain), flush=True)
R = dict(method="readout", n_train=NTR, eval=EVALMODE, dev_start=DEVSTART, prompt_tokens=P, suppress=_A.suppress,
         prepatch_flip=_A.prepatch_flip, prepatch_gain=_A.prepatch_gain)
print("setup", R, flush=True)

t0 = time.time(); TR = list(range(NTR)); HO = list(range(NTR, NTR + 60))
NU, TG = {}, {}
for q in TR + HO:
    NU[q] = np.array(nu_of(po_x[q])).astype(np.float32)
    TG[q] = np.array(logits(teacher_q(po_x[q]))).astype(np.float32)
print("targets", round(time.time() - t0), "s", flush=True)

S = [int(x) for x in lid]
if _A.suppress > 0:
    from collections import Counter
    cnt = Counter()
    for q in TR: cnt[int(mx.argmax(logits(student_run(stu_ids(po_x[q])))).item())] += 1
    S += [t for t, _ in cnt.most_common(_A.suppress) if t not in set(S)]
print("rows", len(S), flush=True)
E0 = np.stack([row_dequant(v) for v in S])                 # current dequantised rows
def build(qs):
    return np.stack([NU[q] for q in qs]), np.stack([TG[q][S] for q in qs])
Nu_tr, Y_tr = build(TR); Nu_ho, Y_ho = build(HO)
L0_tr, L0_ho = Nu_tr @ E0.T, Nu_ho @ E0.T
T_tr, T_ho = Y_tr - L0_tr, Y_ho - L0_ho
G = Nu_tr.T @ Nu_tr
def fit_ridge(lam):
    A = G + lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    DE = np.linalg.solve(A.astype(np.float64), (Nu_tr.T @ T_tr).astype(np.float64)).astype(np.float32)   # (2048, |S|)
    return DE, float(np.mean((T_tr - Nu_tr @ DE) ** 2)), float(np.mean((T_ho - Nu_ho @ DE) ** 2))
lams = [0.001, 0.01, 0.1, 1.0, 10.0] if _A.lam < 0 else [_A.lam]
cands = []
for lam in lams:
    DE_, mtr_, mho_ = fit_ridge(lam); cands.append((mho_, lam, mtr_, DE_))
cands.sort(key=lambda z: z[0])
mho, LAM, mtr, DE = cands[0]
print("lambda sweep (holdout mse):", [(round(c[0], 5), c[1]) for c in cands], "picked", LAM, flush=True)
R["ridge"] = dict(lam=LAM, train_mse=round(mtr, 5), holdout_mse=round(mho, 5))

idx = list(range(len(ex))); y = ey; TEA = dict(lab=[], free=[], lp=[])
for i in idx:
    lg = logits(teacher_q(ex[i])); mx.eval(lg)
    TEA["lab"].append(int(mx.argmax(lg[lid_a]).item())); TEA["free"].append(int(mx.argmax(lg).item())); TEA["lp"].append(nn.log_softmax(lg))
TEA["lab"], TEA["free"] = np.array(TEA["lab"]), np.array(TEA["free"])
BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def summary(o):
    n = len(o["free"]); d = dict(says_correct_label=ci(lid_np[y[:n]] == o["free"]), label_restricted_acc=ci(o["lab"] == y[:n]),
                                 same_first_token_as_teacher=ci(o["free"] == TEA["free"][:n]))
    if len(o["kl"]): d["mean_KL_from_teacher_nats"] = round(float(np.mean(o["kl"])), 3)
    return d
def eval_model(tea=None):
    out = dict(lab=[], free=[], kl=[])
    for i in idx:
        lg = logits(student_run(stu_ids(ex[i]))); out["lab"].append(int(mx.argmax(lg[lid_a]).item())); out["free"].append(int(mx.argmax(lg).item()))
        if tea is not None:
            lp, lt = nn.log_softmax(lg), tea["lp"][i]; out["kl"].append(float((mx.exp(lt) * (lt - lp)).sum().item()))
    return {k: np.array(v) for k, v in out.items()}
R["teacher_with_prompt"] = dict(says_correct_label=ci(lid_np[y] == TEA["free"]), label_restricted_acc=ci(TEA["lab"] == y))
o0 = eval_model(TEA); R["student_no_prompt_stock"] = summary(o0); print("stock", R["student_no_prompt_stock"], flush=True)
coll = [enc(tmpl([{"role": "user", "content": f"Summarise in one short sentence: {t}"}])) for t in po_x[1700:1730]]
gen = lambda: [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
G0 = gen()
# save the original packed words for every edited row, then check exact restore at the end
ORIG = {v: (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(),
            np.array(emb.biases[v].astype(mx.float32)).copy()) for v in S}
def restore_rows():
    for v in S:
        emb.weight[int(v)] = mx.array(ORIG[v][0])
        emb.scales[int(v)] = mx.array(ORIG[v][1]).astype(emb.scales.dtype)
        emb.biases[int(v)] = mx.array(ORIG[v][2]).astype(emb.biases.dtype)
    mx.eval(emb.parameters())
# float readout patch (ceiling of the row edit without quantisation)
set_rows(S, E0 + DE.T)
of = eval_model(TEA); R["student_no_prompt_READOUT_FLOAT"] = summary(of); print("READOUT_FLOAT", R["student_no_prompt_READOUT_FLOAT"], flush=True)
G1 = gen(); R["collateral_identical_generations_pct_float"] = round(float(np.mean([a == b for a, b in zip(G0, G1)])) * 100, 1)
Eq = np.stack([row_dequant(v) for v in S])
R["quantisation_row_mse"] = float(np.mean((Eq - (E0 + DE.T)) ** 2))
oq = eval_model(TEA); R["student_no_prompt_READOUT_QUANTISED"] = summary(oq); print("READOUT_QUANT", R["student_no_prompt_READOUT_QUANTISED"], flush=True)
G1 = gen(); R["collateral_identical_generations_pct_quantised"] = round(float(np.mean([a == b for a, b in zip(G0, G1)])) * 100, 1)
# patch bookkeeping: how many packed words / codes changed in the edited rows
changed = int(sum(int((np.array(emb.weight[v]) != ORIG[v][0]).sum()) for v in S))
R["patch"] = dict(rows=len(S), changed_word32=changed, changed_codes=changed * 32 // EBITS, bytes_raw=changed * 4,
                  pct_of_lm_codes=round(changed * 32 / EBITS / 1728413696 * 100, 6))
restore_rows()
identical = all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0])
                and np.array_equal(np.array(emb.scales[v].astype(mx.float32)), ORIG[v][1])
                and np.array_equal(np.array(emb.biases[v].astype(mx.float32)), ORIG[v][2]) for v in S)
ou = eval_model(); R["uninstall"] = dict(packed_rows_identical=bool(identical), first_tokens_identical=bool(np.array_equal(ou["free"], o0["free"])))
print("patch", R["patch"], "collateral", R["collateral_identical_generations_pct_quantised"], "uninstall", R["uninstall"], flush=True)
json.dump(R, open(f"res_readout_{TAG}.json", "w"), indent=1, default=lambda o: o.item())
print("done", flush=True)
