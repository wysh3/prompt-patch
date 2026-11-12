"""E7 robustness: does the split-gate AG skill survive rephrased inputs?
Fit the B2 design on pool[0:600] (holdout pool[600:700]) with trigger 125110 at the frozen scale, then evaluate
teacher / unedited student / patched student on the dev slice pool[1500:1800] under four input conditions:
plain, lowercased, truncated to 120 chars, lowercased+truncated. Saves res_robustness.json.
"""
import argparse, json, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
ap = argparse.ArgumentParser(); ap.add_argument("--scale", type=float, default=10.0); ap.add_argument("--du", type=int, default=256)
ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--ntr", type=int, default=600); ap.add_argument("--nev", type=int, default=300)
A_ = ap.parse_args(); NTR, NEV, SCALE, DU, LAM = A_.ntr, A_.nev, A_.scale, A_.du, A_.lam
TID = 125110
E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_np = np.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
MASK = list(range(125100, 125164))
def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)
def student_run(ids):
    h = core.embed_tokens(mx.array([ids]))
    for layer in core.layers:
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o); ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
    return h
logits = lambda h: core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)
def nu_lg(ids):
    h = student_run(ids); nu = np.array(core.embedding_norm(h[:, -1:])[0, 0].astype(mx.float32))
    lg = np.array(logits(h)); lg[MASK] = -np.inf; return nu, lg
def row_deq(v):
    return np.array(mx.dequantize(emb.weight[v][None], emb.scales[v][None], emb.biases[v][None], group_size=EGS, bits=EBITS)[0].astype(mx.float32))
def set_row(v, val):
    o = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
    q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
    emb.weight[v] = q[0]; emb.scales[v] = s[0].astype(emb.scales.dtype); emb.biases[v] = b[0].astype(emb.biases.dtype); mx.eval(emb.parameters()); return o
def restore_row(v, o):
    emb.weight[v] = mx.array(o[0]); emb.scales[v] = mx.array(o[1]).astype(emb.scales.dtype); emb.biases[v] = mx.array(o[2]).astype(emb.biases.dtype); mx.eval(emb.parameters())
ag_usr = lambda t: enc(tmpl([{"role": "user", "content": t}]))
ag_tea = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
GEN = ["Explain photosynthesis in two sentences.", "What is the capital of France?", "Write a haiku about rain.", "Summarise the plot of Hamlet in three sentences."]
TPL = ["Answer this question briefly: {}", "Write two sentences about this topic: {}", "Summarise in one short sentence: {}", "Give a title for: {}"]
rms_row = float(np.sqrt((np.stack([row_deq(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
Elab = np.stack([row_deq(int(x)) for x in lid]); Evoc = np.stack([row_deq(x) for x in range(0, 125000, 400)]).mean(0)
u_lab = Elab.mean(0) - Evoc; u_lab /= np.linalg.norm(u_lab)
rng = np.random.default_rng(12345); r = rng.standard_normal(HID).astype(np.float32); r /= np.linalg.norm(r)
set_row(TID, (0.5 * r + 0.866 * u_lab) / np.linalg.norm(0.5 * r + 0.866 * u_lab) * rms_row * SCALE * np.sqrt(HID))
COV = [ag_usr(t.format(x)) for x in po_x[1000:1200] for t in TPL] + [ag_usr(g) for g in GEN]
PL = np.stack([nu_lg(i)[0] for i in COV]); C = (PL.T @ PL) / len(PL); evals, evec = np.linalg.eigh(C); U = evec[:, :DU]
TR = list(range(NTR)); HO = list(range(NTR, NTR + 100)); NV, TG, L0 = {}, {}, {}
for i in TR + HO:
    NV[i], TG[i] = nu_lg(ag_usr(po_x[i]) + [TID]); L0[i] = TG[i][lid]
NuTr = np.stack([NV[i] for i in TR]); X = NuTr @ U
Y = np.stack([nu_lg(ag_tea(po_x[i]))[1][lid] for i in TR]); Yc = Y - Y.mean(1, keepdims=True)
W = np.linalg.solve(X.T @ X + LAM * np.trace(X.T @ X) / HID * np.eye(DU), X.T @ Yc)
m = (NuTr @ U).mean(0); mhat = m / (np.linalg.norm(m) + 1e-12)
need = []
for i in HO:
    s = L0[i] + (NV[i] @ U @ W); nonlab = float(np.max(np.delete(TG[i], lid))); proj = float(NV[i] @ U @ mhat)
    if proj > 0: need.append((nonlab - s.max()) / proj)
BETA = max(0.0, float(np.quantile(need, 0.95)) * 1.05) if need else 0.0
dE = (U @ W).T + BETA * (U @ mhat)[None, :]
Ecur = np.stack([row_deq(v) for v in lid]); o_lab = [ (v, (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())) for v in lid]
def apply_patch():
    for k, v in enumerate(lid):
        q, s, b = mx.quantize(mx.array(np.asarray(Ecur[k] + dE[k], np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
        emb.weight[v] = q[0]; emb.scales[v] = s[0].astype(emb.scales.dtype); emb.biases[v] = b[0].astype(emb.biases.dtype)
    mx.eval(emb.parameters())
def apply_stock():
    for v, o in o_lab:
        emb.weight[v] = mx.array(o[0]); emb.scales[v] = mx.array(o[1]).astype(emb.scales.dtype); emb.biases[v] = mx.array(o[2]).astype(emb.biases.dtype)
    mx.eval(emb.parameters())

# ---- E7 also needs the flip patch: load its code deltas and apply them in memory ----
def unpack(w):
    bits = int(w.bits); Wp = np.array(w.weight); rows, nw = Wp.shape
    b = Wp.view(np.uint8).reshape(rows, nw, 4)
    g = np.unpackbits(b, axis=-1).reshape(rows, nw, 4, 8)[..., ::-1].reshape(rows, nw * 32)
    n = (nw * 32) // bits
    g = g[:, :n * bits].reshape(rows, n, bits)
    c = np.zeros((rows, n), np.int64)
    for i in range(bits): c |= g[..., i].astype(np.int64) << i
    return c.astype(np.int8)
def pack(c, bits):
    rows, n = c.shape
    g = np.zeros((rows, n, bits), np.uint8)
    for i in range(bits): g[..., i] = (c >> i) & 1
    flat = g.reshape(rows, -1); pad = (-flat.shape[1]) % 32
    if pad: flat = np.concatenate([flat, np.zeros((rows, pad), np.uint8)], axis=1)
    nb = flat.shape[1] // 32
    return np.packbits(flat.reshape(rows, nb, 4, 8)[..., ::-1], axis=-1).reshape(rows, nb * 4).view(np.uint32).reshape(rows, nb)
FLIP = np.load("patches/flippatch_L8-12-16-20-24_it800_n300.npz")
FLIP_LAYERS = sorted({int(k.split("_")[1]) for k in FLIP.keys() if k.startswith("idx_")})
FLIP_ORIG = {L: np.array(core.layers[L - 1].feed_forward.w2.weight).copy() for L in FLIP_LAYERS}
def apply_flip():
    for L in FLIP_LAYERS:
        w = core.layers[L - 1].feed_forward.w2; bits = int(w.bits)
        codes = unpack(w); idx = FLIP[f"idx_{L}"].astype(np.int64); sign = FLIP[f"sign_{L}"].astype(np.int64)
        rows, cols = idx // int(FLIP[f"shape_{L}"][1]), idx % int(FLIP[f"shape_{L}"][1])
        codes[rows, cols] = (codes[rows, cols].astype(np.int64) + sign).astype(np.int8)
        w.weight = mx.array(pack(codes, bits)); mx.eval(w.weight)
def remove_flip():
    for L in FLIP_LAYERS:
        core.layers[L - 1].feed_forward.w2.weight = mx.array(FLIP_ORIG[L])
    mx.eval(core.layers[FLIP_LAYERS[0] - 1].feed_forward.w2.weight)

DEV = [po_x[1500 + i] for i in range(NEV)]; YDEV = po_y[1500:1500 + NEV]
CONDS = {"plain": lambda t: t, "lower": lambda t: t.lower(), "trunc120": lambda t: t[:120], "lower_trunc120": lambda t: t[:120].lower()}
def evaluate(trigger=True):
    out = {}
    for cname, fn in CONDS.items():
        teacher, stu, pat, stu_lab, pat_lab = [], [], [], [], []
        for t in DEV:
            tt = fn(t)
            tg = nu_lg(ag_tea(tt))[1]; _, lg0 = nu_lg(ag_usr(tt)); _, lg1 = nu_lg(ag_usr(tt) + ([TID] if trigger else []))
            teacher.append(int(np.argmax(tg))); stu.append(int(np.argmax(lg0))); pat.append(int(np.argmax(lg1)))
            stu_lab.append(int(np.argmax(lg0[lid_np]))); pat_lab.append(int(np.argmax(lg1[lid_np])))
        out[cname] = dict(teacher_says=round(float(np.mean(lid_np[YDEV] == np.array(teacher))) * 100, 1),
                          student_says=round(float(np.mean(lid_np[YDEV] == np.array(stu))) * 100, 1),
                          patched_says=round(float(np.mean(lid_np[YDEV] == np.array(pat))) * 100, 1),
                          patched_label_restricted=round(float(np.mean(np.array(pat_lab) == YDEV)) * 100, 1),
                          student_label_restricted=round(float(np.mean(np.array(stu_lab) == YDEV)) * 100, 1),
                          _free=np.array(stu), _free_pat=np.array(pat))
    return out
apply_stock(); stock = evaluate()
apply_patch(); patched = evaluate()
apply_stock(); apply_flip(); flipstate = evaluate(trigger=False); remove_flip()
apply_stock(); assert np.array_equal(np.array(core.layers[FLIP_LAYERS[0] - 1].feed_forward.w2.weight), FLIP_ORIG[FLIP_LAYERS[0]])
for cname in CONDS:
    stock[cname]["off_state_first_token_identical_pct"] = round(float(np.mean(stock[cname]["_free"] == patched[cname]["_free_pat"])) * 100, 1)
    del stock[cname]["_free"]; del patched[cname]["_free_pat"]
    del stock[cname]["_free_pat"]; del patched[cname]["_free"]
out = dict(config=dict(scale=SCALE, du=DU, lam=LAM, beta=round(BETA, 3), ntr=NTR, nev=NEV), stock=stock, patched=patched, flip=flipstate)
def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if hasattr(o, "tolist"): return _clean(o.tolist())
    return o
json.dump(_clean(out), open("res_robustness.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000], flush=True); print("done", flush=True)
