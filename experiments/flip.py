"""Flip-patching: install a prompt into a 4-bit model by nudging a sparse set of its integer weight codes by one step.
No gradients, no added parameters, no side path: the result is the stock 4-bit file with some integers changed by +-1.

Teacher = frozen model WITH the system prompt.  Student = same model WITHOUT it.
For each edited layer l (in order, earlier edits applied) we want the MLP output matrix W2 (2048 x 7680, 4- or 6-bit, group 64)
to change so that the student's residual stream matches the teacher's.  Linearising the RMSNorm that follows W2:
    minimise  sum_n w_n || g * (dW a_n) / rms_n - r_n ||^2 ,   dW_ij = scale_ij * z_ij ,  z_ij in {-1, 0, +1}
(a_n = MLP hidden activation, r_n = teacher minus student residual, g = norm gain).  Rows of dW decouple and share the
Gram matrix H = sum w a a^T / rms^2, so this is an integer least-squares problem solved by greedy coordinate descent
(one flip per row per sweep, exact gain 2 e G - e^2 H_jj, gradient row updated with one row of H).
usage: flip.py <layers 1-based, comma> <max flips per row> [n_train_queries] [n_eval]"""
import copy, hashlib, json, sys, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

LAYSET = [int(x) - 1 for x in (sys.argv[1] if len(sys.argv) > 1 else "12,16,20,24").split(",")]
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
NTR = int(sys.argv[3]) if len(sys.argv) > 3 else 300
NEV = int(sys.argv[4]) if len(sys.argv) > 4 else 500
NHO, WT, LAMG = 60, 10.0, 0.01
TAG = f"L{'-'.join(str(l + 1) for l in LAYSET)}_it{MAXIT}_n{NTR}"
E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
ev_x, ev_y, po_x, po_y = load()
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_a = mx.array(lid); lid_np = np.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
tea_ids = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
stu_ids = lambda t: enc(tmpl([{"role": "user", "content": t}]))

def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)
def teacher_run(ids, cache, capture=()):
    h = core.embed_tokens(mx.array([ids])); out = {}
    for i, (layer, c) in enumerate(zip(core.layers, cache)):
        h = layer(h, mask_for(layer, h, c), c, None)
        if i in capture: out[i] = h[0]
    return h, out
def student_run(ids, capture=()):
    h = core.embed_tokens(mx.array([ids])); cap = {}
    for i, layer in enumerate(core.layers):
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o); ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
        if i in capture: cap[i] = (a[0], y[0], h[0])
    return h, cap
logits = lambda h: core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)

# ---- mixed-precision code access (generic b-bit MLX packing: codes are a little-endian bit stream;
# verified against mx.dequantize for bits=4 and bits=6 in qtest3 / probe_q.py)
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
    flat = g.reshape(rows, -1)
    pad = (-flat.shape[1]) % 32
    if pad: flat = np.concatenate([flat, np.zeros((rows, pad), np.uint8)], axis=1)
    nb = flat.shape[1] // 32
    b = np.packbits(flat.reshape(rows, nb, 4, 8)[..., ::-1], axis=-1).reshape(rows, nb * 4)
    return b.view(np.uint32).reshape(rows, nb)
def set_codes(l, c):
    w = core.layers[l].feed_forward.w2; w.weight = mx.array(pack(c, int(w.bits))); mx.eval(w.weight)
sha = lambda: hashlib.sha256(b"".join(np.array(core.layers[l].feed_forward.w2.weight).tobytes() for l in LAYSET)).hexdigest()
CODES0 = {l: unpack(core.layers[l].feed_forward.w2) for l in LAYSET}; SHA0 = sha()
for l in LAYSET: assert np.array_equal(pack(CODES0[l], int(core.layers[l].feed_forward.w2.bits)), np.array(core.layers[l].feed_forward.w2.weight))

# ---- alignment and teacher prefix cache
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1]); TAIL = next(i for i in range(1, 50) if a0[-i] != b0[-i]) - 1
def align(t):
    s, te = stu_ids(t), tea_ids(t); n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return s, te, len(s) - n, len(te) - n
s, te, NB, P = align(po_x[0]); PREFIX = te[:P]
cache0 = E.lm.make_cache(); mx.eval(teacher_run(PREFIX, cache0)[0])
def teacher_q(t, extra=(), capture=()):
    s, te, nb, p = align(t); assert te[:p] == PREFIX and nb == NB
    return teacher_run(te[p:] + list(extra), copy.deepcopy(cache0), capture), s
R = dict(layers=[l + 1 for l in LAYSET], max_flips_per_row=MAXIT, prompt_tokens=P, n_train_queries=NTR, n_eval=NEV)
print("setup", R, flush=True)

t0 = time.time(); TR = list(range(NTR)); HO = list(range(NTR, NTR + NHO)); T_res, STU = {}, {}
for q in TR + HO:
    (h, _), s = teacher_q(po_x[q]); ans = int(mx.argmax(logits(h)).item()); STU[q] = s + [ans]
    (h, out), _ = teacher_q(po_x[q], extra=[ans], capture=LAYSET)
    T_res[q] = {l: np.array(out[l].astype(mx.float32)).astype(np.float16) for l in LAYSET}
print("teacher targets", round(time.time() - t0), "s", flush=True)

def true_gap(l, qs):
    """sum_n w_n ||teacher - student||^2 at layer l with the CURRENT weights (real forward, no linearisation)."""
    tot = 0.0
    for q in qs:
        _, cap = student_run(STU[q], capture={l}); d = mx.array(T_res[q][l].astype(np.float32)) - cap[l][2][NB:].astype(mx.float32)
        w = np.ones(d.shape[0], np.float32); w[-(TAIL + 1):] = WT; tot += float(((d * d).sum(-1) * mx.array(w)).sum().item())
    return tot

def greedy(H, Bm, sigma, codes, Kh, Rh, Wh, maxit, kmax=15):
    """integer least squares, rows in parallel. returns z (int8, in {-1,0,1}) at the best holdout checkpoint."""
    hd = np.diag(H).copy(); G = np.ascontiguousarray(Bm.T); z = np.zeros(G.shape, np.int8); s2h = (sigma * sigma) * hd[None, :]
    pen = np.zeros(G.shape, np.float32); rows = np.arange(G.shape[0]); base = float((Wh[:, None] * Rh ** 2).sum())
    checks = sorted({c for c in (10, 25, 50, 100, 200, 400, 800, 1600) if c < maxit} | {maxit}); hist, best = [], (1.0, z.copy(), 0)
    fi, fj = np.zeros(0, np.int64), np.zeros(0, np.int64)
    for it in range(1, maxit + 1):
        A = sigma * G; gain = 2 * np.abs(A) - s2h - pen
        if len(fi): gain[fi, fj] = -2 * A[fi, fj] * z[fi, fj] - s2h[fi, fj]       # flipped entries: gain of reverting
        j = gain.argmax(1); gj = gain[rows, j]; zj = z[rows, j]
        new = np.where(zj == 0, np.sign(A[rows, j]), 0).astype(np.int8)
        cj = codes[rows, j]; bad = ((new == 1) & (cj == kmax)) | ((new == -1) & (cj == 0))
        pen[rows[bad], j[bad]] = np.inf
        act = (gj > 0) & ~bad
        if not act.any() and not bad.any(): break
        e = np.where(act, sigma[rows, j] * (new - zj), 0).astype(np.float32)
        z[rows[act], j[act]] = new[act]; G -= e[:, None] * H[j]
        nz = np.nonzero(z); fi, fj = nz
        if it in checks:
            err = float((Wh[:, None] * (Rh - Kh @ (sigma * z).T) ** 2).sum()) / base
            hist.append(dict(sweeps=it, flips=int(len(fi)), holdout_unexplained_linear=round(err, 4)))
            if err < best[0]: best = (err, z.copy(), it)
    return best[1], hist, best[2]

def fit_layer(l):
    t0 = time.time(); layer = core.layers[l]; w2 = layer.feed_forward.w2
    bits = int(w2.bits); IN = int(w2.weight.shape[1]) * 32 // bits; GS = IN // int(w2.scales.shape[1])
    g = np.array(layer.post_ffn_norm.weight.astype(mx.float32)); S = np.repeat(np.array(w2.scales.astype(mx.float32)), GS, axis=1)
    KtK = mx.zeros((IN, IN)); KtR = mx.zeros((IN, 2048)); Kh, Rh, Wh = [], [], []
    for q in TR + HO:
        _, cap = student_run(STU[q], capture={l}); a, y, h = (v[NB:].astype(mx.float32) for v in cap[l])
        K = a / mx.sqrt((y * y).mean(-1, keepdims=True) + 1e-5); Rr = mx.array(T_res[q][l].astype(np.float32)) - h
        w = np.ones(K.shape[0], np.float32); w[-(TAIL + 1):] = WT
        if q in HO: Kh.append(np.array(K)); Rh.append(np.array(Rr)); Wh.append(w); continue
        Kw = K * mx.array(w)[:, None]; KtK = KtK + Kw.T @ K; KtR = KtR + Kw.T @ Rr; mx.eval(KtK, KtR)
    Kh, Rh, Wh = np.concatenate(Kh), np.concatenate(Rh), np.concatenate(Wh); base = float((Wh[:, None] * Rh ** 2).sum())
    H = np.array(KtK); Bm = np.array(KtR); tr = float(np.trace(H)) / IN; t_stats = time.time() - t0
    # reference: unconstrained float edit of the same matrix (dense ridge), linearised holdout error
    dense = {}
    for lam in (0.01, 0.1, 1.0):
        D = np.linalg.solve(H.astype(np.float64) + lam * tr * np.eye(IN), Bm.astype(np.float64))
        dense[lam] = round(float((Wh[:, None] * (Rh - Kh @ D) ** 2).sum()) / base, 4)
    H[np.diag_indices(IN)] += LAMG * tr; sigma = (g[:, None] * S).astype(np.float32)
    gap0 = true_gap(l, HO); t1 = time.time()
    z, hist, it_best = greedy(H, Bm, sigma, CODES0[l], Kh, Rh, Wh, MAXIT, kmax=(1 << bits) - 1); t_solve = time.time() - t1
    set_codes(l, CODES0[l] + z); gap1 = true_gap(l, HO)
    info = dict(layer=l + 1, bits=bits, flips=int((z != 0).sum()), pct_of_matrix=round(float((z != 0).mean()) * 100, 3), sweeps_used=it_best,
                holdout_unexplained_true=round(gap1 / gap0, 4), dense_float_edit_linear=dense, path=hist,
                seconds_stats=round(t_stats), seconds_solve=round(t_solve))
    print("layer", {k: v for k, v in info.items() if k != "path"}, flush=True); print("   path", hist, flush=True)
    return z, info

def evaluate(idx, tea=None):
    out = dict(lab=[], free=[], kl=[])
    for n, i in enumerate(idx):
        lg = logits(student_run(stu_ids(ev_x[i]))[0]); out["lab"].append(int(mx.argmax(lg[lid_a]).item())); out["free"].append(int(mx.argmax(lg).item()))
        if tea is not None:
            lp, lt = nn.log_softmax(lg), tea["lp"][n]; out["kl"].append(float((mx.exp(lt) * (lt - lp)).sum().item()))
    return {k: np.array(v) for k, v in out.items()}
idx = list(range(NEV)); y = ev_y[:NEV]; TEA = dict(lab=[], free=[], lp=[])
for i in idx:
    (h, _), _ = teacher_q(ev_x[i]); lg = logits(h); mx.eval(lg)
    TEA["lab"].append(int(mx.argmax(lg[lid_a]).item())); TEA["free"].append(int(mx.argmax(lg).item())); TEA["lp"].append(nn.log_softmax(lg))
TEA["lab"], TEA["free"] = np.array(TEA["lab"]), np.array(TEA["free"])
BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1); return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def summary(o):
    n = len(o["free"]); d = dict(says_correct_label=ci(lid_np[y[:n]] == o["free"]), label_restricted_acc=ci(o["lab"] == y[:n]),
                                 same_first_token_as_teacher=ci(o["free"] == TEA["free"][:n]))
    if len(o["kl"]): d["mean_KL_from_teacher_nats"] = round(float(np.mean(o["kl"])), 3)
    return d
R["teacher_with_prompt"] = dict(says_correct_label=ci(lid_np[y] == TEA["free"]), label_restricted_acc=ci(TEA["lab"] == y)); print("teacher", R["teacher_with_prompt"], flush=True)
o0 = evaluate(idx, TEA); R["student_no_prompt_stock"] = summary(o0); print("stock, no prompt", R["student_no_prompt_stock"], flush=True)
# unrelated prompts for a collateral check (greedy 12 tokens)
coll = [enc(tmpl([{"role": "user", "content": f"Summarise in one short sentence: {t}"}])) for t in po_x[1600:1630]]
gen = lambda: [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
G0 = gen()

Z, R["fit"] = {}, []
for l in LAYSET:
    Z[l], info = fit_layer(l); R["fit"].append(info)
SHA1 = sha(); nflip = int(sum((z != 0).sum() for z in Z.values()))
# save the patch in the deliverable format: flat row-major index + sign, plus shape/bits metadata
import os
os.makedirs("patches", exist_ok=True)
PATCHPATH = os.path.join("patches", f"flippatch_{TAG}.npz")
sav = {}
for l in LAYSET:
    fi, fj = np.nonzero(Z[l])
    sav[f"idx_{l + 1}"] = (fi.astype(np.uint32) * CODES0[l].shape[1] + fj.astype(np.uint32))
    sav[f"sign_{l + 1}"] = Z[l][fi, fj].astype(np.int8)
    sav[f"shape_{l + 1}"] = np.array(CODES0[l].shape, np.int32)
    sav[f"bits_{l + 1}"] = np.array(int(core.layers[l].feed_forward.w2.bits), np.int32)
np.savez_compressed(PATCHPATH, **sav)
of = evaluate(idx, TEA); R["student_no_prompt_FLIPPED"] = summary(of); print("FLIPPED", R["student_no_prompt_FLIPPED"], flush=True)
# the same patched integers through the stock library forward (what an unmodified runtime sees)
def lib_logits(ids):
    return E.lm(mx.array([ids])).logits[0, -1].astype(mx.float32)
def stock_eval(iidx):
    free, lab = [], []
    for i in iidx:
        lg = lib_logits(stu_ids(ev_x[i])); free.append(int(mx.argmax(lg).item())); lab.append(int(mx.argmax(lg[lid_a]).item()))
    return np.array(free), np.array(lab)
SIDX = idx[:200]
lf, ll = stock_eval(SIDX)
R["student_no_prompt_FLIPPED_stock_runtime"] = dict(
    says_correct_label=ci(lid_np[y[:len(SIDX)]] == lf), label_restricted_acc=ci(ll == y[:len(SIDX)]),
    free_argmax_agreement_with_manual_pct=round(float((lf == np.array(of["free"][:len(SIDX)])).mean()) * 100, 1),
    label_argmax_agreement_with_manual_pct=round(float((ll == np.array(of["lab"][:len(SIDX)])).mean()) * 100, 1))
print("FLIPPED (stock runtime)", R["student_no_prompt_FLIPPED_stock_runtime"], flush=True)
G1 = gen(); R["collateral_identical_generations_pct"] = round(float(np.mean([a == b for a, b in zip(G0, G1)])) * 100, 1)
R["stock_runtime_agrees"] = bool(R["student_no_prompt_FLIPPED_stock_runtime"]["label_argmax_agreement_with_manual_pct"] >= 95.0)
# patch size: flat index (uint32) + sign, raw and compressed
import io, zlib
raw = b"".join(np.flatnonzero(Z[l]).astype(np.uint32).tobytes() + np.packbits(Z[l][Z[l] != 0] > 0).tobytes() for l in LAYSET)
nedit = sum(int(CODES0[l].size) for l in LAYSET)
def _lm_slots(model_dir="models/omni-2b-4bit"):
    """total scalar quantisation-code slots in the language model (from config + safetensors header)"""
    import json as _json, struct as _struct
    _cfg = _json.load(open(model_dir + "/config.json"))
    _q = _cfg.get("quantization", {}); _d = int(_q.get("bits", 4))
    _bm = {k: v["bits"] for k, v in _q.items() if isinstance(v, dict) and "bits" in v}
    _idx = _json.load(open(model_dir + "/model.safetensors.index.json")); _tot = 0
    for _sh in set(_idx["weight_map"].values()):
        _f = open(model_dir + "/" + _sh, "rb"); _n = _struct.unpack("<Q", _f.read(8))[0]
        _hdr = _json.loads(_f.read(_n)); _f.close()
        for _name, _info in _hdr.items():
            if _name == "__metadata__" or not _name.startswith("language_model."): continue
            if _info.get("dtype") != "U32": continue
            _mod = _name[:-len(".weight")] if _name.endswith(".weight") else _name
            _b = _d
            for _k, _v in _bm.items():
                if _mod == _k or _mod.startswith(_k + "."): _b = _v
            _tot += int(np.prod(_info["shape"])) * 32 // _b
    return _tot
nlm = _lm_slots()
R["patch"] = dict(flips=nflip, pct_of_edited_matrices=round(nflip / nedit * 100, 3), lm_weight_scalars=nlm,
                  pct_of_language_model_weights=round(nflip / nlm * 100, 4),
                  bytes_raw=len(raw), bytes_zlib=len(zlib.compress(raw, 9)))
R["patch"]["file"] = PATCHPATH
print("patch", R["patch"], "collateral identical %", R["collateral_identical_generations_pct"], "stock runtime agrees", R["stock_runtime_agrees"], flush=True)
# control: same number of random +-1 nudges in the same matrices
rng = np.random.default_rng(1)
for l in LAYSET:
    k = int((Z[l] != 0).sum()); zr = np.zeros(Z[l].size, np.int8); pos = rng.choice(Z[l].size, k, replace=False); zr[pos] = rng.choice([-1, 1], k)
    set_codes(l, np.clip(CODES0[l] + zr.reshape(Z[l].shape), 0, (1 << int(core.layers[l].feed_forward.w2.bits)) - 1))
R["control_random_flips"] = summary(evaluate(idx[:200], dict(lp=TEA["lp"][:200]))); print("random flips", R["control_random_flips"], flush=True)
# uninstall: subtract the same integers; the bytes must be identical to the original
for l in LAYSET: set_codes(l, CODES0[l] + Z[l])
assert sha() == SHA1
for l in LAYSET: set_codes(l, unpack(core.layers[l].feed_forward.w2) - Z[l])
ou = evaluate(idx[:100]); R["uninstall"] = dict(sha256_identical_to_original=bool(sha() == SHA0), first_tokens_identical=bool(np.array_equal(ou["free"], o0["free"][:100])))
print("uninstall", R["uninstall"], flush=True)
json.dump(R, open(f"res_flip_{TAG}.json", "w"), indent=1)
np.savez_compressed(f"flippatch_{TAG}.npz", **{f"z{l + 1}": Z[l] for l in LAYSET})
