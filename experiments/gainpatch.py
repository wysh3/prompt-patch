"""Gain-patching: install a prompt's behaviour into the bf16 sandwich-norm gains of a 4-bit MLX model.
No gradients, no new parameters, closed form, exactly removable.
Teacher = model WITH the AG News system prompt; student = same model WITHOUT it.
For each edited layer l (in order, earlier edits applied) we capture, at every aligned train token,
  x_n = residual after the operator sub-layer, o_n = operator output, h_n = residual after the MLP,
  y_n = MLP output.
Sandwich norms give  x = h_prev + g_op * o/rms(o)   and   h = x + g_ffn * y/rms(y),
so a gain edit is a *linear* handle on the residual:  dx = dg_op * o_hat,  dh = dg_ffn * y_hat.
Each channel decouples; the closed-form per-channel least squares is
  dg_i = sum_n w_n r_ni phi_ni / sum_n w_n phi_ni^2 .
usage: gainpatch.py <layers 1-based> [--nev 500] [--eval dev|test] [--targets op,ffn]
"""
import argparse, copy, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM

_ap = argparse.ArgumentParser()
_ap.add_argument("layers"); _ap.add_argument("--ntr", type=int, default=300); _ap.add_argument("--nev", type=int, default=500)
_ap.add_argument("--dev-start", type=int, default=1500); _ap.add_argument("--eval", choices=["dev", "test"], default="dev")
_ap.add_argument("--targets", default="op,ffn"); _ap.add_argument("--tag", default="")
_A = _ap.parse_args()
LAYSET = [int(x) - 1 for x in _A.layers.split(",")]; NTR, NEV = _A.ntr, _A.nev
DEVSTART, EVALMODE, TARGETS = _A.dev_start, _A.eval, _A.targets.split(",")
NHO, WT = 60, 10.0
TAG = _A.tag or f"L{'-'.join(str(l + 1) for l in LAYSET)}_n{NTR}_{EVALMODE}_" + "-".join(TARGETS)

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
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

# ---- original gains (bf16 copies), patched gains kept separately
G_ORIG = {}
for l in LAYSET:
    lay = core.layers[l]
    G_ORIG[(l, "op")] = np.array(lay.post_operator_norm.weight.astype(mx.float32))
    G_ORIG[(l, "ffn")] = np.array(lay.post_ffn_norm.weight.astype(mx.float32))
def apply_gain(l, which, g_new_f32):
    mod = core.layers[l].post_operator_norm if which == "op" else core.layers[l].post_ffn_norm
    mod.weight = mx.array(g_new_f32.astype(np.float32)).astype(mod.weight.dtype)
    mx.eval(mod.weight)

def teacher_run2(ids, cache, capture=()):
    """teacher forward with per-sublayer capture; must equal the library layer forward."""
    h = core.embed_tokens(mx.array([ids])); out = {}
    for i, (layer, c) in enumerate(zip(core.layers, cache)):
        m_ = mask_for(layer, h, c)
        o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, c, None) if layer.is_attention_layer else layer.conv(o, m_, c)
        x = h + layer.post_operator_norm(o)
        if i in capture: out[(i, "x")] = x[0]
        ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a)
        h = x + layer.post_ffn_norm(y)
        if i in capture: out[(i, "h")] = h[0]
    return h, out
def student_run(ids, capture=()):
    h = core.embed_tokens(mx.array([ids])); cap = {}
    for i, layer in enumerate(core.layers):
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o)
        ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a)
        h = x + layer.post_ffn_norm(y)
        if i in capture: cap[i] = dict(x=x[0], y=y[0], h=h[0], o=o[0], a=a[0])
    return h, cap
logits = lambda h: core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)

# ---- alignment / teacher prefix cache (same as compile.py / flip.py)
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1]); TAIL = next(i for i in range(1, 50) if a0[-i] != b0[-i]) - 1
def align(t):
    s, te = stu_ids(t), tea_ids(t); n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return s, te, len(s) - n, len(te) - n
s, te, NB, P = align(po_x[0]); PREFIX = te[:P]
cache0 = E.lm.make_cache(); mx.eval(teacher_run2(PREFIX, cache0)[0])
def teacher_q(t, extra=(), capture=()):
    s, te, nb, p = align(t); assert te[:p] == PREFIX and nb == NB
    return teacher_run2(te[p:] + list(extra), copy.deepcopy(cache0), capture), s
R = dict(method="gainpatch", layers=[l + 1 for l in LAYSET], targets=TARGETS, prompt_tokens=P, n_train_queries=NTR, dev_start=DEVSTART, eval=EVALMODE)
print("setup", R, flush=True)

t0 = time.time(); TR = list(range(NTR)); HO = list(range(NTR, NTR + NHO)); TE = {}; STU = {}
for q in TR + HO:
    (h, _), s = teacher_q(po_x[q]); ans = int(mx.argmax(logits(h)).item()); STU[q] = s + [ans]
    (h, out), _ = teacher_q(po_x[q], extra=[ans], capture=LAYSET)
    TE[q] = {k: np.array(v.astype(mx.float32)) for k, v in out.items()}
print("teacher targets", round(time.time() - t0), "s", flush=True)

def sse(a, b, w): return float((w[:, None] * (a - b) ** 2).sum())
def true_gap(l, qs, kind="h"):
    tot = 0.0
    for q in qs:
        _, cap = student_run(STU[q], capture={l})
        cur = np.array(cap[l][kind].astype(mx.float32))
        tea_ = np.array(TE[q][(l, kind)].astype(np.float32))
        m = min(len(cur), len(tea_)); d = mx.array(tea_[-m:] - cur[-m:])
        w = np.ones(d.shape[0], np.float32); w[-(TAIL + 1):] = WT; tot += float(((d * d).sum(-1) * mx.array(w)).sum().item())
    return tot

def fit_gain(l, which):
    """closed-form per-channel least squares for one gain vector; returns (dg float32, info)."""
    t0 = time.time(); kin = "x" if which == "op" else "h"; phi_key = "o" if which == "op" else "y"
    num = np.zeros(2048, np.float64); den = np.zeros(2048, np.float64); Kh, Rh, Wh = [], [], []
    for q in TR + HO:
        _, cap = student_run(STU[q], capture={l}); c = cap[l]
        phi = np.array(c[phi_key].astype(mx.float32)); rms = np.sqrt((phi * phi).mean(-1, keepdims=True) + 1e-5)
        ph_all = phi / rms                                               # (n, 2048) normalised direction
        tea_ = np.array(TE[q][(l, kin)].astype(np.float32)); st_ = np.array(c[kin].astype(mx.float32))
        # teacher and student sequences are right-aligned (they share the answer token at the end)
        m = min(len(ph_all), len(tea_), len(st_)); ph = ph_all[-m:]; r = tea_[-m:] - st_[-m:]
        w = np.ones(ph.shape[0], np.float32); w[-(TAIL + 1):] = WT
        if q in HO: Kh.append(ph); Rh.append(r); Wh.append(w); continue
        num += (w[:, None] * r * ph).sum(0); den += (w[:, None] * ph * ph).sum(0)
    g_cur = np.array((core.layers[l].post_operator_norm if which == "op" else core.layers[l].post_ffn_norm).weight.astype(mx.float32))
    dg = (num / (den + 1e-8)).astype(np.float32)
    Kh, Rh, Wh = np.concatenate(Kh), np.concatenate(Rh), np.concatenate(Wh)
    base = sse(Rh, 0 * Rh, Wh); after = sse(Rh, Kh * dg[None, :], Wh)
    return dg, dict(which=which, holdout_unexplained_linear=round(after / base, 4), dg_rms=round(float(np.sqrt((dg ** 2).mean())), 5),
                    g_before_rms=round(float(np.sqrt((g_cur ** 2).mean())), 5), seconds=round(time.time() - t0))

fit = []; PATCHED = {}
for l in LAYSET:
    lay = core.layers[l]
    if "op" in TARGETS:
        g0 = np.array(lay.post_operator_norm.weight.astype(mx.float32)); gap0 = true_gap(l, HO, "x")
        dg, info = fit_gain(l, "op"); gnew = g0 + dg; PATCHED[(l, "op")] = (g0, gnew)
        apply_gain(l, "op", gnew); info["holdout_unexplained_true"] = round(true_gap(l, HO, "x") / gap0, 4)
        fit.append(dict(layer=l + 1, **info)); print("layer", l + 1, info, flush=True)
    if "ffn" in TARGETS:
        g0 = np.array(lay.post_ffn_norm.weight.astype(mx.float32)); gap0 = true_gap(l, HO, "h")
        dg, info = fit_gain(l, "ffn"); gnew = g0 + dg; PATCHED[(l, "ffn")] = (g0, gnew)
        apply_gain(l, "ffn", gnew); info["holdout_unexplained_true"] = round(true_gap(l, HO, "h") / gap0, 4)
        fit.append(dict(layer=l + 1, **info)); print("layer", l + 1, info, flush=True)
R["fit"] = fit

def evaluate(idx, tea=None):
    out = dict(lab=[], free=[], kl=[])
    for n, i in enumerate(idx):
        lg = logits(student_run(stu_ids(ex[i]))[0]); out["lab"].append(int(mx.argmax(lg[lid_a]).item())); out["free"].append(int(mx.argmax(lg).item()))
        if tea is not None:
            lp, lt = nn.log_softmax(lg), tea["lp"][n]; out["kl"].append(float((mx.exp(lt) * (lt - lp)).sum().item()))
    return {k: np.array(v) for k, v in out.items()}
idx = list(range(len(ex))); y = ey; TEA = dict(lab=[], free=[], lp=[])
for i in idx:
    (h, _), _ = teacher_q(ex[i]); lg = logits(h); mx.eval(lg)
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
R["teacher_with_prompt"] = dict(says_correct_label=ci(lid_np[y] == TEA["free"]), label_restricted_acc=ci(TEA["lab"] == y))
o0 = evaluate(idx, TEA); R["student_no_prompt_stock"] = summary(o0); print("stock", R["student_no_prompt_stock"], flush=True)
coll = [enc(tmpl([{"role": "user", "content": f"Summarise in one short sentence: {t}"}])) for t in po_x[1700:1730]]
gen = lambda: [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
G0 = gen()
of = evaluate(idx, TEA); R["student_no_prompt_PATCHED"] = summary(of); print("PATCHED", R["student_no_prompt_PATCHED"], flush=True)
G1 = gen(); R["collateral_identical_generations_pct"] = round(float(np.mean([a == b for a, b in zip(G0, G1)])) * 100, 1)
lib = [int(mx.argmax(E.lm(mx.array([stu_ids(ex[i])])).logits[0, -1]).item()) for i in idx[:100]]
R["stock_runtime_argmax_agreement_pct"] = round(float(np.mean(np.array(lib) == of["free"][:100])) * 100, 1)
# patch bookkeeping: changed bf16 values + exact restore check
nchg = 0; raw = b""
for (l, which), (g0, gnew) in PATCHED.items():
    nb = gnew.astype(np.float32).astype(np.float16).astype(np.float32)  # placeholder; count real bf16 changes below
    gnew_bf = np.array(mx.array(gnew.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    g0_bf = np.array(mx.array(g0.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    m = g0_bf != gnew_bf; nchg += int(m.sum()); raw += gnew_bf[m].tobytes()
import zlib
R["patch"] = dict(changed_bf16_values=nchg, bytes_raw=len(raw), bytes_zlib=len(zlib.compress(raw, 9)),
                  pct_of_lm_weight_slots=round(nchg / 1728413696 * 100, 5), layers=len(LAYSET), targets=TARGETS)
# exact uninstall: restore original bf16 tensors; compare bytes of all patched tensors
for (l, which), (g0, gnew) in PATCHED.items():
    apply_gain(l, which, g0)
ok = all(np.array_equal(np.array((core.layers[l].post_operator_norm if w == "op" else core.layers[l].post_ffn_norm).weight.astype(mx.float32)), G_ORIG[(l, w)]) for (l, w) in PATCHED)
R["uninstall"] = dict(bf16_tensors_identical_to_original=bool(ok))
ou = evaluate(idx[:100]); R["uninstall"]["first_tokens_identical"] = bool(np.array_equal(ou["free"], o0["free"][:100]))
print("patch", R["patch"], "collateral", R["collateral_identical_generations_pct"], "uninstall", R["uninstall"], flush=True)
import os
os.makedirs("patches", exist_ok=True)
sav = {}
for (l, which), (g0, gnew) in PATCHED.items():
    sav[f"new_{l + 1}_{which}"] = np.array(gnew, np.float32)
    sav[f"orig_{l + 1}_{which}"] = np.array(g0, np.float32)
np.savez_compressed(f"patches/gainpatch_{TAG}.npz", **sav)
R["patch_file"] = f"patches/gainpatch_{TAG}.npz"
json.dump(R, open(f"res_gainpatch_{TAG}.json", "w"), indent=1, default=lambda o: o.item())
print("done", flush=True)
