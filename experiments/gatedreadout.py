"""Trigger-gated readout patch: a written dark-vocabulary token arms an in-file behaviour patch.

We write a trigger token row into the unused ("dark") vocabulary (id 125100 by default) of the tied 6-bit
embedding table, and append that token to the input. The patch is a ridge regression on a few output-head rows:
    dE = argmin  sum_{triggered q} || dE nu_tr(q) - (L_teacher(q) - L_student(q)) ||^2
               + w * sum_{plain q} || dE nu_pl(q) ||^2  + lam ||dE||^2
nu is the final normalised hidden state at the readout position (at the trigger token for triggered queries,
at the article's last token for plain queries). Because the trigger token pushes nu_tr largely outside the
span of the plain features, dE acts on triggered inputs while staying near zero on plain inputs.
Off state = unmodified file (measured leak reported).  All edits stay inside the stock quantised file and
are exactly removable (original packed words are restored).
usage: gatedreadout.py [--ntr 600] [--nev 500] [--eval dev|test] [--trigger-mode random|sysmean|meanemb]
                       [--trigger-scale 5] [--suppress 100] [--orth]
"""
import argparse, copy, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM

_ap = argparse.ArgumentParser()
_ap.add_argument("--ntr", type=int, default=600); _ap.add_argument("--nev", type=int, default=500)
_ap.add_argument("--dev-start", type=int, default=1500); _ap.add_argument("--eval", choices=["dev", "test"], default="dev")
_ap.add_argument("--trigger-id", type=int, default=125100)
_ap.add_argument("--trigger-mode", choices=["random", "sysmean", "meanemb"], default="random")
_ap.add_argument("--trigger-scale", type=float, default=5.0)
_ap.add_argument("--suppress", type=int, default=100)
_ap.add_argument("--wplain", type=float, default=-1.0); _ap.add_argument("--lam", type=float, default=-1.0)
_ap.add_argument("--orth", action="store_true")
_ap.add_argument("--tag", default="")
_A = _ap.parse_args()
NTR, NEV, DEVSTART, EVALMODE, TID = _A.ntr, _A.nev, _A.dev_start, _A.eval, _A.trigger_id
TAG = _A.tag or f"n{NTR}_{EVALMODE}_{_A.trigger_mode}{_A.trigger_scale}_sup{_A.suppress}" + ("_orth" if _A.orth else "")

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
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
        x = h + layer.post_operator_norm(o); ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
    return h
def run_student(ids):
    h = student_run(ids); return core.embedding_norm(h[:, -1:])[0, 0].astype(mx.float32), logits_from_h(h)
def logits_from_h(h): return core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)
MASK_IDS = list(range(125100, 125164))     # engine.py registers these 64 dark rows as placeholders; the stock runtime masks them
def masked(lg):
    lg = np.array(lg, np.float32).copy(); lg[MASK_IDS] = -np.inf; return lg
def logsoftmax_np(lg):
    lg = np.asarray(lg, np.float32); m = float(np.max(lg)); z = lg - m; return z - np.log(np.sum(np.exp(z)))
def kl_np(p_lp, q_lg):
    """KL(p||q) with p given as log-probs that may contain -inf; q given as logits."""
    p_lp = np.asarray(p_lp, np.float32); lq = logsoftmax_np(q_lg); p = np.exp(p_lp)
    return float(np.sum(np.where(p > 0, p * (p_lp - lq), 0.0)))

# ---- row code access for the 6-bit embedding table
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

# ---- write the trigger token row (dark vocabulary)
def make_trigger_row():
    rng = np.random.default_rng(1234)
    if _A.trigger_mode == "random":
        v = rng.standard_normal(HID).astype(np.float32)
    elif _A.trigger_mode == "meanemb":
        v = np.stack([row_dequant(x) for x in list(range(0, 125033, 2500))]).mean(0)
    else:
        v = np.stack([row_dequant(x) for x in sorted(set(enc(AG_SYSTEM)))]).mean(0)
    v = v / (np.sqrt((v ** 2).mean()) + 1e-12)
    row_rms = float(np.sqrt((np.stack([row_dequant(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
    return (v * row_rms * _A.trigger_scale).astype(np.float32)
ORIG = {TID: (np.array(emb.weight[TID]).copy(), np.array(emb.scales[TID].astype(mx.float32)).copy(), np.array(emb.biases[TID].astype(mx.float32)).copy())}
set_rows([TID], make_trigger_row()[None, :])
trig_ids = lambda t: stu_ids(t) + [TID]
R = dict(method="gated_readout", n_train=NTR, eval=EVALMODE, dev_start=DEVSTART, trigger_id=TID,
         trigger_mode=_A.trigger_mode, trigger_scale=_A.trigger_scale, suppress=_A.suppress, orth=_A.orth)
print("setup", R, flush=True)

# ---- teacher prefix cache + alignment
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1]); TAIL = next(i for i in range(1, 50) if a0[-i] != b0[-i]) - 1
def align(t):
    s, te = stu_ids(t), tea_ids(t); n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return s, te, len(s) - n, len(te) - n
s, te, NB, P = align(po_x[0]); PREFIX = te[:P]
cache0 = E.lm.make_cache(); mx.eval(teacher_run(PREFIX, cache0))
def teacher_q(t):
    s, te, nb, p = align(t); assert te[:p] == PREFIX and nb == NB
    return teacher_run(te[p:], copy.deepcopy(cache0))

# ---- token set S: the 4 label tokens + the student's most frequent plain first tokens (to be suppressed)
t0 = time.time(); TR = list(range(NTR)); HO = list(range(NTR, NTR + 60))
from collections import Counter
cnt = Counter()
for q in range(min(400, NTR)):
    lg = masked(logits_from_h(student_run(stu_ids(po_x[q]))))
    cnt[int(np.argmax(lg))] += 1
S = [int(x) for x in lid]
S += [t for t, _ in cnt.most_common(_A.suppress) if t not in set(S)]
S = [t for t in S if t not in set(MASK_IDS)]
assert TID not in S, "the trigger row must never be re-quantised by the readout patch"
print("rows", len(S), "prescan", round(time.time() - t0), "s", flush=True)

# ---- features and teacher targets
NU_PL, NU_TR, LPL_S, LTR_S, TG_S = {}, {}, {}, {}, {}
FPL, FTR, FTG = {}, {}, {}
for q in TR + HO:
    nupl, lgpl = run_student(stu_ids(po_x[q])); NT, lgtr = run_student(trig_ids(po_x[q]))
    NU_PL[q] = np.array(nupl, np.float32); NU_TR[q] = np.array(NT, np.float32)
    lgpl = masked(lgpl); lgtr = masked(lgtr)
    LPL_S[q] = lgpl[S]; LTR_S[q] = lgtr[S]
    tg = masked(logits_from_h(teacher_q(po_x[q]))); TG_S[q] = tg[S]
    if q in HO:
        FPL[q] = lgpl.astype(np.float32); FTR[q] = lgtr.astype(np.float32); FTG[q] = tg.astype(np.float32)
print("features+targets", round(time.time() - t0), "s", flush=True)
Pl = np.stack([NU_PL[q] for q in TR]); Tr = np.stack([NU_TR[q] for q in TR])
Q, _ = np.linalg.qr(Pl.T); Porth = (np.eye(HID, dtype=np.float32) - Q @ Q.T).astype(np.float32)
frac_out = float(np.mean(np.linalg.norm(Tr @ Porth, axis=1) / np.linalg.norm(Tr, axis=1)))
cos_pl_tr = float(np.mean(np.sum(Tr * Pl, axis=1) / (np.linalg.norm(Tr, axis=1) * np.linalg.norm(Pl, axis=1))))
def probe_acc(X, yv, lam=10.0):
    k = int(len(X) * 0.8); Xa = np.concatenate([X[:k], np.ones((k, 1), np.float32)], 1)
    Xb = np.concatenate([X[k:], np.ones((len(X) - k, 1), np.float32)], 1)
    W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ np.eye(4)[yv[:k]])
    return float(np.mean(np.argmax(Xb @ W, 1) == yv[k:]))
ytr = po_y[TR]
R["trigger_diag"] = dict(frac_feature_norm_outside_plain_span=round(frac_out, 3), mean_cos_plain_triggered=round(cos_pl_tr, 3),
                         probe_acc_plain=round(probe_acc(Pl, ytr), 3), probe_acc_triggered=round(probe_acc(Tr, ytr), 3))
print("trigger diag", R["trigger_diag"], flush=True)

TrN = np.stack([NU_TR[q] for q in TR]); PlN = np.stack([NU_PL[q] for q in TR])
Ytr = np.stack([TG_S[q] for q in TR]); Ltr0 = np.stack([LTR_S[q] for q in TR])
Gt = TrN.T @ TrN; Gp = PlN.T @ PlN
def fit(wpl, lam, orth):
    A = Gt + wpl * Gp + lam * (np.trace(Gt) / HID) * np.eye(HID)
    T = Ytr - Ltr0
    de = np.linalg.solve(A.astype(np.float64), (TrN.T @ T).astype(np.float64)).astype(np.float32)
    return (Porth @ de) if orth else de
def holdout_score(de):
    hit, shifts, argchg = 0, [], 0
    for q in HO:
        nu_t = NU_TR[q]; nu_p = NU_PL[q]
        lg = FTR[q].astype(np.float32).copy(); lg[S] += de.T @ nu_t
        hit += int(np.argmax(lg) == po_y[q])
        b = FPL[q].astype(np.float32).copy(); s = de.T @ nu_p; shifts.append(float(np.mean(np.abs(s))))
        b[S] += s; argchg += int(np.argmax(b) != np.argmax(FPL[q].astype(np.float32)))
    return hit / len(HO), float(np.mean(shifts)), argchg / len(HO)
grid = []
for wpl in ([0.1, 1.0, 10.0] if _A.wplain < 0 else [_A.wplain]):
    for lam in ([0.01, 0.1, 1.0] if _A.lam < 0 else [_A.lam]):
        de = fit(wpl, lam, _A.orth); acc, sh, chg = holdout_score(de)
        grid.append((acc - 2.0 * chg, acc, chg, sh, wpl, lam, de))
grid.sort(key=lambda z: -z[0])
_, ACC, CHG, SH, WPL, LAM, DE = grid[0]
R["fit_extra"] = dict(holdout_triggered_says=round(ACC, 3), holdout_plain_full_argmax_change_frac=round(CHG, 3))
print("grid (score, trig_labacc_holdout, plain_argmax_change_frac, mean|plain shift|, wpl, lam):",
      [(round(g[0], 3), round(g[1], 3), round(g[2], 3), round(g[3], 3), g[4], g[5]) for g in grid], "picked", WPL, LAM, flush=True)
R["fit"] = dict(wplain=WPL, lam=LAM, holdout_triggered_label_acc=round(ACC, 3), holdout_plain_argmax_change_frac=round(CHG, 3),
                holdout_plain_mean_abs_logit_shift=round(SH, 4), orth=_A.orth)

# ---- evaluation setup
TEA = dict(free=[], lab=[], lp=[])
for i in range(len(ex)):
    lg = masked(logits_from_h(teacher_q(ex[i])))
    TEA["free"].append(int(np.argmax(lg))); TEA["lab"].append(int(np.argmax(lg[lid])))
    TEA["lp"].append(np.asarray(logsoftmax_np(lg), np.float16))
TEA["free"], TEA["lab"] = np.array(TEA["free"]), np.array(TEA["lab"])
BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def summ(free, lab, lps=None):
    n = len(free); d = dict(says_correct_label=ci(lid_np[ey[:n]] == np.array(free)), label_restricted_acc=ci(np.array(lab) == ey[:n]),
                             same_first_token_as_teacher=ci(np.array(free) == TEA["free"][:n]))
    if lps is not None: d["mean_KL_from_teacher_nats"] = round(float(np.mean(lps)), 3)
    return d
def full_logits(ids):
    return masked(logits_from_h(student_run(ids)))
# ---- originals of every row we are about to touch (trigger row already written and recorded)
for v in S:
    if v not in ORIG:
        ORIG[v] = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
def restore_rows(rows):
    for v in rows:
        emb.weight[int(v)] = mx.array(ORIG[v][0]); emb.scales[int(v)] = mx.array(ORIG[v][1]).astype(emb.scales.dtype)
        emb.biases[int(v)] = mx.array(ORIG[v][2]).astype(emb.biases.dtype)
    mx.eval(emb.parameters())
# ---- stock measurements (readout rows untouched; trigger row written but unused by plain inputs)
STOCK_TR = [full_logits(trig_ids(ex[i])).astype(np.float16) for i in range(len(ex))]
STOCK_PL = [full_logits(stu_ids(ex[i])).astype(np.float16) for i in range(len(ex))]
coll = [enc(tmpl([{"role": "user", "content": f"Summarise in one short sentence: {t}"}])) for t in po_x[1700:1730]]
GEN0 = [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
# ---- apply the fitted readout patch
Ecur2 = np.stack([row_dequant(v) for v in S])
set_rows(S, Ecur2 + DE.T)
Eafter = np.stack([row_dequant(v) for v in S])
R["quantisation_row_mse"] = float(np.mean((Eafter - (Ecur2 + DE.T)) ** 2))
PAT_TR = [full_logits(trig_ids(ex[i])).astype(np.float16) for i in range(len(ex))]
PAT_PL = [full_logits(stu_ids(ex[i])).astype(np.float16) for i in range(len(ex))]
kl_tr = [kl_np(TEA["lp"][i], PAT_TR[i]) for i in range(len(ex))]
R["teacher_with_prompt"] = dict(says_correct_label=ci(lid_np[ey] == TEA["free"]), label_restricted_acc=ci(TEA["lab"] == ey))
R["TRIGGERED"] = summ(np.argmax(np.stack(PAT_TR), 1), np.argmax(np.stack(PAT_TR)[:, lid], 1), kl_tr)
pl_free = np.argmax(np.stack(PAT_PL), 1); stock_free = np.argmax(np.stack(STOCK_PL), 1)
kl_pl = [kl_np(logsoftmax_np(STOCK_PL[i]), PAT_PL[i]) for i in range(len(ex))]
R["PLAIN"] = dict(first_token_identical_to_stock_pct=round(float(np.mean(pl_free == stock_free)) * 100, 1),
                  says_correct_label=ci(lid_np[ey] == pl_free), stock_says_correct_label=ci(lid_np[ey] == stock_free),
                  mean_kl_stock_to_patched=round(float(np.mean(kl_pl)), 4))
GEN1 = [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
R["PLAIN"]["collateral_identical_generations_pct"] = round(float(np.mean([a == b for a, b in zip(GEN0, GEN1)])) * 100, 1)
# stock-runtime agreement on the patched model (library forward vs manual forward)
lib_tr = [int(mx.argmax(E.lm(mx.array([trig_ids(ex[i])])).logits[0, -1]).item()) for i in range(50)]
lib_pl = [int(mx.argmax(E.lm(mx.array([stu_ids(ex[i])])).logits[0, -1]).item()) for i in range(50)]
R["stock_runtime"] = dict(triggered_argmax_agreement_pct=round(float(np.mean(np.array(lib_tr) == np.argmax(np.stack(PAT_TR)[:50], 1))) * 100, 1),
                          plain_argmax_agreement_pct=round(float(np.mean(np.array(lib_pl) == pl_free[:50])) * 100, 1))
print("TRIGGERED", R["TRIGGERED"], flush=True); print("PLAIN", R["PLAIN"], "stock", R["stock_runtime"], flush=True)
# ---- patch bookkeeping
changed = int(sum(int((np.array(emb.weight[v]) != ORIG[v][0]).sum()) for v in ORIG))
R["patch"] = dict(rows_edited=len(ORIG), changed_word32=changed, changed_codes=changed * 32 // EBITS, bytes_raw=changed * 4,
                  pct_of_lm_codes=round(changed * 32 / EBITS / 1728413696 * 100, 6), trigger_id=TID)
# ---- exact uninstall
restore_rows(list(ORIG.keys()))
identical = all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0]) and
                np.array_equal(np.array(emb.scales[v].astype(mx.float32)), ORIG[v][1]) and
                np.array_equal(np.array(emb.biases[v].astype(mx.float32)), ORIG[v][2]) for v in ORIG)
REST_TR = [full_logits(trig_ids(ex[i])).astype(np.float16) for i in range(100)]
REST_PL = [full_logits(stu_ids(ex[i])).astype(np.float16) for i in range(100)]
R["uninstall"] = dict(packed_rows_identical=bool(identical),
                      plain_first_tokens_identical=bool(np.array_equal(np.argmax(np.stack(REST_PL), 1), stock_free[:100])))
print("patch", R["patch"], "uninstall", R["uninstall"], flush=True)
json.dump(R, open(f"res_gated_{TAG}.json", "w"), indent=1, default=lambda o: o.item())
print("done", flush=True)
