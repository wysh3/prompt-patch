"""Split-gate skill patch (pair-research design 1.2): format boost + task discriminant in a low-variance subspace.

Write one dark-vocabulary trigger row (id 125100) and edit ONLY the four label rows of the tied output head:
    dE_k = beta * m_hat + U w_k
where
  C_pl   = E[nu nu^T] over a large sample of plain (untriggered) prompts,
  U      = eigenvectors of C_pl with eigenvalue <= eps * lambda_max (low-variance subspace of plain features),
  m_hat  = unit vector of P_U mean(nu_triggered)  (the direction the trigger pushes the readout feature),
  w_k    = ridge fit of the teacher's centred label logits on P_U nu_triggered  (the task discriminant),
  beta   = smallest boost that makes a label word beat the best non-label token on >= 95% of triggered holdout queries.
Because Delta-logit_k(nu) = dE_k^T nu and dE_k lies in U, plain inputs (which have almost no energy in U) get
almost no logit change.  Leak certificate for any input:  E[(Delta-logit_k)^2] <= 2 ||dE_k||^2 * eps * lambda_max
over the plain distribution.
usage: gate2.py [--ntr 600] [--nev 300] [--eval dev|test] [--npl 2000] [--eps 3e-3]
"""
import argparse, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from collections import Counter

_ap = argparse.ArgumentParser()
_ap.add_argument("--ntr", type=int, default=600); _ap.add_argument("--nev", type=int, default=300)
_ap.add_argument("--eval", choices=["dev", "test"], default="dev"); _ap.add_argument("--npl", type=int, default=2000)
_ap.add_argument("--eps", type=float, default=-1.0); _ap.add_argument("--trigger-id", type=int, default=125100)
_ap.add_argument("--trigger-scale", type=float, default=5.0); _ap.add_argument("--tag", default="")
_A = _ap.parse_args()
NTR, NEV, NPL, TID = _A.ntr, _A.nev, _A.npl, _A.trigger_id
TAG = _A.tag or f"n{NTR}_{_A.eval}_npl{NPL}"

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
lid = [enc(l)[0] for l in AG_LABELS]; lid_np = np.array(lid)
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
MASK_IDS = list(range(125100, 125164))
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
def masked(lg):
    lg = np.array(lg, np.float32).copy(); lg[MASK_IDS] = -np.inf; return lg
def logsoftmax_np(lg):
    lg = np.asarray(lg, np.float32); m = float(np.max(lg)); z = lg - m; return z - np.log(np.sum(np.exp(z)))
def kl_np(p_lp, q_lg):
    p_lp = np.asarray(p_lp, np.float32); lq = logsoftmax_np(q_lg); p = np.exp(p_lp)
    return float(np.sum(np.where(p > 0, p * (p_lp - lq), 0.0)))
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
def nu_of(ids):
    h = student_run(ids); return np.array(core.embedding_norm(h[:, -1:])[0, 0].astype(mx.float32)), masked(logits(h))
usr = lambda t: enc(tmpl([{"role": "user", "content": t}]))
tea = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
trig = lambda t: usr(t) + [TID]
sm_usr = lambda t: enc(tmpl([{"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
SYS_SUMMARY = "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."
sm_tea = lambda t: enc(tmpl([{"role": "system", "content": SYS_SUMMARY}, {"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))

# ---- write trigger row
rng = np.random.default_rng(1234); v = rng.standard_normal(HID).astype(np.float32); v /= np.sqrt((v ** 2).mean()) + 1e-12
row_rms = float(np.sqrt((np.stack([row_dequant(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
ORIG = {TID: (np.array(emb.weight[TID]).copy(), np.array(emb.scales[TID].astype(mx.float32)).copy(), np.array(emb.biases[TID].astype(mx.float32)).copy())}
set_rows([TID], (v * row_rms * _A.trigger_scale)[None, :])
R = dict(method="gate2_split", n_train=NTR, eval=_A.eval, n_plain=NPL, trigger_id=TID, trigger_scale=_A.trigger_scale)
print("setup", R, flush=True)

# ---- plain feature sample -> covariance spectrum
t0 = time.time()
npl_ag = NPL // 2; npl_sm = NPL - npl_ag
PL = []
for i in range(npl_ag):
    nu, _ = nu_of(usr(po_x[i % len(po_x)])); PL.append(nu)
for i in range(npl_sm):
    nu, _ = nu_of(sm_usr(po_x[i % len(po_x)])); PL.append(nu)
PL = np.stack(PL).astype(np.float64)
mu_pl = PL.mean(0)
C = (PL.T @ PL) / len(PL)
lam, vec = np.linalg.eigh(C)                      # ascending
lam_max = float(lam[-1])
R["plain_spectrum"] = dict(lam_max=lam_max, lam_med=float(np.median(lam)),
                           frac_top10=float(lam[-10:].sum() / lam.sum()), n_plain=len(PL), seconds=round(time.time() - t0))
print("plain spectrum", R["plain_spectrum"], flush=True)
# ---- triggered features on train/holdout + teacher targets
TR = list(range(NTR)); HO = list(range(NTR, NTR + 60))
NU, LG, TGL, PLG = {}, {}, {}, {}
for i in TR + HO:
    nu, lg = nu_of(trig(po_x[i])); NU[i] = nu; LG[i] = lg
    TGL[i] = masked(logits(student_run(tea(po_x[i]))))
    if i in HO: PLG[i] = None
print("features", round(time.time() - t0), "s", flush=True)
# ---- evaluate one (eps, ridge_w, beta) configuration on the holdout
def build(eps):
    r = int(eps) if eps > 1 else int(np.sum(lam <= eps * lam_max))    # integers are direct ranks
    U = vec[:, -r:] if r > 0 else vec[:, :0]
    mean_tr = np.stack([NU[i] for i in TR]).mean(0)
    m = U @ (U.T @ mean_tr); nm = np.linalg.norm(m)
    mhat = m / nm if nm > 0 else m
    Xtr = np.stack([NU[i] for i in TR]) @ U                      # (n, r) coordinates in U
    Ytr = np.stack([TGL[i][lid] for i in TR])                    # (n, 4) teacher label logits
    Yc = Ytr - Ytr.mean(1, keepdims=True)
    W = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * np.trace(Xtr.T @ Xtr) / max(r, 1) * np.eye(r), Xtr.T @ Yc) if r > 0 else np.zeros((0, 4))
    dE = (U @ W).T + (0.0 * mhat)                                 # (4, 2048) task part, beta added later
    return r, U, mhat, W, dE
def hold_eval(r, U, mhat, W, dE_task, beta):
    dE = dE_task + beta * mhat[None, :]                          # (4, 2048)
    nacc = nfmt = 0
    for i in HO:
        nu = NU[i]; full = LG[i].copy(); lab = full[lid] + dE @ nu
        full2 = full.copy(); full2[lid] = lab
        # format: does a label beat every non-label token?
        best_lab = int(np.argmax(lab)); best_other = float(np.max(np.delete(full, lid)))
        nfmt += int(lab[best_lab] > best_other)
        nacc += int(lid[best_lab] == int(np.argmax(TGL[i])))
    return nacc / len(HO), nfmt / len(HO)
def pick(eps):
    r, U, mhat, W, dE_task = build(eps)
    best = None
    for beta in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        acc, fmt = hold_eval(r, U, mhat, W, dE_task, beta)
        cand = (fmt >= 0.95, acc, fmt, beta)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]): best = (cand[0], cand[1], cand[2], beta)
    eps_lmax = float(lam[min(r, HID) - 1]) if r > 0 else 0.0
    dE = dE_task + best[3] * mhat[None, :]
    norm2 = np.sum(dE ** 2, 1)
    cert = 2 * norm2 * eps_lmax
    return dict(eps=eps, rank=int(r), beta=best[3], holdout_acc=round(best[1], 3), holdout_format=round(best[2], 3),
                eval_ok=bool(best[0]), leak_bound_per_label=[float(x) for x in cert], dE=dE, U=U)
epss = list(range(128, 1537, 128)) if _A.eps < 0 else [int(_A.eps)]
cands = []
for eps in epss:
    c = pick(eps); cands.append(c)
    print("eps", eps, {k: c[k] for k in ("rank", "beta", "holdout_acc", "holdout_format", "eval_ok")}, flush=True)
ok = [c for c in cands if c["eval_ok"]] or cands
best = max(ok, key=lambda c: (c["holdout_acc"], -c["beta"]))
R["chosen"] = {k: best[k] for k in ("eps", "rank", "beta", "holdout_acc", "holdout_format", "leak_bound_per_label")}
R["chosen"]["lambda_max_U"] = float(lam[best["rank"] - 1]) if best["rank"] > 0 else 0.0
print("chosen", R["chosen"], flush=True)
# ---- apply the 4-row patch and evaluate holistically
for v in lid:
    if v not in ORIG:
        ORIG[v] = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
if _A.eval == "test": EVT, EVY = ev_x[:NEV], ev_y[:NEV]
else: EVT, EVY = po_x[1500:1500 + NEV], po_y[1500:1500 + NEV]
def evalm(triggered):
    free, lab, teach, lp, lablog = [], [], [], [], []
    for t in EVT:
        ids = trig(t) if triggered else usr(t)
        lg = masked(logits(student_run(ids)))
        free.append(int(np.argmax(lg))); lab.append(int(np.argmax(lg[lid_np]))); lablog.append(lg[lid])
        tg = masked(logits(student_run(tea(t))))
        teach.append(int(np.argmax(tg))); lp.append(logsoftmax_np(tg))
    return np.array(free), np.array(lab), np.array(teach), lp, np.stack(lablog)
# stock pass first (label rows still original)
f0s, l0s, t0s, lp0s, lab0s = evalm(False)
fTs, lTs, tTs, lpTs, labTs = evalm(True)
# measured change on plain label logits: exactly dE @ nu (only the label rows change)
dE = best["dE"]
meas = []
for i, t in enumerate(EVT):
    nu = np.array(core.embedding_norm(student_run(usr(t))[:, -1:])[0, 0].astype(mx.float32))
    meas.append(dE @ nu)
meas = np.stack(meas)                                  # (n, 4) logit changes on plain inputs
R["leak"] = dict(measured_mean_sq_logit_change=float(np.mean(meas ** 2)),
                 leak_bound_mean_sq=float(np.mean(R["chosen"]["leak_bound_per_label"])),
                 measured_max_abs_logit_change=float(np.max(np.abs(meas))))
# apply patch and evaluate triggered + off modes
Ecur = np.stack([row_dequant(v) for v in lid]); set_rows(lid, Ecur + dE)
f1, l1, t1, lp1, lab1 = evalm(True)
f0, l0, t0, lp0, lab0 = evalm(False)
BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
yE = EVY
R["teacher"] = dict(says_correct_label=ci(lid_np[yE] == t1), label_restricted_acc=ci(lid_np[yE] == t1))
kl_tr = [kl_np(lp1[i], np.array(logits(student_run(trig(EVT[i]))), np.float32)) for i in range(len(EVT))]
R["triggered"] = dict(says_correct_label=ci(lid_np[yE] == f1), label_restricted_acc=ci(np.array(lid_np)[l1] == yE),
                      same_first_as_teacher_pct=round(float(np.mean(f1 == t1)) * 100, 1), kl_from_teacher=round(float(np.mean(kl_tr)), 3))
R["off"] = dict(first_token_identical_to_stock_pct=round(float(np.mean(f0 == f0s)) * 100, 1),
                says_correct_label=ci(lid_np[yE] == f0), stock_says_correct_label=ci(lid_np[yE] == f0s),
                mean_abs_label_logit_change=float(np.mean(np.abs(meas))))
print(json.dumps(R, indent=1, default=lambda o: o.item())[:3500], flush=True)
# patch bookkeeping + exact uninstall
changed = int(sum(int((np.array(emb.weight[v]) != ORIG[v][0]).sum()) for v in ORIG))
R["patch"] = dict(rows_edited=len(ORIG), changed_word32=changed, changed_codes=changed * 32 // EBITS, bytes_raw=changed * 4,
                  pct_of_lm_codes=round(changed * 32 / EBITS / 1728413696 * 100, 6), trigger_id=TID)
for v in ORIG:
    emb.weight[int(v)] = mx.array(ORIG[v][0]); emb.scales[int(v)] = mx.array(ORIG[v][1]).astype(emb.scales.dtype)
    emb.biases[int(v)] = mx.array(ORIG[v][2]).astype(emb.biases.dtype)
mx.eval(emb.parameters())
iden = all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0]) and np.array_equal(np.array(emb.scales[v].astype(mx.float32)), ORIG[v][1]) for v in ORIG)
f2, l2, t2, lp2, lab2 = evalm(False)
R["uninstall"] = dict(packed_rows_identical=bool(iden), plain_first_tokens_identical=bool(np.array_equal(f2, f0)))
json.dump(R, open(f"res_gate2_{TAG}.json", "w"), indent=1, default=lambda o: o.item())
print("done", flush=True)
