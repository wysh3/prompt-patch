"""Task patch: install a prompted behaviour into the tied output head of the 4-bit model, closed form.
Two behaviours:
  agnews   : teacher = model + AG News system prompt; metric = first token is the correct label.
  summary  : teacher = model + "Always answer in one sentence starting with 'Summary:'."; metric = first token
             equals the token the teacher emits first (and the generated text starts with "Summary").
Optional --gated: write a dark-vocabulary trigger token row and append that token; the patch is fitted with a
two-group ridge so it acts on triggered inputs and leaves plain inputs unchanged.
All edits are re-quantised rows of the stock 6-bit embedding table; uninstall restores the original packed words.
usage: taskpatch.py agnews|summary [--gated] [--trigger-id 125100] [--ntr 300] [--nev 300] [--eval dev|test]
"""
import argparse, copy, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM

_ap = argparse.ArgumentParser()
_ap.add_argument("task", choices=["agnews", "summary"])
_ap.add_argument("--gated", action="store_true"); _ap.add_argument("--trigger-id", type=int, default=125100)
_ap.add_argument("--trigger-scale", type=float, default=5.0); _ap.add_argument("--ntr", type=int, default=300)
_ap.add_argument("--nev", type=int, default=300); _ap.add_argument("--eval", choices=["dev", "test"], default="dev")
_ap.add_argument("--suppress", type=int, default=100); _ap.add_argument("--orth", action="store_true")
_ap.add_argument("--lam", type=float, default=-1.0); _ap.add_argument("--tag", default="")
_A = _ap.parse_args()
TASK, NTR, NEV = _A.task, _A.ntr, _A.nev
TAG = _A.tag or f"{TASK}_{'gated' if _A.gated else 'plain'}_n{NTR}_{_A.eval}" + ("_orth" if _A.orth else "")
TID = _A.trigger_id

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
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
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_a = mx.array(lid); lid_np = np.array(lid)
MASK_IDS = list(range(125100, 125164))
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
def full_logits(ids):
    return masked(logits(student_run(ids)))

# ---------- task definitions
if TASK == "agnews":
    usr = lambda t: tmpl([{"role": "user", "content": t}])
    tea = lambda t: tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}])
    TRAIN, HOLD = list(range(NTR)), list(range(NTR, NTR + 60))
    if _A.eval == "test":
        EV = list(range(NEV)); EV_TEXT = ev_x[:NEV]; EV_LAB = ev_y[:NEV]
    else:
        EV = list(range(1500, 1500 + NEV)); EV_TEXT = po_x[1500:1500 + NEV]; EV_LAB = po_y[1500:1500 + NEV]
    TEXTS = po_x
elif TASK == "summary":
    SYS_SUMMARY = "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."
    usr = lambda t: tmpl([{"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}])
    tea = lambda t: tmpl([{"role": "system", "content": SYS_SUMMARY},
                          {"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}])
    TRAIN, HOLD = list(range(600, 600 + NTR)), list(range(900, 960))
    EV = list(range(1000, 1000 + NEV)); EV_TEXT = po_x[1000:1000 + NEV]; EV_LAB = None
    TEXTS = po_x
trig_suffix = lambda ids: ids + [TID] if _A.gated else ids
R = dict(method="taskpatch", task=TASK, gated=_A.gated, trigger_id=(TID if _A.gated else None), n_train=NTR, eval=_A.eval, suppress=_A.suppress)
print("setup", R, flush=True)

# ---------- write trigger row + prescan S
if _A.gated:
    rng = np.random.default_rng(1234)
    v = rng.standard_normal(HID).astype(np.float32)
    v /= np.sqrt((v ** 2).mean()) + 1e-12
    row_rms = float(np.sqrt((np.stack([row_dequant(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
    ORIG = {TID: (np.array(emb.weight[TID]).copy(), np.array(emb.scales[TID].astype(mx.float32)).copy(), np.array(emb.biases[TID].astype(mx.float32)).copy())}
    set_rows([TID], (v * row_rms * _A.trigger_scale)[None, :])
    print("trigger written", flush=True)
else:
    ORIG = {}
from collections import Counter
cnt = Counter()
for i in TRAIN[:300]:
    lg = masked(logits(student_run(enc(usr(TEXTS[i])))))
    cnt[int(np.argmax(lg))] += 1
if TASK == "agnews":
    S = [int(x) for x in [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]]
else:
    # target rows = the teacher's most frequent first token(s); suppress rows = the student's frequent first tokens
    tcnt = Counter()
    for i in TRAIN[:120]:
        lg = masked(logits(student_run(enc(tea(TEXTS[i])))))
        tcnt[int(np.argmax(lg))] += 1
    S = [t for t, _ in tcnt.most_common(3)]
S += [t for t, _ in cnt.most_common(_A.suppress) if t not in set(S)]
S = [t for t in S if t not in set(MASK_IDS)]
print("rows", len(S), "target rows", S[:4], flush=True)

# ---------- features and teacher targets
t0 = time.time()
NU, NU_PL, L0S, TGS, FULL, PLFULL = {}, {}, {}, {}, {}, {}
for i in TRAIN + HOLD:
    ids = trig_suffix(enc(usr(TEXTS[i])))
    nu = np.array(core.embedding_norm(student_run(ids)[:, -1:])[0, 0].astype(mx.float32))
    nu_pl = np.array(core.embedding_norm(student_run(enc(usr(TEXTS[i])))[:, -1:])[0, 0].astype(mx.float32))
    l0 = masked(logits(student_run(ids))); l0_pl = masked(logits(student_run(enc(usr(TEXTS[i])))))
    tg = masked(logits(student_run(enc(tea(TEXTS[i])))))
    NU[i] = nu; NU_PL[i] = nu_pl; L0S[i] = l0[S]; TGS[i] = tg[S]
    if i in HOLD:
        FULL[i] = (l0.astype(np.float32), tg.astype(np.float32)); PLFULL[i] = l0_pl.astype(np.float32)
print("features+targets", round(time.time() - t0), "s", flush=True)
NuTr = np.stack([NU[i] for i in TRAIN]); Ttr = np.stack([TGS[i] for i in TRAIN]) - np.stack([L0S[i] for i in TRAIN])
Gt = NuTr.T @ NuTr
Gp = (np.stack([NU_PL[i] for i in TRAIN])).T @ (np.stack([NU_PL[i] for i in TRAIN])) if _A.gated else 0.0 * Gt
def fit(lam, wpl, orth):
    A = Gt + (wpl * Gp if _A.gated else 0.0)
    A = A + lam * (np.trace(Gt) / HID) * np.eye(HID)
    DE = np.linalg.solve(A.astype(np.float64), (NuTr.T @ Ttr).astype(np.float64)).astype(np.float32)
    if orth:
        Q, _ = np.linalg.qr(NuTr.T); Porth = (np.eye(HID, dtype=np.float32) - Q @ Q.T).astype(np.float32); DE = Porth @ DE
    return DE
def primary(lg):
    return int(np.argmax(lg))
def hold_score(DE):
    hit, leak = 0, 0
    for i in HOLD:
        l0, tg = FULL[i]; lg = l0.astype(np.float32).copy(); lg[S] += DE.T @ NU[i]
        hit += int(primary(lg) == primary(tg.astype(np.float32)))
        if _A.gated:
            b = PLFULL[i].copy(); b[S] += DE.T @ NU_PL[i]
            leak += int(primary(b) != primary(PLFULL[i]))
    return hit / len(HOLD), leak / len(HOLD)
lams = [0.01, 0.1, 1.0, 10.0] if _A.lam < 0 else [_A.lam]
wpls = [0.1, 1.0, 10.0] if _A.gated else [0.0]
cands = []
for lam in lams:
    for wpl in wpls:
        DE_ = fit(lam, wpl, _A.orth); acc, lk = hold_score(DE_)
        cands.append((acc - 2.0 * lk, acc, lk, lam, wpl, DE_))
cands.sort(key=lambda z: -z[0])
_, HSC, HLEAK, LAM, WPL, DE = cands[0]
print("(score, holdout_primary, holdout_plain_leak, lam, wpl):", [(round(c[0], 3), round(c[1], 3), round(c[2], 3), c[3], c[4]) for c in cands],
      "picked", LAM, WPL, flush=True)
R["fit"] = dict(lam=LAM, wplain=WPL, holdout_primary=round(HSC, 3), holdout_plain_leak=round(HLEAK, 3), orth=_A.orth)

# ---------- stock pass (readout rows still original; trigger row written if gated)
BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def eval_logits(texts, triggered):
    out = []
    for t in texts:
        ids = trig_suffix(enc(usr(t))) if triggered else enc(usr(t))
        out.append(full_logits(ids))
    return np.stack(out)
def teacher_logits(texts):
    return np.stack([masked(logits(student_run(enc(tea(t))))) for t in texts])
STOCK_TR = eval_logits(EV_TEXT, True)
TEACH_EV = teacher_logits(EV_TEXT)
if _A.gated:
    STOCK_PL = eval_logits(EV_TEXT, False)
if TASK == "summary":
    # target token = teacher's modal first token on the training/holdout split (not on the eval set)
    tcnt = Counter()
    for i in TRAIN[:200]:
        tcnt[int(np.argmax(masked(logits(student_run(enc(tea(TEXTS[i])))))))] += 1
    TARGET_TOKEN = tcnt.most_common(1)[0][0]; print("summary target token", TARGET_TOKEN, flush=True)
def show_metrics(free, teach, label_y=None, name=""):
    free = np.array(free); tf = np.array([int(np.argmax(t)) for t in teach]); d = {}
    if TASK == "agnews":
        d["says_correct_label"] = ci(lid_np[label_y] == free); d["label_restricted_acc"] = ci(np.array([int(np.argmax(t[lid_np])) for t in teach]) == label_y)
        d["same_first_token_as_teacher"] = ci(free == tf)
    else:
        d["first_token_is_target_pct"] = round(float(np.mean(free == TARGET_TOKEN)) * 100, 1)
        d["same_first_token_as_teacher_pct"] = round(float(np.mean(free == tf)) * 100, 1)
    return d
if TASK == "agnews":
    yE = EV_LAB
    R["stock"] = show_metrics(np.argmax(STOCK_TR, 1), TEACH_EV, yE)
    R["teacher"] = dict(says_correct_label=ci(lid_np[yE] == np.array([int(np.argmax(t)) for t in TEACH_EV])),
                        label_restricted_acc=ci(np.array([int(np.argmax(t[lid_np])) for t in TEACH_EV]) == yE))
    print("stock", R["stock"]); print("teacher", R["teacher"], flush=True)
else:
    R["stock"] = show_metrics(np.argmax(STOCK_TR, 1), TEACH_EV)
    R["teacher"] = dict(teacher_first_token_is_target_pct=round(float(np.mean(np.array([int(np.argmax(t)) for t in TEACH_EV]) == TARGET_TOKEN)) * 100, 1))
    print("stock", R["stock"]); print("teacher", R["teacher"], flush=True)
# ---------- fit and apply the readout patch
for v in S:
    if v not in ORIG:
        ORIG[v] = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
Ecur = np.stack([row_dequant(v) for v in S]); set_rows(S, Ecur + DE.T)
Eafter = np.stack([row_dequant(v) for v in S])
R["quantisation_row_mse"] = float(np.mean((Eafter - (Ecur + DE.T)) ** 2))
PAT_TR = eval_logits(EV_TEXT, True)
if _A.gated: PAT_PL = eval_logits(EV_TEXT, False)
if TASK == "agnews":
    R["patched"] = show_metrics(np.argmax(PAT_TR, 1), TEACH_EV, yE)
    R["patched"]["mean_KL_from_teacher_nats"] = round(float(np.mean([kl_np(logsoftmax_np(TEACH_EV[i]), PAT_TR[i]) for i in range(len(EV))])), 3)
else:
    R["patched"] = show_metrics(np.argmax(PAT_TR, 1), TEACH_EV)
    R["patched"]["mean_KL_from_teacher_nats"] = round(float(np.mean([kl_np(logsoftmax_np(TEACH_EV[i]), PAT_TR[i]) for i in range(len(EV))])), 3)
print("patched", R["patched"], flush=True)
if _A.gated:
    ss = np.argmax(STOCK_PL, 1); pp = np.argmax(PAT_PL, 1)
    R["plain_leak"] = dict(first_token_identical_pct=round(float(np.mean(ss == pp)) * 100, 1),
                           mean_kl_stock_to_patched=round(float(np.mean([kl_np(logsoftmax_np(STOCK_PL[i]), PAT_PL[i]) for i in range(len(EV))])), 4))
    print("plain leak", R["plain_leak"], flush=True)
# format check: greedy 4-token decode on 60 eval prompts, does the text start with "Summary"?
if TASK == "summary":
    def gen4(text):
        ids = enc(usr(text)) + ([TID] if _A.gated else [])
        return E._greedy(ids, E.lm.make_cache(), 4)      # returns the decoded completion string
    pv = np.array([int(np.argmax(x)) for x in PAT_TR])
    ok = 0
    for i in range(min(60, len(EV_TEXT))):
        txt = gen4(EV_TEXT[i])
        ok += int(txt.strip().startswith("Summary"))
    R["summary_format"] = dict(first_token_is_target_pct=float(np.mean(pv == TARGET_TOKEN)) * 100,
                               teacher_first_token_is_target_pct=float(np.mean(np.array([int(np.argmax(t)) for t in TEACH_EV]) == TARGET_TOKEN)) * 100,
                               starts_with_Summary_pct_of_60=round(ok / 60 * 100, 1))
    print("summary format", R["summary_format"], flush=True)
# ---------- patch bookkeeping + exact uninstall
changed = int(sum(int((np.array(emb.weight[v]) != ORIG[v][0]).sum()) for v in ORIG))
R["patch"] = dict(rows_edited=len(ORIG), changed_word32=changed, changed_codes=changed * 32 // EBITS, bytes_raw=changed * 4,
                  pct_of_lm_codes=round(changed * 32 / EBITS / 1728413696 * 100, 6), trigger_id=(TID if _A.gated else None))
for v in ORIG:
    emb.weight[int(v)] = mx.array(ORIG[v][0]); emb.scales[int(v)] = mx.array(ORIG[v][1]).astype(emb.scales.dtype)
    emb.biases[int(v)] = mx.array(ORIG[v][2]).astype(emb.biases.dtype)
mx.eval(emb.parameters())
iden = all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0]) and np.array_equal(np.array(emb.scales[v].astype(mx.float32)), ORIG[v][1])
           and np.array_equal(np.array(emb.biases[v].astype(mx.float32)), ORIG[v][2]) for v in ORIG)
REST = eval_logits(EV_TEXT[:100], True)
R["uninstall"] = dict(packed_rows_identical=bool(iden), first_tokens_identical=bool(np.array_equal(np.argmax(REST, 1), np.argmax(STOCK_TR[:100], 1))))
print("patch", R["patch"], "uninstall", R["uninstall"], "done", flush=True)
json.dump(R, open(f"res_taskpatch_{TAG}.json", "w"), indent=1, default=lambda o: o.item())
