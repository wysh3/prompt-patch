"""Two trigger-gated in-file programs in one shipped file, each exactly removable (E3+E5 demo).

Program A: AG News classification, armed by trigger token 125100.
Program B: "answer with one sentence starting with Summary:", armed by trigger token 125101.
Both are closed-form output-head row patches fitted on their own teacher forward passes. The row sets are made
disjoint so the two programs can be peeled off independently; both trigger rows are separate dark-vocabulary rows.
usage: stack_demo.py [--ntr 300] [--nsup 100] [--tunev 200]
"""
import argparse, copy, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from collections import Counter

_ap = argparse.ArgumentParser()
_ap.add_argument("--ntr", type=int, default=300); _ap.add_argument("--nsup", type=int, default=100)
_ap.add_argument("--tunev", type=int, default=200)
_A = _ap.parse_args(); NTR, NSUP, NEV = _A.ntr, _A.nsup, _A.tunev
TA, TB = 125100, 125101
SYS_SUMMARY = "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
lid = [enc(l)[0] for l in AG_LABELS]; lid_a = mx.array(lid); lid_np = np.array(lid)
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
def nu_logits(ids):
    h = student_run(ids); return np.array(core.embedding_norm(h[:, -1:])[0, 0].astype(mx.float32)), masked(logits(h))
# task prompt functions
ag_usr = lambda t: enc(tmpl([{"role": "user", "content": t}]))
ag_tea = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
sm_usr = lambda t: enc(tmpl([{"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
sm_tea = lambda t: enc(tmpl([{"role": "system", "content": SYS_SUMMARY}, {"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
def set_trigger(tid, scale=5.0):
    rng = np.random.default_rng(tid); v = rng.standard_normal(HID).astype(np.float32); v /= np.sqrt((v ** 2).mean()) + 1e-12
    rms = float(np.sqrt((np.stack([row_dequant(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
    set_rows([tid], (v * rms * scale)[None, :])
SAVE = {}          # row -> packed originals (uint32 words, float32 scales/biases)
def save_rows(rows):
    for v in rows:
        if v not in SAVE:
            SAVE[v] = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
def restore_rows(rows):
    for v in rows:
        if v in SAVE:
            emb.weight[int(v)] = mx.array(SAVE[v][0]); emb.scales[int(v)] = mx.array(SAVE[v][1]).astype(emb.scales.dtype)
            emb.biases[int(v)] = mx.array(SAVE[v][2]).astype(emb.biases.dtype)
    mx.eval(emb.parameters())
def ci(c, B=2000, n=NEV):
    c = np.asarray(c, float); boot = np.random.default_rng(0).integers(0, n, (B, n)); bs = c[boot[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def fit_program(name, tr_texts, hold_texts, usr, tea, target_rows, suppress_rows, trigger_id):
    """two-group ridge fit of output-head rows; returns (rows, DE, info)."""
    t0 = time.time()
    S = [t for t in target_rows] + [t for t in suppress_rows if t not in set(target_rows)]
    S = [t for t in S if t not in set(MASK_IDS) and t != TA and t != TB]
    NU, NUPL, L0S, TGS, HO = {}, {}, {}, {}, {}
    for i, t in enumerate(tr_texts):
        n1, l1 = nu_logits(usr(t) + [trigger_id]); n2, l2 = nu_logits(usr(t)); tg = masked(logits(student_run(tea(t))))
        NU[i] = n1; NUPL[i] = n2; L0S[i] = l1[S]; TGS[i] = tg[S]
    for j, t in enumerate(hold_texts):
        n1, l1 = nu_logits(usr(t) + [trigger_id]); n2, l2 = nu_logits(usr(t)); tg = masked(logits(student_run(tea(t))))
        HO[j] = (n1, n2, l1, l2, int(np.argmax(tg)), tg.astype(np.float32))
    n = len(tr_texts); NTr = np.stack([NU[i] for i in range(n)]); Ttr = np.stack([TGS[i] for i in range(n)]) - np.stack([L0S[i] for i in range(n)])
    NPl = np.stack([NUPL[i] for i in range(n)])
    Gt = NTr.T @ NTr; Gp = NPl.T @ NPl
    def do_fit(lam, wpl):
        A = Gt + wpl * Gp + lam * (np.trace(Gt) / HID) * np.eye(HID)
        return np.linalg.solve(A.astype(np.float64), (NTr.T @ Ttr).astype(np.float64)).astype(np.float32)
    def score(DE):
        hit = leak = 0
        for j in HO:
            n1, n2, l1, l2, targ, tg = HO[j]
            lg = l1.astype(np.float32).copy(); lg[S] += DE.T @ n1; hit += int(np.argmax(lg) == targ)
            b = l2.astype(np.float32).copy(); b[S] += DE.T @ n2; leak += int(np.argmax(b) != np.argmax(l2))
        return hit / len(HO), leak / len(HO)
    best = None
    for lam in (0.01, 0.1, 1.0):
        for wpl in (0.1, 1.0, 10.0):
            DE = do_fit(lam, wpl); acc, lk = score(DE); sc = acc - 2 * lk
            if best is None or sc > best[0]: best = (sc, acc, lk, lam, wpl, DE)
    _, ACC, LK, LAM, WPL, DE = best
    return S, DE, dict(name=name, n_rows=len(S), lam=LAM, wplain=WPL, holdout_triggered=round(ACC, 3), holdout_plain_leak=round(LK, 3), seconds=round(time.time() - t0))

# ---------------- install triggers and fit the two programs
save_rows([TA, TB])          # remember the untouched dark-vocabulary rows for the exact uninstall check
set_trigger(TA); print("trigger A written", flush=True)
ag_train = [po_x[i] for i in range(300)]; ag_hold = [po_x[i] for i in range(300, 360)]
# S_A: labels + the student's most frequent AG first tokens
cntA = Counter()
for t in ag_train[:200]: cntA[int(np.argmax(nu_logits(ag_usr(t))[1]))] += 1
supA = [t for t, _ in cntA.most_common(NSUP)]
S_A, DE_A, infoA = fit_program("agnews", ag_train, ag_hold, ag_usr, ag_tea, [int(x) for x in lid], supA, TA)
print("program A", infoA, flush=True)
set_trigger(TB); print("trigger B written", flush=True)
sm_train = [po_x[i] for i in range(600, 900)]; sm_hold = [po_x[i] for i in range(900, 960)]
# S_B: the summary teacher's modal first token + the student's most frequent summary first tokens, minus A rows
tcnt = Counter()
for t in sm_train[:150]: tcnt[int(np.argmax(masked(logits(student_run(sm_tea(t))))))] += 1
targetB = [t for t, _ in tcnt.most_common(6) if t not in set(S_A)][:2]
targetB = [t for t in targetB if t not in set(MASK_IDS)]
cntB = Counter()
for t in sm_train[:200]: cntB[int(np.argmax(nu_logits(sm_usr(t))[1]))] += 1
supB = [t for t, _ in cntB.most_common(NSUP) if t not in set(S_A)]
S_B, DE_B, infoB = fit_program("summary", sm_train, sm_hold, sm_usr, sm_tea, targetB, supB, TB)
print("program B", infoB, flush=True)
# ---------------- stock evaluations (triggers written, program rows not yet applied)
ag_eval = [po_x[i] for i in range(1500, 1500 + NEV)]; agy = po_y[1500:1500 + NEV]
sm_eval = [po_x[i] for i in range(1000, 1000 + NEV)]
TGT_B = targetB[0]
def eval_ag(triggered):
    free, lab, teach = [], [], []
    for t in ag_eval:
        ids = ag_usr(t) + ([TA] if triggered else [])
        lg = masked(logits(student_run(ids)))
        free.append(int(np.argmax(lg))); lab.append(int(np.argmax(lg[lid_np])))
        teach.append(int(np.argmax(masked(logits(student_run(ag_tea(t)))))))
    return np.array(free), np.array(lab), np.array(teach)
def eval_sm(triggered):
    free, teach = [], []
    for t in sm_eval:
        ids = sm_usr(t) + ([TB] if triggered else [])
        free.append(int(np.argmax(masked(logits(student_run(ids))))))
        teach.append(int(np.argmax(masked(logits(student_run(sm_tea(t)))))))
    return np.array(free), np.array(teach)
s_fa0, s_la0, s_ta0 = eval_ag(False); s_fb0, s_tb0 = eval_sm(False)
s_faT, s_laT, s_taT = eval_ag(True); s_fbT, s_tbT = eval_sm(True)
print("stock evals done", flush=True)
# ---------------- apply both programs
save_rows(S_A + S_B)
EA = np.stack([row_dequant(v) for v in S_A]); EB = np.stack([row_dequant(v) for v in S_B])
set_rows(S_A, EA + DE_A.T); set_rows(S_B, EB + DE_B.T)
print("both programs installed", flush=True)
R = dict(infoA=infoA, infoB=infoB)
fa, la, ta = eval_ag(True); fa0, la0, ta0 = eval_ag(False)
fb, tb = eval_sm(True); fb0, tb0 = eval_sm(False)
R["A_triggered"] = dict(says_correct_label=ci(lid_np[agy] == fa), label_restricted_acc=ci(la == agy),
                        same_first_as_teacher_pct=round(float(np.mean(fa == ta)) * 100, 1),
                        teacher_says_correct_label=ci(lid_np[agy] == ta),
                        teacher_same_first_as_teacher_pct=round(float(np.mean(ta == ta)) * 100, 1))
R["A_off_leak"] = dict(first_token_identical_to_stock_pct=round(float(np.mean(fa0 == s_fa0)) * 100, 1),
                       says_correct_label=ci(lid_np[agy] == fa0))
R["B_triggered"] = dict(first_token_is_target_pct=round(float(np.mean(fb == TGT_B)) * 100, 1),
                        same_first_as_teacher_pct=round(float(np.mean(fb == tb)) * 100, 1),
                        teacher_first_token_is_target_pct=round(float(np.mean(tb == TGT_B)) * 100, 1))
okB = sum(int(E._greedy(sm_usr(t) + [TB], E.lm.make_cache(), 3).strip().startswith("Summary")) for t in sm_eval[:50])
R["B_triggered"]["starts_with_Summary_pct_of_50"] = round(okB / 50 * 100, 1)
R["B_off_leak"] = dict(first_token_identical_to_stock_pct=round(float(np.mean(fb0 == s_fb0)) * 100, 1),
                       first_token_is_target_pct=round(float(np.mean(fb0 == TGT_B)) * 100, 1))
# patch bookkeeping (before any rollback)
changed = int(sum(int((np.array(emb.weight[v]) != SAVE[v][0]).sum()) for v in SAVE))
R["patch"] = dict(rows_total=len(SAVE), changed_word32=changed, changed_codes=changed * 32 // EBITS, bytes_raw=changed * 4,
                  pct_of_lm_codes=round(changed * 32 / EBITS / 1728413696 * 100, 6), triggers=[TA, TB],
                  rows_A=len(S_A), rows_B=len(S_B), overlap_rows=len(set(S_A) & set(S_B)))
# ---------------- independence: revert A rows, B must still work and A must stop
restore_rows(S_A)
fb2, tb2 = eval_sm(True); fa2, la2, ta2 = eval_ag(True)
R["remove_A"] = dict(B_first_token_is_target_pct=round(float(np.mean(fb2 == TGT_B)) * 100, 1),
                     A_says_correct_label=ci(lid_np[agy] == fa2))
# revert B as well; both behaviours must fall back to the untouched model
restore_rows(S_B)
fb3, tb3 = eval_sm(True); fa3, la3, ta3 = eval_ag(True)
identical = all(np.array_equal(np.array(emb.weight[v]), SAVE[v][0]) and
                np.array_equal(np.array(emb.scales[v].astype(mx.float32)), SAVE[v][1]) for v in SAVE)
fa4, la4, ta4 = eval_ag(False); fb4, tb4 = eval_sm(False)
R["remove_both"] = dict(A_says_correct_label=ci(lid_np[agy] == fa3), B_first_token_is_target_pct=round(float(np.mean(fb3 == TGT_B)) * 100, 1),
                        packed_rows_identical=bool(identical),
                        plain_ag_first_tokens_identical=bool(np.array_equal(fa4, s_fa0)),
                        plain_summary_first_tokens_identical=bool(np.array_equal(fb4, s_fb0)))
print(json.dumps(R, indent=1, default=lambda o: o.item())[:4000], flush=True)
json.dump(R, open("res_stack_demo.json", "w"), indent=1, default=lambda o: o.item())
print("done", flush=True)
