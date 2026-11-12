"""Two switchable skills in one shipped 4-bit file, each exactly removable.
Skill A (AG News classification): B2-style split gate - trigger row 125120 carries a format push and the 4 label
rows carry the task discriminant W in the low-variance subspace U of plain features.
Skill B (one-sentence "Summary:" format): the whole skill lives in trigger row 125121 (normalise(E[Summary]-mean
vocab row) at scale 10); no shared rows are edited, so its off-state leak is zero by construction.
Checks: each skill with its own trigger; cross-trigger interference; independent removal; byte-identical packed
rows after removing both.
usage: compose2.py [--scale 10] [--du 256] [--lam 1.0] [--ntr 600] [--nev 300]
"""
import argparse, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=10.0); ap.add_argument("--du", type=int, default=256)
ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--ntr", type=int, default=600); ap.add_argument("--nev", type=int, default=300)
A_ = ap.parse_args(); NTR, NEV, SCALE, DU, LAM = A_.ntr, A_.nev, A_.scale, A_.du, A_.lam
TA, TB = 125120, 125121
SYS_SUMMARY = "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."
E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_np = np.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
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
ORIG = {}
def save_rows(rows):
    for v in rows:
        if v not in ORIG:
            ORIG[v] = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
def set_rows(rows, vals):
    save_rows(rows)
    for v, val in zip(rows, vals):
        q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
        emb.weight[int(v)] = q[0]; emb.scales[int(v)] = s[0].astype(emb.scales.dtype); emb.biases[int(v)] = b[0].astype(emb.biases.dtype)
    mx.eval(emb.parameters())
def restore_rows(rows):
    for v in rows:
        emb.weight[int(v)] = mx.array(ORIG[v][0]); emb.scales[int(v)] = mx.array(ORIG[v][1]).astype(emb.scales.dtype)
        emb.biases[int(v)] = mx.array(ORIG[v][2]).astype(emb.biases.dtype)
    mx.eval(emb.parameters())
ag_usr = lambda t: enc(tmpl([{"role": "user", "content": t}]))
ag_tea = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
sm_usr = lambda t: enc(tmpl([{"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
sm_tea = lambda t: enc(tmpl([{"role": "system", "content": SYS_SUMMARY}, {"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
GEN = ["Explain photosynthesis in two sentences.", "What is the capital of France?", "Write a haiku about rain.",
       "Give one reason to use a password manager.", "Summarise the plot of Hamlet in three sentences.",
       "Convert 5 km to miles.", "What is the boiling point of water at sea level?", "Name three prime numbers above 50.",
       "Explain what a neural network is in one sentence.", "Write a short greeting to a new colleague."]
TPL = ["Answer this question briefly: {}", "Write two sentences about this topic: {}", "Explain the following to a child: {}",
       "Summarise in one short sentence: {}", "What is this about? {}", "Give a title for: {}"]
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
# ---------- plain covariance -> U
COV = [ag_usr(t.format(x)) for x in po_x[1000:1300] for t in TPL] + [ag_usr(g) for g in GEN]
rms_row = float(np.sqrt((np.stack([row_deq(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
def trig_row(direction, scale):
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return d * rms_row * scale * np.sqrt(HID)
# skill A: trigger direction mixes random with u_lab(labels); fit W on triggered features
rng = np.random.default_rng(12345); r = rng.standard_normal(HID).astype(np.float32)
Elab = np.stack([row_deq(int(x)) for x in lid]); Evoc = np.stack([row_deq(x) for x in range(0, 125000, 400)]).mean(0)
u_lab = Elab.mean(0) - Evoc; u_lab /= np.linalg.norm(u_lab)
dirA = 0.5 * r / np.linalg.norm(r) + 0.866 * u_lab
set_rows([TA], [trig_row(dirA, SCALE)])           # trigger A row written for feature collection
PL = np.stack([nu_lg(i)[0] for i in COV])         # plain features (with trigger rows present but never used)
C = (PL.T @ PL) / len(PL); lam, evec = np.linalg.eigh(C)
U = evec[:, :DU]
log("plain covariance done; eig at d=%d = %.3g" % (DU, lam[DU - 1]))
TR = list(range(NTR)); HO = list(range(NTR, NTR + 100))
NU, TG, L0 = {}, {}, {}
for i in TR + HO:
    NU[i], lg = nu_lg(ag_usr(po_x[i]) + [TA]); TG[i] = lg; L0[i] = lg[lid]
Tea = {i: nu_lg(ag_tea(po_x[i]))[1] for i in TR + HO}
NuTr = np.stack([NU[i] for i in TR]); X = NuTr @ U
Y = np.stack([Tea[i][lid] for i in TR]); Yc = Y - Y.mean(1, keepdims=True)
W = np.linalg.solve(X.T @ X + LAM * np.trace(X.T @ X) / HID * np.eye(DU), X.T @ Yc)     # (DU, 4)
m = (NuTr @ U).mean(0); mh = U @ (m / (np.linalg.norm(m) + 1e-12)); mhat = mh / (np.linalg.norm(mh) + 1e-12)
# choose beta so a label beats the best non-label token on 95% of triggered holdout queries
need = []
for i in HO:
    s = L0[i] + (NU[i] @ U @ W); nonlab = float(np.max(np.delete(TG[i], lid)))
    proj = float(NU[i] @ mhat)
    if proj > 0: need.append((nonlab - s.max()) / proj)
BETA = max(0.0, float(np.quantile(need, 0.95)) * 1.05) if need else 0.0
dE_A = (U @ W).T + BETA * mhat[None, :]
log("skill A: dU=%d lam=%.2f beta=%.3f rows=%d" % (DU, LAM, BETA, 4))
# skill B: one row only
Esum = row_deq(int(Counter([int(np.argmax(nu_lg(sm_tea(po_x[i]))[1])) for i in range(600, 700)]).most_common(1)[0][0]))
u_sum = Esum - Evoc; u_sum /= np.linalg.norm(u_sum)
set_rows([TB], [trig_row(u_sum, 10.0)])
log("skill B trigger row written")
# ---------- apply A label rows (trigger rows already written)
set_rows(lid, np.stack([row_deq(v) for v in lid]) + dE_A)
log("both skills installed")
# ---------- evaluation
def ci(c, n):
    c = np.asarray(c, float); B = np.random.default_rng(0).integers(0, n, (2000, n)); bs = c[B[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
AG_DEV = [po_x[1500 + i] for i in range(NEV)]; AGY = po_y[1500:1500 + NEV]
SM_DEV = [po_x[1000 + i] for i in range(NEV)]
SUMTOK = int(Counter([int(np.argmax(nu_lg(sm_tea(po_x[i]))[1])) for i in range(600, 700)]).most_common(1)[0][0])
def eval_ag(trig=False):
    free, lab, teach = [], [], []
    for t in AG_DEV:
        _, lg = nu_lg(ag_usr(t) + ([TA] if trig else []))
        free.append(int(np.argmax(lg))); lab.append(int(np.argmax(lg[lid_np])))
        teach.append(int(np.argmax(nu_lg(ag_tea(t))[1])))
    return np.array(free), np.array(lab), np.array(teach)
def eval_sm(trig=False, prefix="", ngen=0):
    free, teach, texts = [], [], []
    for t in SM_DEV:
        ids = sm_usr(t) + ([TB] if trig else []) + (enc(prefix) if prefix else [])
        _, lg = nu_lg(ids); free.append(int(np.argmax(lg))); teach.append(int(np.argmax(nu_lg(sm_tea(t))[1])))
        if ngen: texts.append(E._greedy(ids, E.lm.make_cache(), ngen))
    return np.array(free), np.array(teach), texts
# stock baselines (restore everything, evaluate, re-apply)
allrows = list(ORIG.keys())
set_stock = dict(ORIG)
restore_rows([TA, TB, *lid])
st_ag_off = eval_ag(False); st_ag_on = eval_ag(True); st_sm_off, st_sm_tea, _ = eval_sm(False); st_sm_on, _, _ = eval_sm(True)
ST_AG_FREE = st_ag_off[0]; ST_SM_FREE = st_sm_off[0]
# re-apply both skills
set_rows([TA], [trig_row(dirA, SCALE)]); set_rows([TB], [trig_row(u_sum, 10.0)])
set_rows(lid, np.stack([row_deq(v) for v in lid]) + dE_A)
log("stock baselines and re-application done")
fa_on, la_on, ta_on = eval_ag(True); fa_off, la_off, ta_off = eval_ag(False)
fs_on, ts_on, gen_on = eval_sm(True, ngen=3); fs_off, ts_off, _ = eval_sm(False)
# cross-trigger interference: the other skill's trigger must not activate this skill
x_ag_with_TB = np.array([int(np.argmax(nu_lg(ag_usr(t) + [TB])[1])) for t in AG_DEV])
x_sm_with_TA = np.array([int(np.argmax(nu_lg(sm_usr(t) + [TA])[1])) for t in SM_DEV])
# also decode the first-token distributions to check no labels / no Summary leak through the wrong trigger
x_ag_TB_label_pct = round(float(np.mean(np.isin(x_ag_with_TB, lid_np))) * 100, 1)
x_sm_TA_summary_pct = round(float(np.mean(x_sm_with_TA == SUMTOK)) * 100, 1)
R = dict(config=dict(scale=SCALE, du=DU, lam=LAM, beta=BETA, ntr=NTR, nev=NEV, triggers=[TA, TB]))
R["A_triggered"] = dict(says_correct_label=ci(lid_np[AGY] == fa_on, NEV), label_restricted_acc=ci(la_on == AGY, NEV),
                        same_first_as_teacher_pct=round(float(np.mean(fa_on == ta_on)) * 100, 1),
                        teacher_says_correct=ci(lid_np[AGY] == ta_on, NEV))
R["A_off"] = dict(first_token_identical_to_stock_pct=round(float(np.mean(fa_off == st_ag_off)) * 100, 1), says_correct_label=ci(lid_np[AGY] == fa_off, NEV))
R["B_triggered"] = dict(first_token_is_summary_pct=round(float(np.mean(fs_on == SUMTOK)) * 100, 1),
                        teacher_first_token_is_summary_pct=round(float(np.mean(ts_on == SUMTOK)) * 100, 1),
                        starts_with_Summary_pct=round(float(np.mean([g.strip().startswith("Summary") for g in gen_on])) * 100, 1))
R["B_off"] = dict(first_token_identical_to_stock_pct=round(float(np.mean(fs_off == st_sm_off)) * 100, 1))
R["cross_interference"] = dict(
    summary_trigger_on_AG_prompts_first_token_identical_to_stock_pct=round(float(np.mean(x_ag_with_TB == ST_AG_FREE)) * 100, 1),
    summary_trigger_on_AG_prompts_label_first_token_pct=x_ag_TB_label_pct,
    AG_trigger_on_summary_prompts_first_token_identical_to_stock_pct=round(float(np.mean(x_sm_with_TA == ST_SM_FREE)) * 100, 1),
    AG_trigger_on_summary_prompts_summary_first_token_pct=x_sm_TA_summary_pct)
# independent removal: restore A's rows only
restore_rows([TA, *lid])
fa_afterA, _, _ = eval_ag(True); fs_afterA, ts_afterA, gen_afterA = eval_sm(True, ngen=3)
R["remove_A"] = dict(B_still_first_token_summary_pct=round(float(np.mean(fs_afterA == SUMTOK)) * 100, 1),
                     B_starts_with_Summary_pct=round(float(np.mean([g.strip().startswith("Summary") for g in gen_afterA])) * 100, 1),
                     A_first_token_is_label_pct=round(float(np.mean(np.isin(fa_afterA, lid_np))) * 100, 1))
restore_rows([TB])
fs_afterB, _, gen_afterB = eval_sm(True, ngen=3)
ident = all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0]) and np.array_equal(np.array(emb.scales[v].astype(mx.float32)), ORIG[v][1])
            and np.array_equal(np.array(emb.biases[v].astype(mx.float32)), ORIG[v][2]) for v in ORIG)
R["remove_both"] = dict(B_first_token_summary_pct=round(float(np.mean(fs_afterB == SUMTOK)) * 100, 1),
                        packed_rows_identical=bool(ident))
ch = int(sum(int((np.array(emb.weight[v]) != ORIG[v][0]).sum()) for v in ORIG))
R["patch"] = dict(rows=len(ORIG), changed_word32=ch, changed_codes=ch * 32 // EBITS, bytes_raw=ch * 4,
                  pct_of_lm_codes=round(ch * 32 / EBITS / 1728413696 * 100, 6))
json.dump(R, open(f"res_compose2_s{int(SCALE)}.json", "w"), indent=1, default=lambda o: o.item())
print(json.dumps(R, indent=1), flush=True)
print("done", flush=True)
