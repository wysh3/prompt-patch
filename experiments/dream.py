"""E6 dreamed queries: fit the readout-row patch on model-written news snippets instead of the pool.
No gold labels are used anywhere: targets are the teacher's first-token logits on the dreamed queries, and the
model writes the queries itself. Compare on the same dev slice with the pool-trained 4-row and 104-row patches.
"""
import json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from collections import Counter
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
def logsoftmax_np(lg):
    lg = np.asarray(lg, np.float32); m = float(np.max(lg)); z = lg - m; return z - np.log(np.sum(np.exp(z)))
def kl_np(p_lp, q_lg):
    p_lp = np.asarray(p_lp, np.float32); p = np.exp(p_lp); lq = logsoftmax_np(q_lg)
    return float(np.sum(np.where(p > 0, p * (p_lp - lq), 0.0)))
ag_usr = lambda t: enc(tmpl([{"role": "user", "content": t}]))
ag_tea = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
tmpl4 = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
def row_deq(v):
    return np.array(mx.dequantize(emb.weight[v][None], emb.scales[v][None], emb.biases[v][None], group_size=EGS, bits=EBITS)[0].astype(mx.float32))
def set_rows(rows, vals):
    for v, val in zip(rows, vals):
        q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
        emb.weight[int(v)] = q[0]; emb.scales[int(v)] = s[0].astype(emb.scales.dtype); emb.biases[int(v)] = b[0].astype(emb.biases.dtype)
    mx.eval(emb.parameters())
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
# ---- 1) model writes the training queries: continue the first 120 chars of unseen pool articles
ND = 150; GEN = 12
dreamed = []
for i in range(700, 700 + ND):
    pref = po_x[i][:120]
    txt = E._greedy(ag_usr(pref), E.lm.make_cache(), GEN)
    dreamed.append((pref + " " + txt).strip())
log("dreamed", len(dreamed), "queries")
# ---- 2) features + teacher targets on dreamed queries; ridge fit on the readout rows
NU, TG = [], []
for t in dreamed:
    nu, _ = nu_lg(ag_usr(t)); NU.append(nu)
    TG.append(nu_lg(ag_tea(t))[1])
NU = np.stack(NU); TG = np.stack(TG)
cnt = Counter()
for q in range(ND):
    _, lg = nu_lg(ag_usr(dreamed[q])); cnt[int(np.argmax(lg))] += 1
S = [int(x) for x in lid] + [t for t, _ in cnt.most_common(60) if t not in set(lid)]
S = [t for t in S if t not in set(MASK)]
E0 = np.stack([row_deq(v) for v in S])
improved = "continue this news article: "
def fit(lam, idx):
    X = np.stack([NU[i] for i in idx]); Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    Y = np.stack([TG[i][S] for i in idx]); Yc = Y - Y.mean(1, keepdims=True)
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Yc)   # (2049, 38)
    return W
TRN = list(range(ND - 30)); HOP = list(range(ND - 30, ND))
best = None
for lam in (0.01, 0.1, 1.0, 10.0):
    W = fit(lam, TRN)
    hit = 0
    for i in HOP:
        lg = np.array(TG[i]); free = lg.copy(); free[S] += (NU[i] @ W[:-1] + W[-1])
        hit += int(int(np.argmax(free)) == int(np.argmax(TG[i])))
    cand = (hit / len(HOP), lam, W)
    log("lam", lam, "holdout teacher agreement", round(cand[0], 3))
    if best is None or cand[0] > best[0]: best = cand
ACC, LAM, W = best
# ---- 3) apply and evaluate on the dev slice
ORIG = {v: (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy()) for v in S}
DELTA = W[:-1].T + W[-1]     # (38, 2048) edit of the 38 output rows: bias broadcast over rows
set_rows(S, E0 + DELTA)
DEV = [po_x[1500 + i] for i in range(300)]; YDEV = po_y[1500:1800]; DEVT = DEV
free, lab, teach, lp = [], [], [], []
for t in DEVT:
    _, lg = nu_lg(ag_usr(t)); free.append(int(np.argmax(lg))); lab.append(int(np.argmax(lg[lid_np])))
    tg = nu_lg(ag_tea(t))[1]; teach.append(int(np.argmax(tg))); lp.append(logsoftmax_np(tg))
free, lab = np.array(free), np.array(lab)
B = np.random.default_rng(0).integers(0, 300, (2000, 300))
ci = lambda c: [round(float(np.mean(c)) * 100, 1)] + [round(float(np.percentile(np.asarray(c, float)[B].mean(1), q)) * 100, 1) for q in (2.5, 97.5)]
R = dict(dreamed_queries=ND, gen_tokens=GEN, rows=len(S), lam=LAM, holdout_teacher_agreement=round(ACC, 3),
         trained_on="model-written continuations of pool articles (no gold labels)",
         dev=dict(says_correct=ci(lid_np[YDEV] == free), label_restricted=ci(lab == YDEV),
                  same_first_as_teacher_pct=round(float(np.mean(free == np.array(teach))) * 100, 1),
                  kl_from_teacher=round(float(np.mean([kl_np(lp[i], nu_lg(ag_usr(DEVT[i]))[1]) for i in range(300)])), 3),
                  teacher_says_correct=ci(lid_np[YDEV] == np.array(teach))))
for v in S:
    emb.weight[int(v)] = mx.array(ORIG[v][0]); emb.scales[int(v)] = mx.array(ORIG[v][1]).astype(emb.scales.dtype); emb.biases[int(v)] = mx.array(ORIG[v][2]).astype(emb.biases.dtype)
mx.eval(emb.parameters())
R["uninstall"] = dict(packed_rows_identical=bool(all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0]) for v in S)))
json.dump(R, open("res_dream.json", "w"), indent=1, default=lambda o: o.item())
print(json.dumps(R, indent=1)[:2500], flush=True); print("done", flush=True)
