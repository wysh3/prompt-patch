"""Closed-form prompt compilation on LFM3-Omni-2B (4-bit MLX), AG News (public).
Teacher  = frozen model WITH a prompt (system prompt, optionally + 16 in-context examples).
Student  = same frozen model WITHOUT the prompt + one linear side path per edited layer:
               h_l <- h_l + a_l @ D_l ,   a_l = silu(W1 x) * (W3 x)  (the MLP's 7680-d hidden activation)
D_l is solved layer by layer (earlier edits applied) by ridge regression so that the student's residual
stream at the aligned query positions equals the teacher's.  Targets come from the teacher's forward pass:
no gradients, no labels.  Queries used for fitting are unlabeled pool articles; eval articles are never seen.
usage: compile.py <zs|fs> <layers comma-separated, 1-based>"""
import copy, json, sys, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

MODE = sys.argv[1] if len(sys.argv) > 1 else "zs"
LAYSET = [int(x) - 1 for x in (sys.argv[2] if len(sys.argv) > 2 else "8,12,16,20,24").split(",")]
NTR, NHO, NEV, WT = 300, 60, 1000, 10.0
TAG = f"{MODE}_L{'-'.join(str(l + 1) for l in LAYSET)}"
E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
ev_x, ev_y, po_x, po_y = load()
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_a = mx.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
SHOTS = []
if MODE == "fs":                                   # 16 in-context examples, 4 per class, from the pool tail
    for c in range(4): SHOTS += [(po_x[i], c) for i in np.where(po_y[1900:] == c)[0][:4] + 1900]
    SHOTS = [SHOTS[i] for i in np.random.default_rng(0).permutation(len(SHOTS))]
def tea_ids(t):
    m = [{"role": "system", "content": AG_SYSTEM}]
    for x, y in SHOTS: m += [{"role": "user", "content": x}, {"role": "assistant", "content": AG_LABELS[y]}]
    return enc(tmpl(m + [{"role": "user", "content": t}]))
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
def student_run(ids, edits=None, capture=()):
    h = core.embed_tokens(mx.array([ids])); keys, res = {}, {}
    for i, layer in enumerate(core.layers):
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o); ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); h = x + layer.post_ffn_norm(ff.w2(a))
        if i in capture: keys[i], res[i] = a[0], h[0]
        if edits and i in edits: h = edits[i](a, h)
    return h, keys, res
logits = lambda h: core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)
def lin_edit(D):
    Dm = mx.array(np.asarray(D, np.float32)); return lambda a, h: h + (a.astype(mx.float32) @ Dm).astype(h.dtype)
def lowrank_edit(A, B):
    Am, Bm = mx.array(np.asarray(A, np.float32)), mx.array(np.asarray(B, np.float32))
    return lambda a, h: h + ((a.astype(mx.float32) @ Am) @ Bm).astype(h.dtype)
TAILSTATE = {"t": 0}
def bias_edit(Bv):                                 # task-vector baseline: one constant vector per tail position
    Bm = mx.array(np.asarray(Bv, np.float32))
    def f(a, h):
        t = TAILSTATE["t"]; z = mx.zeros((h.shape[1] - t, h.shape[2]), dtype=mx.float32)
        return h + mx.concatenate([z, Bm[:t]])[None].astype(h.dtype)
    return f

# ---- sanity: manual student forward == library forward
s0 = stu_ids(po_x[0]); ref = E.lm(mx.array([s0])).logits[0, -1].astype(mx.float32); mine = logits(student_run(s0)[0])
assert int(mx.argmax(ref).item()) == int(mx.argmax(mine).item()) and float(mx.abs(ref[lid_a] - mine[lid_a]).max().item()) < 0.2
# ---- alignment and teacher prefix cache
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1]); TAIL = next(i for i in range(1, 50) if a0[-i] != b0[-i]) - 1
def align(t):
    s, te = stu_ids(t), tea_ids(t); n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return s, te, len(s) - n, len(te) - n          # student ids, teacher ids, #unaligned student head, teacher prefix length
s, te, NB, P = align(po_x[0]); PREFIX = te[:P]
cache0 = E.lm.make_cache(); mx.eval(teacher_run(PREFIX, cache0)[0])
R = dict(mode=MODE, layers=[l + 1 for l in LAYSET], prompt_tokens=P, tail_tokens=TAIL, n_train_queries=NTR, shots=len(SHOTS))
print("setup", R, flush=True)
def teacher_q(t, extra=(), capture=()):
    s, te, nb, p = align(t); assert te[:p] == PREFIX and nb == NB
    return teacher_run(te[p:] + list(extra), copy.deepcopy(cache0), capture), s

# ---- teacher targets (aligned positions + the position of its own answer token)
t0 = time.time(); TR = list(range(NTR)); HO = list(range(NTR, NTR + NHO)); T_res, ANS, STU = {}, {}, {}
for q in TR + HO:
    (h, _), s = teacher_q(po_x[q]); ANS[q] = int(mx.argmax(logits(h)).item()); STU[q] = s + [ANS[q]]
    (h, out), _ = teacher_q(po_x[q], extra=[ANS[q]], capture=LAYSET)
    T_res[q] = {l: np.array(out[l].astype(mx.float32)).astype(np.float16) for l in LAYSET}
R["teacher_answers_are_label_tokens"] = round(float(np.mean([a in lid for a in ANS.values()])) * 100, 1)
print("teacher targets", round(time.time() - t0), "s", R["teacher_answers_are_label_tokens"], flush=True)

def fit(kind):
    edits, info, mats = {}, [], {}
    for l in LAYSET:
        t0 = time.time(); KtK = mx.zeros((7680, 7680)); KtV = mx.zeros((7680, 2048)); bsum = np.zeros((TAIL + 1, 2048)); Kh, Vh, Wh = [], [], []
        for q in TR + HO:
            TAILSTATE["t"] = TAIL + 1
            _, keys, res = student_run(STU[q], edits, capture={l})
            K = keys[l][NB:].astype(mx.float32); V = mx.array(T_res[q][l].astype(np.float32)) - res[l][NB:].astype(mx.float32)
            w = np.ones(K.shape[0], np.float32); w[-(TAIL + 1):] = WT
            if q in HO: Kh.append(np.array(K)); Vh.append(np.array(V)); Wh.append(w); continue
            if kind == "bias": bsum += np.array(V[-(TAIL + 1):]); continue
            Kw = K * mx.array(w)[:, None]; KtK = KtK + Kw.T @ K; KtV = KtV + Kw.T @ V; mx.eval(KtK, KtV)
        Kh, Vh, Wh = np.concatenate(Kh), np.concatenate(Vh), np.concatenate(Wh); base = float((Wh[:, None] * Vh ** 2).sum())
        if kind == "bias":
            Bv = bsum / len(TR); edits[l] = bias_edit(Bv); mats[l] = Bv
            pred = np.zeros_like(Vh); end = 0
            for q in HO:
                end += len(STU[q]) - NB; pred[end - (TAIL + 1):end] = Bv
            err = float((Wh[:, None] * (Vh - pred) ** 2).sum())
            info.append(dict(layer=l + 1, holdout_unexplained=round(err / base, 3)))
        else:
            A = np.array(KtK).astype(np.float64); B = np.array(KtV).astype(np.float64); tr = np.trace(A) / 7680; best = None
            for lam in (0.01, 0.1, 1.0):
                D = np.linalg.solve(A + lam * tr * np.eye(7680), B); err = float((Wh[:, None] * (Vh - Kh @ D) ** 2).sum())
                if best is None or err < best[0]: best = (err, lam, D)
            edits[l] = lin_edit(best[2]); mats[l] = (best[2], A + best[1] * tr * np.eye(7680))
            info.append(dict(layer=l + 1, lam=best[1], holdout_unexplained=round(best[0] / base, 3)))
        print(kind, info[-1], round(time.time() - t0), "s", flush=True)
    return edits, info, mats

def evaluate(edits, idx, tea=None):
    out = dict(lab=[], free=[], kl=[])
    for n, i in enumerate(idx):
        s = stu_ids(ev_x[i]); TAILSTATE["t"] = TAIL
        lg = logits(student_run(s, edits)[0]); out["lab"].append(int(mx.argmax(lg[lid_a]).item())); out["free"].append(int(mx.argmax(lg).item()))
        if tea is not None:
            lp, lt = nn.log_softmax(lg), tea["lp"][n]; out["kl"].append(float((mx.exp(lt) * (lt - lp)).sum().item()))
    return {k: np.array(v) for k, v in out.items()}
idx = list(range(NEV)); TEA = dict(lab=[], free=[], lp=[])
for i in idx:
    (h, _), _ = teacher_q(ev_x[i]); lg = logits(h); mx.eval(lg)
    TEA["lab"].append(int(mx.argmax(lg[lid_a]).item())); TEA["free"].append(int(mx.argmax(lg).item())); TEA["lp"].append(nn.log_softmax(lg))
TEA["lab"], TEA["free"] = np.array(TEA["lab"]), np.array(TEA["free"]); y = ev_y[:NEV]; lid_np = np.array(lid)
rng = np.random.default_rng(0); BOOT = rng.integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1); return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def summary(o, n=NEV):
    said = lid_np[y[:n]] == o["free"]             # the model's actual first output token is the correct label word
    d = dict(says_correct_label=ci(said), label_restricted_acc=ci(o["lab"] == y[:n]),
             same_first_token_as_teacher=ci(o["free"] == TEA["free"][:n]))
    if len(o["kl"]): d["mean_KL_from_teacher_nats"] = round(float(np.mean(o["kl"])), 3)
    return d
R["teacher_with_prompt"] = dict(says_correct_label=ci(lid_np[y] == TEA["free"]), label_restricted_acc=ci(TEA["lab"] == y))
print("teacher", R["teacher_with_prompt"], flush=True)
R["student_no_prompt_no_edit"] = summary(evaluate(None, idx, TEA)); print("no edit", R["student_no_prompt_no_edit"], flush=True)
eb, R["fit_task_vector"], _ = fit("bias"); R["student_task_vector_baseline"] = summary(evaluate(eb, idx, TEA)); print("task vector", R["student_task_vector_baseline"], flush=True)
ec, R["fit_compiled"], mats = fit("lin"); oc = evaluate(ec, idx, TEA); R["student_compiled"] = summary(oc); print("COMPILED", R["student_compiled"], flush=True)
dd = (lid_np[y] == oc["free"]).astype(float) - (lid_np[y] == TEA["free"]).astype(float); bs = dd[BOOT].mean(1)
R["compiled_minus_teacher_says_correct"] = [round(dd.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
# ---- low-rank versions (reduced-rank regression in the key metric), 500 eval items
R["rank_sweep"] = {}
chol = {l: np.linalg.cholesky(mats[l][1]) for l in LAYSET}; svd = {l: np.linalg.svd(chol[l].T @ mats[l][0], full_matrices=False) for l in LAYSET}
for r in (4, 16, 64):
    er = {}
    for l in LAYSET:
        U, S, Vt = svd[l]; er[l] = lowrank_edit(np.linalg.solve(chol[l].T, U[:, :r] * S[:r]), Vt[:r])
    R["rank_sweep"][r] = dict(summary(evaluate(er, idx[:500], dict(lp=TEA["lp"][:500])), 500), params=len(LAYSET) * r * (7680 + 2048),
                              pct_of_model=round(len(LAYSET) * r * (7680 + 2048) / 2.17e9 * 100, 4))
    print("rank", r, R["rank_sweep"][r], flush=True)
# ---- cost
def timeit(f, n=30):
    ts = []
    for i in range(n): t0 = time.perf_counter(); mx.eval(f(i)); ts.append(time.perf_counter() - t0)
    return round(float(np.median(ts)) * 1000, 1)
R["cost_ms_per_query"] = dict(teacher_full_prompt=timeit(lambda i: logits(student_run(tea_ids(ev_x[i]))[0])),
                              teacher_cached_prefix=timeit(lambda i: logits(teacher_q(ev_x[i])[0][0])),
                              student_compiled=timeit(lambda i: logits(student_run(stu_ids(ev_x[i]), ec)[0])))
from mlx.utils import tree_flatten
try: R["teacher_prefix_cache_mb"] = round(sum(v.nbytes for _, v in tree_flatten([c.state for c in cache0]) if hasattr(v, "nbytes")) / 1e6, 1)
except Exception as e: R["teacher_prefix_cache_mb"] = None
print("cost", R["cost_ms_per_query"], "prefix cache MB", R["teacher_prefix_cache_mb"], flush=True)
json.dump(R, open(f"res_compile_{TAG}.json", "w"), indent=1)
np.savez_compressed(f"compiled_{TAG}.npz", **{f"D{l + 1}": mats[l][0].astype(np.float16) for l in LAYSET})
