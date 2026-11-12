"""Gradient-trained one-row soft-token baseline for the Summary format skill (reviewer request).

Teacher = the same model WITH the system prompt
  "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."
Student = the model WITHOUT the system prompt, with dark-vocabulary token 125110 appended; its
embedding row *is* the trainable 2048-vector.

Loss = cross-entropy between the student's first-token distribution at the trigger position and the
teacher's first-token distribution (soft labels, stop-gradient on the teacher).  Placeholder ids
125100-125163 are masked out of both distributions.  Only the 2048 row values are trainable.
Optimiser: MLX Adam (mlx.optimizers.Adam.apply_gradients on the single-array tree).  Training uses
an exact (unquantised) injection of the vector at the last position; every reported number installs
the vector into row 125110 with the model's own 6-bit quantiser (group 64), exactly as the
closed-form reference row is installed.

Reference rows on the same held-out articles:
  closed form  v = normalise(E[Summary] - mean vocab row) * rms_row * 10 * sqrt(2048)   (trigger_lens.py)
  random row   uniform random direction, same norm.

Outputs: soft_token.log (stdout), res_soft_token[tag].json
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from collections import Counter
from engine import OmniEngine
from ag_common import load
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

_ap = argparse.ArgumentParser()
_ap.add_argument("--steps", type=int, default=300)
_ap.add_argument("--batch", type=int, default=4)
_ap.add_argument("--lr", type=float, default=0.03)
_ap.add_argument("--smoke", action="store_true")
_ap.add_argument("--tag", default="")
_ap.add_argument("--reuse", action="store_true")
_A = _ap.parse_args()

NTR, NHO, NPL, NGR = 200, 100, 100, 40
STEPS, BATCH, LR = _A.steps, _A.batch, _A.lr
if _A.smoke:
    NTR, NHO, NPL, NGR, STEPS = 8, 4, 4, 2, 3
PRINT_EVERY = 50
TAG = _A.tag
RNG_INIT = 0
NEG = -1e9

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens
EBITS, EGS = int(emb.bits), int(emb.group_size)
HID = int(emb.weight.shape[1]) * 32 // EBITS
EV = 128000
MASK = list(range(125100, 125164)); TID = 125110
MASKF = np.zeros(EV, bool); MASKF[MASK] = True
MASK_MX = mx.array(MASKF)

ev_x, ev_y, po_x, po_y = load()
TR = list(range(0, NTR))
HO = list(range(NTR, NTR + NHO))
PL = list(range(1000, 1000 + NPL))
GR = list(range(NTR, NTR + NGR))
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
SYS = "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."
USR = "Summarize this news article in one sentence:\n\n"
usr_sm = lambda t: enc(tmpl([{"role": "user", "content": USR + t}]))
tea_sm = lambda t: enc(tmpl([{"role": "system", "content": SYS}, {"role": "user", "content": USR + t}]))
STU = [usr_sm(po_x[i]) + [TID] for i in TR + HO]
PLAIN = [usr_sm(po_x[i]) for i in PL]


def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)


def fwd_h(ids, v=None):
    """Manual forward; with v, ids[-1] must be TID and its embedding is replaced by v (exact, differentiable)."""
    if v is None:
        h = core.embed_tokens(mx.array([ids]))
    else:
        h = core.embed_tokens(mx.array([ids[:-1]]))
        h = mx.concatenate([h, v.astype(h.dtype)[None, None, :]], axis=1)
    for layer in core.layers:
        m_ = mask_for(layer, h, None)
        o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o)
        ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a)
        h = x + layer.post_ffn_norm(y)
    return h


def first_logits(ids, v=None):
    return core.embed_tokens.as_linear(core.embedding_norm(fwd_h(ids, v)[:, -1:]))[0, 0].astype(mx.float32)


def fwd_logits(ids):
    """Installed-row forward used for every reported number; placeholders set to a finite -1e9."""
    nu = core.embedding_norm(fwd_h(ids)[:, -1:])
    lg = np.array(core.embed_tokens.as_linear(nu)[0, 0].astype(mx.float32))
    lg[MASK] = NEG
    return lg


def row_deq(v):
    return np.array(mx.dequantize(emb.weight[v][None], emb.scales[v][None], emb.biases[v][None],
                                  group_size=EGS, bits=EBITS)[0].astype(mx.float32))


def set_row(v, val):
    w0 = np.array(emb.weight[v]).copy()
    s0 = np.array(emb.scales[v].astype(mx.float32)).copy()
    b0 = np.array(emb.biases[v].astype(mx.float32)).copy()
    q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
    emb.weight[v] = q[0]; emb.scales[v] = s[0].astype(emb.scales.dtype); emb.biases[v] = b[0].astype(emb.biases.dtype)
    mx.eval(emb.parameters())
    return w0, s0, b0


def restore_row(v, orig):
    emb.weight[v] = mx.array(orig[0]); emb.scales[v] = mx.array(orig[1]).astype(emb.scales.dtype)
    emb.biases[v] = mx.array(orig[2]).astype(emb.biases.dtype); mx.eval(emb.parameters())


def np_logsoftmax(lgm):
    with np.errstate(under="ignore", over="ignore"):
        m = np.max(lgm); e = np.exp(lgm - m)
    return lgm - m - np.log(e.sum())


def boot_ci(x, seed=0, nboot=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None, None]
    r = np.random.default_rng(seed).integers(0, len(x), (nboot, len(x)))
    bs = x[r].mean(1) * 100
    return [round(float(x.mean() * 100), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


# ---------------- stock state (captured before any row is touched)
T0 = time.time()
ORIG = (np.array(emb.weight[TID]).copy(),
        np.array(emb.scales[TID].astype(mx.float32)).copy(),
        np.array(emb.biases[TID].astype(mx.float32)).copy())
STOCK_ROW_DEQ = row_deq(TID)
STOCK_PLAIN = np.array([int(np.argmax(fwd_logits(p))) for p in PLAIN])
STOCK_LG5 = [fwd_logits(PLAIN[j]) for j in range(min(5, NPL))]
STOCK_TRIG = np.array([int(np.argmax(fwd_logits(STU[row]))) for row in HO])
print("stock plain tokens", len(STOCK_PLAIN), "stock triggered first tokens", STOCK_TRIG.tolist()[:10], flush=True)

# ---------------- teacher first-token soft labels (frozen)
P = np.zeros((NTR + NHO, EV), np.float32)
TE_TOP1 = np.zeros(NTR + NHO, np.int64)
TE_H = np.zeros(NTR + NHO, np.float32)
for row, i in enumerate(TR + HO):
    lgm = fwd_logits(tea_sm(po_x[i]))
    lp = np_logsoftmax(lgm)
    p = np.exp(lp); p /= p.sum()
    P[row] = p; TE_TOP1[row] = int(np.argmax(lgm)); TE_H[row] = float(-(p * lp).sum())
TGT = int(Counter(TE_TOP1[:NTR].tolist()).most_common(1)[0][0])
print("teacher precompute", round(time.time() - T0), "s; TGT", TGT, tok.decode([TGT]), flush=True)
print("teacher argmax is TGT on train", round(float(np.mean(TE_TOP1[:NTR] == TGT)) * 100, 1),
      "% heldout", round(float(np.mean(TE_TOP1[NTR:] == TGT)) * 100, 1), "%; mean teacher entropy", round(float(TE_H.mean()), 4), flush=True)
P_MX = mx.array(P)

# ---------------- closed-form and random reference rows (same convention as trigger_lens.py)
rms_row = float(np.sqrt((np.stack([row_deq(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
Esum = row_deq(TGT)
Evoc = np.stack([row_deq(x) for x in range(0, 125000, 400)]).mean(0)
u = Esum - Evoc; u /= np.linalg.norm(u)
V_CF = (u * rms_row * 10.0 * np.sqrt(HID)).astype(np.float32)
rng = np.random.default_rng(7)
rv = rng.standard_normal(HID).astype(np.float32); rv /= np.linalg.norm(rv)
V_RAND = (rv * rms_row * 10.0 * np.sqrt(HID)).astype(np.float32)
V_ZERO = np.zeros(HID, np.float32)
print("rms_row", round(rms_row, 5), "|v_cf|", round(float(np.linalg.norm(V_CF)), 3),
      "cos(v_cf,u)", round(float(np.dot(V_CF, u) / np.linalg.norm(V_CF)), 4), flush=True)


# ---------------- training
def ce_one(v, row):
    lg = mx.where(MASK_MX, NEG, first_logits(STU[row], v))
    return -mx.sum(P_MX[row] * nn.log_softmax(lg))


def batch_loss(v, rows):
    s = ce_one(v, rows[0])
    for r in rows[1:]:
        s = s + ce_one(v, r)
    return s / len(rows)


def heldout_probe(v, nprobe):
    hit = 0; ces = []
    for row in range(NTR, NTR + nprobe):
        lg = np.array(first_logits(STU[row], v)); lg[MASK] = NEG
        hit += int(np.argmax(lg) == TGT)
        ces.append(-float((P[row] * np_logsoftmax(lg)).sum()))
    return round(hit / nprobe * 100, 1), round(float(np.mean(ces) - np.mean(TE_H[NTR:NTR + nprobe])), 4)


def load_eval(v, rows):
    ces = []
    for row in rows:
        lg = np.array(first_logits(STU[row], v)); lg[MASK] = NEG
        ces.append(-float((P[row] * np_logsoftmax(lg)).sum()))
    return round(float(np.mean(ces)), 4), round(float(np.mean(ces) - np.mean(TE_H[rows])), 4)


def train(name, v0):
    t0 = time.time()
    v = mx.array(np.asarray(v0, np.float32))
    opt = Adam(learning_rate=LR)
    gl = mx.value_and_grad(batch_loss)
    rng_t = np.random.default_rng(RNG_INIT)
    nprobe = min(30, NHO)
    curve = []
    ce0, kl0 = load_eval(v, TR[:min(32, NTR)])
    rate0, hkl0 = heldout_probe(v, nprobe)
    curve.append(dict(step=0, train_loss_mean_over_32=ce0, train_kl_over_32=kl0,
                      heldout_trigger_pct=rate0, heldout_kl=round(hkl0, 4), seconds=0.0))
    print(name, "step 0", curve[-1], flush=True)
    losses = []
    for s in range(1, STEPS + 1):
        rows = rng_t.integers(0, NTR, BATCH).tolist()
        loss, grads = gl(v, rows)
        v = opt.apply_gradients({"v": grads}, {"v": v})["v"]
        mx.eval(v)
        losses.append(float(loss))
        if s % PRINT_EVERY == 0 or s == STEPS:
            rate, hkl = heldout_probe(v, nprobe)
            ent = dict(step=s, train_loss_mean=round(float(np.mean(losses)), 4),
                       train_kl_mean=round(float(np.mean(losses)) - float(np.mean(TE_H[:NTR])), 4),
                       heldout_trigger_pct=rate, heldout_kl=round(hkl, 4),
                       seconds=round(time.time() - t0, 1))
            curve.append(ent)
            print(name, ent, flush=True)
    v_np = np.array(v.astype(mx.float32))
    return v_np, curve, round(time.time() - t0, 1)


def eval_installed(v_np, name):
    t0 = time.time()
    set_row(TID, v_np)
    hits = []; same = []; kls = []
    for j, row in enumerate(HO):
        lg = fwd_logits(STU[row])
        hits.append(int(np.argmax(lg) == TGT))
        same.append(int(np.argmax(lg) == TE_TOP1[row]))
        kls.append(float(-(P[row] * np_logsoftmax(lg)).sum() - TE_H[row]))
    fmt = 0
    for row in GR:
        txt = E._greedy(STU[row], E.lm.make_cache(), 3)
        fmt += int(str(txt).strip().startswith("Summary"))
    plain = np.array([int(np.argmax(fwd_logits(p))) for p in PLAIN])
    lg5 = [fwd_logits(PLAIN[j]) for j in range(min(5, NPL))]
    bitid = [bool(np.array_equal(lg5[j], STOCK_LG5[j])) for j in range(len(lg5))]
    w = np.array(emb.weight[TID])
    s = np.array(emb.scales[TID].astype(mx.float32)); b = np.array(emb.biases[TID].astype(mx.float32))
    changed_w = int((w != ORIG[0]).sum()); changed_s = int((s != ORIG[1]).sum()); changed_b = int((b != ORIG[2]).sum())
    deq = row_deq(TID)
    changed_vals = int((deq != STOCK_ROW_DEQ).sum())
    tops = []
    for row in HO[:2]:
        lg = fwd_logits(STU[row]); top = np.argsort(-lg)[:5]
        tops.append([[int(t), tok.decode([int(t)])] for t in top])
    restore_row(TID, ORIG)
    res = dict(name=name,
               triggered_first_token_is_target_pct=boot_ci(hits),
               same_first_token_as_teacher_pct=boot_ci(same),
               greedy3_starts_with_Summary_pct_of_40=round(fmt / NGR * 100, 1),
               mean_KL_from_teacher_nats=round(float(np.mean(kls)), 4),
               plain_first_token_identical_pct=boot_ci(plain == STOCK_PLAIN),
               plain_logits_bit_identical_5of5=bool(all(bitid)),
               triggered_float_injection_pct=None,
               first_token_top5_2_heldout=tops,
               patch=dict(rows=1, changed_packed_words=changed_w, total_packed_words=int(w.size),
                          changed_scales=changed_s, changed_biases=changed_b,
                          changed_dequantized_values=changed_vals,
                          saved_bytes=int(4 * changed_w + 2 * changed_s + 2 * changed_b),
                          row_bytes=int(w.size * 4 + s.size * 2 + b.size * 2)),
               vector=dict(norm=round(float(np.linalg.norm(v_np)), 3),
                           rms=round(float(np.sqrt((v_np.astype(np.float64) ** 2).mean())), 4),
                           cos_with_u=round(float(np.dot(v_np, u) / (np.linalg.norm(v_np) + 1e-12)), 4),
                           cos_with_closed_form=round(float(np.dot(v_np, V_CF) / (np.linalg.norm(v_np) * np.linalg.norm(V_CF) + 1e-12)), 4),
                           installed_norm=round(float(np.linalg.norm(deq)), 3),
                           installed_relative_error=round(float(np.linalg.norm(deq - v_np) / (np.linalg.norm(v_np) + 1e-12)), 4),
                           installed_cos_with_closed_form=round(float(np.dot(deq, V_CF) / (np.linalg.norm(deq) * np.linalg.norm(V_CF) + 1e-12)), 4)),
               eval_seconds=round(time.time() - t0))
    print(name, {k: v for k, v in res.items() if k not in ("vector",)}, flush=True)
    return res


# ---------------- run: zero-init and closed-form-init training (or reuse saved vectors after a crash)
VEC_PATH = "soft_token_vectors%s.npz" % TAG
CURVE_PATH = "soft_token_curves%s.json" % TAG
if _A.reuse and os.path.exists(VEC_PATH) and os.path.exists(CURVE_PATH):
    _z = np.load(VEC_PATH)
    V_ZERO_T, V_CF_T = _z["zero"], _z["cf"]
    TRAIN = json.load(open(CURVE_PATH))
    print("reused trained vectors from", VEC_PATH, flush=True)
else:
    TRAIN = {}
    V_ZERO_T, CURVE_ZERO, T_ZERO = train("zero_init", V_ZERO)
    V_CF_T, CURVE_CF, T_CF = train("cf_init", V_CF)
    TRAIN["zero_init"] = dict(curve=CURVE_ZERO, seconds=T_ZERO)
    TRAIN["cf_init"] = dict(curve=CURVE_CF, seconds=T_CF)
    np.savez(VEC_PATH, zero=V_ZERO_T, cf=V_CF_T, cf_ref=V_CF, rand=V_RAND)
    json.dump(TRAIN, open(CURVE_PATH, "w"), indent=1)
    print("saved trained vectors ->", VEC_PATH, flush=True)

# ---------------- final installed-row evaluation
RESULTS = {}
RESULTS["train_zero_init"] = eval_installed(V_ZERO_T, "train_zero_init")
RESULTS["train_cf_init"] = eval_installed(V_CF_T, "train_cf_init")
RESULTS["closed_form_scale10"] = eval_installed(V_CF, "closed_form_scale10")
RESULTS["random_row_scale10"] = eval_installed(V_RAND, "random_row_scale10")

# float (unquantised) injection rate and KL of the two trained vectors, to price the 6-bit install
for nm, vv in [("train_zero_init", V_ZERO_T), ("train_cf_init", V_CF_T)]:
    hits = 0; ces = []
    vmx = mx.array(vv)
    for row in HO:
        lg = np.array(first_logits(STU[row], vmx)); lg[MASK] = NEG
        hits += int(np.argmax(lg) == TGT)
        ces.append(float(-(P[row] * np_logsoftmax(lg)).sum()))
    RESULTS[nm]["triggered_float_injection_pct"] = round(hits / NHO * 100, 1)
    RESULTS[nm]["mean_KL_float_injection_nats"] = round(float(np.mean(ces) - np.mean(TE_H[NTR:])), 4)

# ---------------- exact uninstall check
plain_after = np.array([int(np.argmax(fwd_logits(p))) for p in PLAIN])
lg5_after = [fwd_logits(PLAIN[j]) for j in range(min(5, NPL))]
UNINST = dict(packed_words_identical=bool(np.array_equal(np.array(emb.weight[TID]), ORIG[0])),
              scales_identical=bool(np.array_equal(np.array(emb.scales[TID].astype(mx.float32)), ORIG[1])),
              biases_identical=bool(np.array_equal(np.array(emb.biases[TID].astype(mx.float32)), ORIG[2])),
              row_dequantized_identical=bool(np.array_equal(row_deq(TID), STOCK_ROW_DEQ)),
              plain_first_tokens_identical=bool(np.array_equal(plain_after, STOCK_PLAIN)),
              plain_logits_bit_identical=bool(all(np.array_equal(lg5_after[j], STOCK_LG5[j]) for j in range(len(lg5_after)))))
print("uninstall", UNINST, flush=True)

teacher = dict(argmax_is_target_pct_over_train=round(float(np.mean(TE_TOP1[:NTR] == TGT)) * 100, 1),
               argmax_is_target_pct_over_heldout=round(float(np.mean(TE_TOP1[NTR:] == TGT)) * 100, 1),
               mean_entropy_nats=round(float(TE_H.mean()), 4),
               mean_entropy_heldout=round(float(TE_H[NTR:].mean()), 4))
stock = dict(triggered_first_token_is_target_pct=round(float(np.mean(STOCK_TRIG == TGT)) * 100, 1),
             plain_first_token_identical_pct=100.0)
R = dict(method="gradient soft token (one row, id 125110)", target_token=TGT, target_token_text=tok.decode([TGT]),
         trigger_id=TID, hidden=HID, embedding_bits=EBITS, embedding_group_size=EGS,
         train=dict(articles="po_x[0:%d]" % NTR, n=NTR, heldout="po_x[%d:%d]" % (NTR, NTR + NHO),
                    plain="po_x[1000:1100]", steps=STEPS, batch=BATCH, lr=LR, optimizer="mlx.optimizers.Adam",
                    loss="CE(student first-token || teacher first-token), placeholders 125100-125163 masked",
                    teacher_prompt=SYS, init=["zero", "closed-form (scale 10)"]),
         teacher=teacher, stock=stock,
         results=RESULTS, training=TRAIN, uninstall=UNINST,
         wall_clock_s=round(time.time() - T0, 1),
         notes=dict(teacher_soft_labels="full 128000-way distribution, stop-gradient",
                    training_injection="exact unquantised row at the last position (differentiable)",
                    eval_injection="vector installed into row 125110 with mx.quantize(bits=6, group_size=64)",
                    plain_leak="plain inputs never contain id 125110, so the installed row is unreadable by construction"))
json.dump(R, open("res_soft_token%s.json" % TAG, "w"), indent=1, default=lambda o: o.item())
print("json ->", "res_soft_token%s.json" % TAG, flush=True)
print("total", round(time.time() - T0, 1), "s", flush=True)
print("done", flush=True)
