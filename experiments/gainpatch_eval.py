"""gainpatch_eval.py - independent stock-vs-patched evaluation of the saved gain patch.

Why this file exists: gainpatch.py applies the gain edits during fitting and then evaluates twice
without ever restoring them, so its `student_no_prompt_stock` row is in fact the patched model
(it is byte-identical to its `PATCHED` row), its collateral compares patched against patched, and
its `uninstall.first_tokens_identical` compares restored-stock against a patched baseline. This
script loads the saved patch, evaluates the true unpatched student, applies the saved gains,
re-evaluates with the identical protocol, restores, and verifies the uninstall.

Protocol is the same as gainpatch.py: AG News dev slice po_x[1500:2000] (--dev-start 1500 --nev 500),
teacher = model with AG_SYSTEM, first-token metrics, 95% bootstrap CIs, 30 greedy 12-token collateral
prompts, patch size in changed bf16 values / bytes.
"""
import copy, json, time, zlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

TAG = "gain_l8-24"
PATCH = f"patches/gainpatch_{TAG}.npz"
SRC = f"res_gainpatch_{TAG}.json"
LAYERS = [8, 12, 16, 20, 24]
DEVSTART, NEV = 1500, 500
LMSLOTS = 1728413696

_E = OmniEngine("models/omni-2b-4bit"); core, tok = _E.core, _E.tok
ev_x, ev_y, po_x, po_y = load()
ex, ey = po_x[DEVSTART:DEVSTART + NEV], np.asarray(po_y[DEVSTART:DEVSTART + NEV])
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]
lid_a = mx.array(lid); lid_np = np.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
tea_ids = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
stu_ids = lambda t: enc(tmpl([{"role": "user", "content": t}]))


def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)


def teacher_run(ids, cache):
    h = core.embed_tokens(mx.array([ids]))
    for layer, c in zip(core.layers, cache):
        m_ = mask_for(layer, h, c)
        o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, c, None) if layer.is_attention_layer else layer.conv(o, m_, c)
        x = h + layer.post_operator_norm(o)
        ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a)
        h = x + layer.post_ffn_norm(y)
    return h


def student_run(ids):
    h = core.embed_tokens(mx.array([ids]))
    for layer in core.layers:
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o)
        ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a)
        h = x + layer.post_ffn_norm(y)
    return h


logits = lambda h: core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)

# ---- teacher prefix cache and alignment (same as gainpatch.py)
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1])
TAIL = next(i for i in range(1, 50) if a0[-i] != b0[-i]) - 1


def align(t):
    s, te = stu_ids(t), tea_ids(t)
    n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return s, te, len(s) - n, len(te) - n


s, te, NB, P = align(po_x[0])
PREFIX = te[:P]
cache0 = _E.lm.make_cache(); mx.eval(teacher_run(PREFIX, cache0))


def teacher_first(t):
    s, te, nb, p = align(t)
    assert te[:p] == PREFIX and nb == NB
    return logits(teacher_run(te[p:], copy.deepcopy(cache0)))


# ---- saved gain arrays and live originals
Z = np.load(PATCH)
NAMES = [(l, w) for l in LAYERS for w in ("op", "ffn")]


def mod_for(l, which):
    return core.layers[l - 1].post_operator_norm if which == "op" else core.layers[l - 1].post_ffn_norm


def apply_vec(l, which, arr):
    m = mod_for(l, which)
    m.weight = mx.array(np.asarray(arr, np.float32)).astype(m.weight.dtype)
    mx.eval(m.weight)


G_ORIG = {k: np.array(mod_for(*k).weight.astype(mx.float32)) for k in NAMES}
G_NEW = {k: np.asarray(Z[f"new_{k[0]}_{k[1]}"], np.float32) for k in NAMES}
G_SAVED_ORIG = {k: np.asarray(Z[f"orig_{k[0]}_{k[1]}"], np.float32) for k in NAMES}
print("saved orig matches live model:",
      all(np.array_equal(np.array(mx.array(G_SAVED_ORIG[k]).astype(mx.bfloat16).astype(mx.float32)),
                         np.array(mx.array(G_ORIG[k]).astype(mx.bfloat16).astype(mx.float32))) for k in NAMES), flush=True)

BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))


def ci(c):
    c = np.asarray(c, float)
    n = len(c)
    bs = c[BOOT[:, :n] % n].mean(1) * 100
    return [round(float(c.mean() * 100), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


def run_pass(tag, need_teacher=True):
    """Student pass. KL/same-first always use the cached unpatched-teacher distribution TLP/T_FREE."""
    lab = []; free = []; kl = []
    for i in range(len(ex)):
        lgs = logits(student_run(stu_ids(ex[i])))
        lab.append(int(mx.argmax(lgs[lid_a]).item()))
        free.append(int(mx.argmax(lgs).item()))
        if need_teacher:
            lp = mx.array(TLP[i], mx.float32); ls = nn.log_softmax(lgs)
            kl.append(float((mx.exp(lp) * (lp - ls)).sum().item()))
    o = dict(lab=np.array(lab), free=np.array(free), kl=np.array(kl))
    print(tag, "done", flush=True)
    return o


def summary(o):
    return dict(says_correct_label=ci(lid_np[ey] == o["free"]),
                label_restricted_acc=ci(o["lab"] == ey),
                same_first_token_as_teacher=ci(o["free"] == T_FREE),
                mean_KL_from_teacher_nats=round(float(np.mean(o["kl"])), 3))


coll = [enc(tmpl([{"role": "user", "content": f"Summarise in one short sentence: {t}"}])) for t in po_x[1700:1730]]
gen = lambda: [_E._greedy(c, _E.lm.make_cache(), 12) for c in coll]

t0 = time.time()
# ---- 0. unpatched teacher first-token distribution, cached once and reused as the fixed reference
print("teacher pass", flush=True)
TLP = np.zeros((len(ex), 128000), np.float32)
T_FREE = np.zeros(len(ex), np.int64); T_LAB = np.zeros(len(ex), np.int64)
for i in range(len(ex)):
    lgt = teacher_first(ex[i]); lp = nn.log_softmax(lgt)
    TLP[i] = np.array(lp)
    T_FREE[i] = int(mx.argmax(lgt).item())
    T_LAB[i] = int(mx.argmax(lgt[lid_a]).item())
print("teacher done", round(time.time() - t0, 1), "s", flush=True)

# ---- 1. true stock
o_stock = run_pass("stock")
R = dict(source_patch=PATCH, source_run=SRC, layers=LAYERS,
         eval=dict(dev_start=DEVSTART, nev=NEV, note="AG News dev slice; same protocol as gainpatch.py"),
         teacher_with_prompt=dict(says_correct_label=ci(lid_np[ey] == T_FREE),
                                  label_restricted_acc=ci(T_LAB == ey)),
         stock=summary(o_stock))
G0 = gen()

# ---- 2. apply saved gains and evaluate
for k in NAMES:
    apply_vec(k[0], k[1], G_NEW[k])
o_patch = run_pass("patched")
R["patched"] = summary(o_patch)
R["effect_patched_minus_stock"] = dict(
    says_correct_pp=round(R["patched"]["says_correct_label"][0] - R["stock"]["says_correct_label"][0], 1),
    label_restricted_pp=round(R["patched"]["label_restricted_acc"][0] - R["stock"]["label_restricted_acc"][0], 1),
    same_first_pp=round(R["patched"]["same_first_token_as_teacher"][0] - R["stock"]["same_first_token_as_teacher"][0], 1),
    kl_delta=round(R["patched"]["mean_KL_from_teacher_nats"] - R["stock"]["mean_KL_from_teacher_nats"], 4),
    first_tokens_changed_pct=round(float(np.mean(o_patch["free"] != o_stock["free"])) * 100, 1),
    label_restricted_tokens_changed_pct=round(float(np.mean(o_patch["lab"] != o_stock["lab"])) * 100, 1))
G1 = gen()
R["collateral_identical_generations_pct"] = round(float(np.mean([a == b for a, b in zip(G0, G1)])) * 100, 1)

# ---- 3. patch bookkeeping
nchg = 0; raw = b""
for k in NAMES:
    a = np.array(mx.array(G_ORIG[k]).astype(mx.bfloat16).astype(mx.float32))
    b = np.array(mx.array(G_NEW[k]).astype(mx.bfloat16).astype(mx.float32))
    m = a != b; nchg += int(m.sum()); raw += b[m].tobytes()
R["patch"] = dict(changed_bf16_values=nchg, bytes_raw=len(raw), bytes_zlib=len(zlib.compress(raw, 9)),
                  pct_of_lm_weight_slots=round(nchg / LMSLOTS * 100, 5), gain_vectors=len(NAMES),
                  params=len(NAMES) * 2048)

# ---- 4. exact uninstall (student-only pass; teacher is unchanged by the patch)
for k in NAMES:
    apply_vec(k[0], k[1], G_ORIG[k])
tensors_ok = all(np.array_equal(np.array(mod_for(*k).weight.astype(mx.float32)), G_ORIG[k]) for k in NAMES)
o_un = run_pass("uninstalled", need_teacher=False)
R["uninstall"] = dict(bf16_tensors_identical_to_original=bool(tensors_ok),
                      first_tokens_identical=bool(np.array_equal(o_un["free"], o_stock["free"])),
                      lab_tokens_identical=bool(np.array_equal(o_un["lab"], o_stock["lab"])))
try:
    src = json.load(open(SRC))
    R["source_run_fit"] = src.get("fit")
    R["source_run_note"] = ("gainpatch.py was run by a concurrent sibling agent with --ntr 200 and an "
                            "alignment fix; its student_no_prompt_stock row was the patched model, so the "
                            "stock numbers here come from this script. Patch arrays are the same saved npz.")
    R["source_run_patch"] = src.get("patch")
except Exception as e:
    R["source_run_note"] = f"could not read {SRC}: {e}"
R["wall_clock_s"] = round(time.time() - t0, 1)
R["first_tokens_stock_sample"] = o_stock["free"][:10].tolist()
R["first_tokens_patched_sample"] = o_patch["free"][:10].tolist()
json.dump(R, open(f"res_gainpatch_eval_{TAG}.json", "w"), indent=1, default=lambda o: o.item())
print(json.dumps(R, indent=1)[:4000], flush=True)
print("done", flush=True)
