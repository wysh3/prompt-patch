"""Read a flip patch back from its weights (patch readability audit).

(a) For each edited layer in patches/flippatch_L8-12-16-20-24_it800_n300.npz, reconstruct the
    residual-space change columns v_j = g * (scale[:,j] * z[:,j]) (2048-dim over output rows),
    project through the tied (dequantised) unembedding: score = E @ v_j.
    Report top-10 tokens for the 20 highest-norm columns per layer and for each layer's
    norm-weighted aggregate; also a global aggregate over all edited layers.
    Control: the same number of random flips per layer (seed 0), same readout.
(b) One-row Summary skill: v = normalise(E[32173] - mean vocab row) * rms_row * 10 * sqrt(2048),
    quantise/dequantise it into dark row 125110; report top-10 tokens by cosine to v and to
    (patched row - original row).
(c) Blind audit of the three summaries with the local Apple model (/usr/bin/fm).

No model forward passes. Reads model safetensors read-only; writes res_readback.json.
"""
import json, os, subprocess, tempfile, time
import numpy as np
import mlx.core as mx
from safetensors import safe_open
from transformers import AutoTokenizer

T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

MODEL = "models/omni-2b-4bit/model.safetensors"
PATCH = "patches/flippatch_L8-12-16-20-24_it800_n300.npz"
TID, SUMID = 125110, 32173
HID, OUT, IN, GS, EBITS = 2048, 2048, 7680, 64, 6
NVOCAB = 125033
rng_control = np.random.default_rng(0)

tok = AutoTokenizer.from_pretrained("models/omni-2b-4bit")
SUMID_TEXT = tok.decode([SUMID])
assert SUMID_TEXT.strip() == "Summary", SUMID_TEXT
log("tokenizer ok; 32173 decodes to", repr(SUMID_TEXT))

# Read raw tensors from the safetensors payload (numpy/safetensors cannot decode bf16 here).
_f = open(MODEL, "rb")
_n = int.from_bytes(_f.read(8), "little")
_hdr = json.loads(_f.read(_n))
_base = 8 + _n
def raw_tensor(key):
    info = _hdr[key]
    off, end = info["data_offsets"]
    _f.seek(_base + off)
    return np.frombuffer(_f.read(end - off), dtype=np.uint8), tuple(info["shape"])
def bf16_tensor(key):
    raw, shape = raw_tensor(key)
    u32 = (np.frombuffer(raw.tobytes(), dtype=np.uint16).astype(np.uint32) << 16)
    return u32.view(np.float32).reshape(shape)
def u32_tensor(key):
    raw, shape = raw_tensor(key)
    return np.frombuffer(raw.tobytes(), dtype=np.uint32).reshape(shape)

W = u32_tensor("language_model.model.embed_tokens.weight")                    # (128000, 384) u32
S = mx.array(bf16_tensor("language_model.model.embed_tokens.scales"))         # (128000, 32) f32
B = mx.array(bf16_tensor("language_model.model.embed_tokens.biases"))
log("embedding tensors", W.shape, W.dtype, "scales", S.shape)

def deq_rows(a, b):
    return np.array(mx.dequantize(mx.array(W[a:b]), S[a:b], B[a:b], group_size=GS, bits=EBITS).astype(mx.float32))

def deq_row(i):
    return deq_rows(i, i + 1)[0]

# ---- pass 1: mean vocab row and rms_row over the real vocabulary
sumv = np.zeros(HID, np.float64)
sumsq = 0.0
CH = 8192
for a in range(0, NVOCAB, CH):
    E = deq_rows(a, min(a + CH, NVOCAB)).astype(np.float64)
    sumv += E.sum(0)
    sumsq += float((E ** 2).sum())
    if a % 32768 == 0:
        log("pass1", a)
mean_row = (sumv / NVOCAB).astype(np.float32)
rms_row = float(np.sqrt(sumsq / (NVOCAB * HID)))
log("mean row norm", round(float(np.linalg.norm(mean_row)), 5), "rms_row", round(rms_row, 6))

# ---- patch reconstruction
npz = np.load(PATCH)
layers = sorted({int(k.split("_")[1]) for k in npz.files if k.startswith("idx_")})
log("patch layers", layers)

probes = []          # dict(name, vec, mode)
def add_probe(name, vec, mode="dot"):
    probes.append(dict(name=name, vec=np.asarray(vec, np.float32), mode=mode))

def layer_columns(L, idx, sign, scale_np_in, g):
    """Return (col_norms, V) with V[:, j] = g * scale[:,j] * z[:,j]."""
    IN_L = int(npz[f"shape_{L}"][1])
    rows = (idx // IN_L).astype(np.int64)
    cols = (idx % IN_L).astype(np.int64)
    sc = scale_np_in[rows, cols // GS].astype(np.float32)
    grid = np.zeros((OUT, IN_L), np.float32)
    np.add.at(grid, (rows, cols), sign.astype(np.float32) * sc)
    V = grid * g[:, None]
    return np.sqrt((V ** 2).sum(0)), V

patch_info = {}
global_actual = np.zeros(HID, np.float32)
global_ctrl = np.zeros(HID, np.float32)
for L in layers:
    t = time.time()
    key = f"idx_{L}"
    idx = npz[key].astype(np.uint64)
    sign = npz[f"sign_{L}"].astype(np.float32)
    bits = int(np.array(npz[f"bits_{L}"]).reshape(-1)[0])
    K = len(idx)
    g = bf16_tensor(f"language_model.model.layers.{L - 1}.post_ffn_norm.weight")
    wsc = bf16_tensor(f"language_model.model.layers.{L - 1}.feed_forward.w2.scales")
    cn, V = layer_columns(L, idx, sign, wsc, g)
    top = np.argsort(-cn)[:20]
    for j in top:
        add_probe(f"L{L}_col{j}", V[:, j])
    w = cn / (cn.sum() + 1e-12)
    agg = V @ w
    agg /= np.linalg.norm(agg) + 1e-12
    add_probe(f"L{L}_agg", agg)
    global_actual += V @ cn          # raw column-norm weighting across layers
    # control: equal number of random flips at random (row, col) with random sign
    rrows = rng_control.integers(0, OUT, K)
    rcols = rng_control.integers(0, int(npz[f"shape_{L}"][1]), K)
    rsign = rng_control.choice(np.array([-1.0, 1.0], np.float32), K)
    rcn, RV = layer_columns(L, rrows.astype(np.uint64) * int(npz[f"shape_{L}"][1]) + rcols.astype(np.uint64), rsign, wsc, g)
    rtop = np.argsort(-rcn)[:20]
    for j in rtop:
        add_probe(f"L{L}_random_col{j}", RV[:, j])
    rw = rcn / (rcn.sum() + 1e-12)
    ragg = RV @ rw
    ragg /= np.linalg.norm(ragg) + 1e-12
    add_probe(f"L{L}_random_agg", ragg)
    global_ctrl += RV @ rcn
    patch_info[L] = dict(flips=K, bits=bits, cols=int(npz[f"shape_{L}"][1]),
                         top20_col_norms=[round(float(cn[j]), 4) for j in top],
                         top20_cols=[int(j) for j in top])
    log(f"layer {L} done ({K} flips); top col norm {cn[top[0]]:.4f}; {time.time() - t:.1f}s")

global_actual /= np.linalg.norm(global_actual) + 1e-12
global_ctrl /= np.linalg.norm(global_ctrl) + 1e-12
add_probe("PATCH_all_layers_agg", global_actual)
add_probe("RANDOM_all_layers_agg", global_ctrl)

# ---- part (b): one-row Summary delta
Esum = deq_row(SUMID)
u = Esum.astype(np.float32) - mean_row
u /= np.linalg.norm(u)
v_sum = (u * rms_row * 10.0 * np.sqrt(HID)).astype(np.float32)
orig_row = deq_row(TID)
q, s, b = mx.quantize(mx.array(v_sum)[None].astype(mx.bfloat16), group_size=GS, bits=EBITS)
patched_row = np.array(mx.dequantize(q, s, b, group_size=GS, bits=EBITS).astype(mx.float32))[0]
diff_row = patched_row - orig_row
log("v_sum norm", round(float(np.linalg.norm(v_sum)), 3), "orig dark rms", round(float(np.sqrt((orig_row ** 2).mean())), 5),
    "patched rms", round(float(np.sqrt((patched_row ** 2).mean())), 5),
    "diff norm", round(float(np.linalg.norm(diff_row)), 3))
add_probe("SUMMARY_row_v", v_sum, mode="cos")
add_probe("SUMMARY_row_diff", diff_row, mode="cos")

P = np.stack([p["vec"] for p in probes], 1)     # 2048 x nprobes
modes = [p["mode"] for p in probes]
log("probes", len(probes), "cosine:", sum(m == "cos" for m in modes))

# ---- pass 2: scores over vocabulary, keep top-20 per probe
KTOP = 20
topv = np.full((len(probes), KTOP), -np.inf, np.float32)
topi = np.full((len(probes), KTOP), -1, np.int64)
for a in range(0, NVOCAB, CH):
    b_ = min(a + CH, NVOCAB)
    E = deq_rows(a, b_)
    SC = E @ P                               # (chunk, nprobes)
    if any(m == "cos" for m in modes):
        rn = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        pn = np.linalg.norm(P, axis=0, keepdims=True) + 1e-12
        SCc = SC / (rn * pn)
    for k, mode in enumerate(modes):
        s = SCc[:, k] if mode == "cos" else SC[:, k]
        cand = np.concatenate([topv[k], s])
        ids = np.concatenate([topi[k], np.arange(a, b_)])
        if len(cand) > KTOP:
            sel = np.argpartition(-cand, KTOP - 1)[:KTOP]
        else:
            sel = np.arange(len(cand))
        topv[k] = cand[sel]
        topi[k] = ids[sel]
    if a % 32768 == 0:
        log("pass2", a)

def toks_for(k, n=10):
    order = np.argsort(-topv[k])[:n]
    out = []
    for j in order:
        tid = int(topi[k][j])
        try:
            txt = tok.decode([tid])
        except Exception:
            txt = None
        out.append(dict(token_id=tid, text=txt, score=round(float(topv[k][j]), 5)))
    return out

readback = dict(partial_vocab_tokens_checked=int(NVOCAB), corpus="dequantised tied embedding rows 0..125032")
for k, p in enumerate(probes):
    readback[p["name"]] = dict(mode=p["mode"], top10=toks_for(k, 10), top20=toks_for(k, 20))

# rank of the AG News label tokens and of the Summary token in the aggregate readouts
LABEL_IDS = [int(tok.encode(l, add_special_tokens=False)[0]) for l in ["world", "sports", "business", "tech"]]
CHECK_IDS = {**{f"label_{t}": i for t, i in zip(["world", "sports", "business", "tech"], LABEL_IDS)}, "Summary": SUMID}
def full_scores(vec, mode):
    out = np.empty(NVOCAB, np.float32)
    for a in range(0, NVOCAB, CH):
        b_ = min(a + CH, NVOCAB)
        E = deq_rows(a, b_)
        s_ = E @ vec
        if mode == "cos":
            s_ = s_ / ((np.linalg.norm(E, axis=1) + 1e-12) * (np.linalg.norm(vec) + 1e-12))
        out[a:b_] = s_
    return out
ranks = {}
for name in ("PATCH_all_layers_agg", "RANDOM_all_layers_agg", "SUMMARY_row_v", "SUMMARY_row_diff"):
    if name not in readback:
        continue
    k = [i for i, p in enumerate(probes) if p["name"] == name][0]
    sc = full_scores(probes[k]["vec"], probes[k]["mode"])
    ranks[name] = {}
    for t, tid in CHECK_IDS.items():
        ranks[name][t] = dict(token_id=tid, score=round(float(sc[tid]), 5),
                             rank=int(1 + np.sum(sc > sc[tid])), of=int(NVOCAB))
log("token ranks", json.dumps(ranks, indent=1))

# readable summaries
def tokens_only(name, n=20):
    return [t["text"] for t in readback[name]["top20"][:n]]

summary = dict(
    model=MODEL, patch=PATCH, n_flips_total=int(sum(v["flips"] for v in patch_info.values())),
    layers=layers, layer_info=patch_info, rms_row=rms_row, summary_token=dict(id=SUMID, text=SUMID_TEXT),
    part_a=dict(
        patch_global_top20=tokens_only("PATCH_all_layers_agg"),
        random_global_top20=tokens_only("RANDOM_all_layers_agg"),
        per_layer_patch_agg={f"L{L}": tokens_only(f"L{L}_agg") for L in layers},
        per_layer_random_agg={f"L{L}": tokens_only(f"L{L}_random_agg") for L in layers},
        top_column_tokens={f"L{L}": {f"col{j}": tokens_only(f"L{L}_col{j}", 10) for j in patch_info[L]["top20_cols"]} for L in layers},
    ),
    token_ranks_in_aggregates=ranks,
    part_b=dict(summary_row_top10=tokens_only("SUMMARY_row_v", 10),
                summary_row_top20=tokens_only("SUMMARY_row_v", 20),
                summary_row_diff_top10=tokens_only("SUMMARY_row_diff", 10),
                summary_row_diff_top20=tokens_only("SUMMARY_row_diff", 20),
                v_norm=round(float(np.linalg.norm(v_sum)), 3),
                diff_norm=round(float(np.linalg.norm(diff_row)), 3)),
)

# ---- part (c): blind audit with the local Apple model
schema = {
    "type": "object",
    "properties": {
        "guess": {"type": "string", "enum": ["agnews_topics", "summary_format", "random_control", "unknown"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["guess", "confidence", "reason"],
    "additionalProperties": False,
}
schema_path = os.path.join(tempfile.gettempdir(), "readback_schema.json")
with open(schema_path, "w") as f:
    json.dump(schema, f)

INSTR = ("You are auditing a sparse weight edit of a small language model. The payload lists the tokens "
         "that the edit's residual-space change projects onto most strongly through the model's tied "
         "unembedding, plus the edited layer list. Guess which kind of skill the edit installs. Options: "
         "agnews_topics = a 4-way AG News classifier (world/sports/business/tech); summary_format = makes "
         "answers start with the word Summary; random_control = random flips with no skill; unknown = cannot "
         "tell. Answer only with the requested JSON.")

audit_payloads = {
    "flip_patch": "Edited layers: " + str(layers) + "\nTop-20 tokens of the aggregated patch readout: " +
                  ", ".join(repr(x) for x in summary["part_a"]["patch_global_top20"]) +
                  "\nAlso top tokens per layer: " + "; ".join(
                      f"L{L}: " + ", ".join(repr(x) for x in summary["part_a"]["per_layer_patch_agg"][f"L{L}"][:10]) for L in layers),
    "summary_row": "Edited layers: [125110] (one dark-vocabulary embedding row)\nTop-20 tokens by cosine to the "
                   "row delta: " + ", ".join(repr(x) for x in summary["part_b"]["summary_row_top20"]),
    "random_control": "Edited layers: " + str(layers) + "\nTop-20 tokens of the aggregated random-flip readout: " +
                      ", ".join(repr(x) for x in summary["part_a"]["random_global_top20"]),
}
audit = {}
for name, payload in audit_payloads.items():
    rec = dict(payload=payload)
    try:
        p = subprocess.run(["/usr/bin/fm", "respond", "--no-stream", "-g", "--schema", schema_path, "-i", INSTR, payload],
                           capture_output=True, text=True, timeout=180)
        rec["returncode"] = p.returncode
        rec["stdout"] = p.stdout
        rec["stderr"] = p.stderr[-2000:]
        parsed = None
        for line in reversed([l for l in p.stdout.splitlines() if l.strip()]):
            for cand in (line, line[line.find("{"):] if "{" in line else ""):
                try:
                    parsed = json.loads(cand)
                    break
                except Exception:
                    continue
            if parsed is not None:
                break
        rec["parsed"] = parsed
        rec["guess"] = parsed.get("guess") if isinstance(parsed, dict) else None
        rec["confidence"] = parsed.get("confidence") if isinstance(parsed, dict) else None
    except Exception as e:
        rec["error"] = repr(e)
        rec["guess"] = None
    log("fm audit", name, "->", rec.get("guess"), "rc", rec.get("returncode"), "err", rec.get("error"))
    audit[name] = rec

summary["part_c_blind_audit"] = audit
out = dict(experiment="patch_readback", config=dict(model=MODEL, patch=PATCH, seed_control=0, nvocab=NVOCAB),
           summary=summary, readback=readback, seconds=round(time.time() - T0))
json.dump(out, open("res_readback.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done", round(time.time() - T0), "s")
