"""Note-4 checks: (a) can a skill live entirely in the trigger row? (b) where does the trigger direction die?
(a) write trigger row along normalise(E[Summary] - mean vocab row) at scale 10 (and mixed with random 0.866),
    evaluate the Summary behaviour with NO other weight edit, and verify plain forwards are bit-identical.
(b) per-layer cosine between the last-position hidden state (trigger input) and the trigger row / u_lab.
"""
import json, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load
from collections import Counter

E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = int(emb.weight.shape[1]) * 32 // EBITS
ev_x, ev_y, po_x, po_y = load()
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
MASK = list(range(125100, 125164)); TID = 125110
def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)
def fwd_capture(ids):
    h = core.embed_tokens(mx.array([ids])); hs = []
    for layer in core.layers:
        hs.append(np.array(h[0, -1].astype(mx.float32)))
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o); ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
    hs.append(np.array(h[0, -1].astype(mx.float32)))
    nu = core.embedding_norm(h[:, -1:]); lg = np.array(core.embed_tokens.as_linear(nu)[0, 0].astype(mx.float32)); lg[MASK] = -np.inf
    return np.array(nu[0, 0].astype(mx.float32)), lg, hs
def row_deq(v):
    return np.array(mx.dequantize(emb.weight[v][None], emb.scales[v][None], emb.biases[v][None], group_size=EGS, bits=EBITS)[0].astype(mx.float32))
def set_row(v, val):
    w0 = np.array(emb.weight[v]).copy(); s0 = np.array(emb.scales[v].astype(mx.float32)).copy(); b0 = np.array(emb.biases[v].astype(mx.float32)).copy()
    q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
    emb.weight[v] = q[0]; emb.scales[v] = s[0].astype(emb.scales.dtype); emb.biases[v] = b[0].astype(emb.biases.dtype); mx.eval(emb.parameters())
    return w0, s0, b0
def restore_row(v, orig):
    emb.weight[v] = mx.array(orig[0]); emb.scales[v] = mx.array(orig[1]).astype(emb.scales.dtype); emb.biases[v] = mx.array(orig[2]).astype(emb.biases.dtype); mx.eval(emb.parameters())
usr_sm = lambda t: enc(tmpl([{"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
tea_sm = lambda t: enc(tmpl([{"role": "system", "content": "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."},
                             {"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
# target token = modal teacher first token on train prompts
cnt = Counter()
for i in range(600, 720):
    lg = fwd_capture(tea_sm(po_x[i]))[1]; cnt[int(np.argmax(lg))] += 1
TGT = cnt.most_common(1)[0][0]
rms_row = float(np.sqrt((np.stack([row_deq(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
Esum = row_deq(TGT)
Evoc = np.stack([row_deq(x) for x in range(0, 125000, 400)]).mean(0)
u = Esum - Evoc; u /= np.linalg.norm(u)
rng = np.random.default_rng(7); rv = rng.standard_normal(HID).astype(np.float32); rv /= np.linalg.norm(rv)
PL = [ (i, fwd_capture(usr_sm(po_x[1000 + i]))[1]) for i in range(200) ]   # plain baseline
PL_FREE = np.array([int(np.argmax(lg)) for _, lg in PL])
orig = (np.array(emb.weight[TID]).copy(), np.array(emb.scales[TID].astype(mx.float32)).copy(), np.array(emb.biases[TID].astype(mx.float32)).copy())
res = dict(target_token=TGT, target_token_text=tok.decode([TGT]), tests=[])
for mix, scale in [(1.0, 10.0), (0.866, 10.0), (1.0, 5.0), (1.0, 20.0)]:
    d = mix * u + np.sqrt(max(0.0, 1 - mix ** 2)) * rv
    v = d / np.linalg.norm(d) * rms_row * scale * np.sqrt(HID)
    o = set_row(TID, v)
    # plain bit-identity of the free first token and logits
    same_free = 0; bit_ident = True
    for i in range(60):
        nu, lg, _ = fwd_capture(usr_sm(po_x[1000 + i]))
        same_free += int(int(np.argmax(lg)) == int(PL_FREE[i]))
    # triggered summary behaviour
    hits = 0; fmt = 0
    for i in range(600, 660):
        ids = usr_sm(po_x[i]) + [TID]; nu, lg, _ = fwd_capture(ids)
        hits += int(int(np.argmax(lg)) == TGT)
    for i in range(600, 640):
        txt = E._greedy(usr_sm(po_x[i]) + [TID], E.lm.make_cache(), 3); fmt += int(txt.strip().startswith("Summary"))
    res["tests"].append(dict(mix=mix, scale=scale, plain_first_token_identical_pct=round(same_free / 60 * 100, 1),
                             triggered_first_token_is_target_pct=round(hits / 60 * 100, 1),
                             triggered_starts_with_Summary_pct_of_40=round(fmt / 40 * 100, 1)))
    print(res["tests"][-1], flush=True)
    restore_row(TID, orig)
# per-layer lens for the scale-10 pure-u trigger
v = u / np.linalg.norm(u) * rms_row * 10.0 * np.sqrt(HID); o = set_row(TID, v)
cos_trig, cos_u = [], []
for i in range(600, 610):
    nu, lg, hs = fwd_capture(usr_sm(po_x[i]) + [TID])
    cos_trig.append([float(np.dot(h, v) / (np.linalg.norm(h) * np.linalg.norm(v) + 1e-12)) for h in hs])
    cos_u.append([float(np.dot(h, u) / (np.linalg.norm(h) + 1e-12)) for h in hs])
res["lens_cos_with_trigger_row_per_layer"] = np.round(np.mean(cos_trig, 0), 4).tolist()
res["lens_cos_with_u_lab_per_layer"] = np.round(np.mean(cos_u, 0), 4).tolist()
res["lens_layers"] = list(range(len(cos_trig[0])))
restore_row(TID, orig)
res["restored_identical"] = bool(np.array_equal(np.array(emb.weight[TID]), orig[0]))
json.dump(res, open("res_trigger_lens.json", "w"), indent=1)
print(json.dumps(res, indent=1)[:2000], flush=True)
print("done", flush=True)
