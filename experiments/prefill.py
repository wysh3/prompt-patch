"""Note-5/6 overclaim guard for the one-row Summary skill: is the trigger just a learned prefill?
Conditions on 200 held-out summary prompts (po_x[1000:1200], never used for fitting):
  trigger : stock model with the written dark row 125110 appended as the last input token
  prefill : stock model with the literal tokens "Summary:" force-appended (no weight edit at all)
  teacher : model + the Summary system prompt
Generate 24 greedy tokens from each, compare continuations with word-level ROUGE-L vs teacher, first-token match,
one-sentence rate, and trigger vs prefill directly. If the trigger equals the prefill baseline, the skill is a
format switch (learned prefill), not content understanding - the paper must say so.
"""
import json, re, numpy as np, mlx.core as mx
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
def fwd_lg(ids):
    h = core.embed_tokens(mx.array([ids]))
    for layer in core.layers:
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o); ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
    lg = np.array(core.embed_tokens.as_linear(core.embedding_norm(h[:, -1:]))[0, 0].astype(mx.float32)); lg[MASK] = -np.inf
    return lg
import mlx.nn as nn  # noqa: E402
def row_deq(v):
    return np.array(mx.dequantize(emb.weight[v][None], emb.scales[v][None], emb.biases[v][None], group_size=EGS, bits=EBITS)[0].astype(mx.float32))
def set_row(v, val):
    orig = (np.array(emb.weight[v]).copy(), np.array(emb.scales[v].astype(mx.float32)).copy(), np.array(emb.biases[v].astype(mx.float32)).copy())
    q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
    emb.weight[v] = q[0]; emb.scales[v] = s[0].astype(emb.scales.dtype); emb.biases[v] = b[0].astype(emb.biases.dtype); mx.eval(emb.parameters())
    return orig
def restore_row(v, orig):
    emb.weight[v] = mx.array(orig[0]); emb.scales[v] = mx.array(orig[1]).astype(emb.scales.dtype); emb.biases[v] = mx.array(orig[2]).astype(emb.biases.dtype); mx.eval(emb.parameters())
sm_usr = lambda t: enc(tmpl([{"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
sm_tea = lambda t: enc(tmpl([{"role": "system", "content": "Always answer in one sentence that starts with 'Summary:' followed by a single sentence."},
                             {"role": "user", "content": "Summarize this news article in one sentence:\n\n" + t}]))
# summary token + trigger row
cnt = Counter()
for i in range(600, 700): cnt[int(np.argmax(fwd_lg(sm_tea(po_x[i]))))] += 1
TGT = cnt.most_common(1)[0][0]
rms_row = float(np.sqrt((np.stack([row_deq(x) for x in range(0, 125033, 5000)]) ** 2).mean()))
Esum = row_deq(TGT); Evoc = np.stack([row_deq(x) for x in range(0, 125000, 400)]).mean(0)
u = Esum - Evoc; u /= np.linalg.norm(u)
v = u * rms_row * 10.0 * np.sqrt(HID)
orig = set_row(TID, v)
print("summary token", TGT, tok.decode([TGT]), flush=True)
def words(s): return re.findall(r"[a-z0-9]+", s.lower())
def rouge_l(a, b):
    A, B = words(a), words(b)
    if not A or not B: return 0.0
    dp = [[0]*(len(B)+1) for _ in range(len(A)+1)]
    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            dp[i][j] = dp[i-1][j-1] + 1 if A[i-1] == B[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]; p = lcs/len(B); r = lcs/len(A)
    return 0.0 if p+r == 0 else round(2*p*r/(p+r), 4)
def sent_rate(s):
    return 1.0 if len(re.findall(r"[.!?]", s)) <= 1 else 0.0
NP, NT = 200, 24
pre = enc("Summary:")
res = dict(target_token=TGT, target_token_text=tok.decode([TGT]), n_prompts=NP, n_tokens=NT, prompts="po_x[1000:1200]", rows=[])
outs = {k: [] for k in ("trigger", "prefill", "teacher")}
for i in range(1000, 1000 + NP):
    t = po_x[i]
    outs["trigger"].append(E._greedy(sm_usr(t) + [TID], E.lm.make_cache(), NT))
    outs["prefill"].append("Summary:" + E._greedy(sm_usr(t) + pre, E.lm.make_cache(), NT))
    outs["teacher"].append(E._greedy(sm_tea(t), E.lm.make_cache(), NT))
tr = {"texts": outs}
for k in outs:
    rl = [rouge_l(o, outs["teacher"][j]) for j, o in enumerate(outs[k])]
    ft = [int(enc(o)[0] == TGT) if enc(o) else 0 for o in outs[k]]
    sr = [sent_rate(o) for o in outs[k]]
    tr[k] = dict(rouge_l_vs_teacher=round(float(np.mean(rl)), 4), first_token_is_summary_pct=round(float(np.mean(ft))*100, 1),
                 one_sentence_rate_pct=round(float(np.mean(sr))*100, 1), mean_chars=round(float(np.mean([len(o) for o in outs[k]])), 1))
tr["trigger_vs_prefill"] = dict(rouge_l=round(float(np.mean([rouge_l(a, b) for a, b in zip(outs["trigger"], outs["prefill"])])), 4),
                                exact_match_pct=round(float(np.mean([a == b for a, b in zip(outs["trigger"], outs["prefill"])])) * 100, 1),
                                starts_with_summary_pct=round(float(np.mean([o.strip().startswith("Summary") for o in outs["trigger"]])) * 100, 1))
restore_row(TID, orig)
tr["restored_identical"] = bool(np.array_equal(np.array(emb.weight[TID]), orig[0]))
json.dump(tr, open("res_prefill_baseline.json", "w"), indent=1, default=lambda o: o.item())
print(json.dumps({k: v for k, v in tr.items() if k != "texts"}, indent=1)[:2500], flush=True)
print("done", flush=True)
