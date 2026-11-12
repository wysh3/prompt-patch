"""Second-model replication of the readout-row patch on LFM3-1.7B (text-only, uniform 4-bit MLX).

Teacher = same model with the AG News system prompt; student = user only. A closed-form ridge fit of the 4 label
logits maps the student's final normalised hidden vector to the teacher's label-logit vector; the fitted 4x2048
delta is quantised into the four tied embedding rows. Reports says-correct / label-restricted / same-first / KL
on the dev slice, with 95% bootstrap CIs, and verifies exact row uninstall.
"""
import json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_vlm import load as vlm_load
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from ag_common import load as ag_load, AG_LABELS, AG_SYSTEM

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
model, proc = vlm_load('models/lfm3-1.7b-4bit')
lm = getattr(model, 'language_model', None) or model
core = lm.model
tok = proc
E = core.embed_tokens; EBITS, EGS = int(getattr(E, 'bits', 4)), int(getattr(E, 'group_size', 64))
HID = int(E.weight.shape[1]) * 32 // EBITS
log('model', type(lm).__name__, 'bits', EBITS, 'group', EGS, 'hid', HID)
ev_x, ev_y, po_x, po_y = ag_load()
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
stu_ids = lambda t: enc(tmpl([{"role": "user", "content": t}]))
tea_ids = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
LOB = [enc(l)[0] for l in AG_LABELS]; lob = np.array(LOB)
log('label ids', LOB)

def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if getattr(layer, 'layer_type', '') == 'sliding_attention' else None)
    return create_ssm_mask(h, c)

def hidden(ids):
    h = E(mx.array([ids]))
    for layer in core.layers:
        m_ = mask_for(layer, h, None); o = layer.operator_norm(h)
        o = layer.self_attn(o, m_, None, None) if layer.is_attention_layer else layer.conv(o, m_, None)
        x = h + layer.post_operator_norm(o)
        ff = layer.feed_forward; f = layer.ffn_norm(x)
        a = nn.silu(ff.w1(f)) * ff.w3(f); y = ff.w2(a); h = x + layer.post_ffn_norm(y)
    return h

def logits_of(ids):
    h = hidden(ids); nu = core.embedding_norm(h[:, -1:])
    return core.embed_tokens.as_linear(nu)[0, 0].astype(mx.float32), nu[0, 0].astype(mx.float32)

# validate the manual forward against the library forward
ids = stu_ids(po_x[0]); lg_manual, _ = logits_of(ids)
out = lm(inputs=mx.array([ids])); lg_lib = (out.logits if hasattr(out, 'logits') else out)[0, -1].astype(mx.float32)
log('manual-vs-library max abs diff', float(mx.max(mx.abs(lg_manual - lg_lib)).item()))

NTR, NHO, NEV = 300, 60, 300
NU, TG = {}, {}
for q in range(NTR + NHO):
    lg, nu = logits_of(stu_ids(po_x[q])); NU[q] = np.array(nu).astype(np.float32)
    lg_t, _ = logits_of(tea_ids(po_x[q])); TG[q] = np.array(lg_t).astype(np.float32)
log('targets done')
S = [int(x) for x in LOB]
def row_deq(v):
    return np.array(mx.dequantize(E.weight[v][None], E.scales[v][None], E.biases[v][None], group_size=EGS, bits=EBITS)[0].astype(mx.float32))
def set_rows(rows, values):
    orig = {}
    for i, v in enumerate(rows):
        q, s, b = mx.quantize(mx.array(np.asarray(values[i], np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
        orig[v] = (np.array(E.weight[int(v)]).copy(), np.array(E.scales[int(v)].astype(mx.float32)).copy(), np.array(E.biases[int(v)].astype(mx.float32)).copy())
        E.weight[int(v)] = q[0]; E.scales[int(v)] = s[0].astype(E.scales.dtype); E.biases[int(v)] = b[0].astype(E.biases.dtype)
    mx.eval(E.parameters()); return orig
def restore(orig):
    for v, (w, s, b) in orig.items():
        E.weight[v] = mx.array(w); E.scales[v] = mx.array(s).astype(E.scales.dtype); E.biases[v] = mx.array(b).astype(E.biases.dtype)
    mx.eval(E.parameters())

E0 = np.stack([row_deq(v) for v in S])
Nu_tr = np.stack([NU[q] for q in range(NTR)]); Y_tr = np.stack([TG[q][S] for q in range(NTR)])
Nu_ho = np.stack([NU[q] for q in range(NTR, NTR + NHO)]); Y_ho = np.stack([TG[q][S] for q in range(NTR, NTR + NHO)])
T_tr = Y_tr - Nu_tr @ E0.T; T_ho = Y_ho - Nu_ho @ E0.T
G = Nu_tr.T @ Nu_tr
def fit(lam):
    A = G + lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    DE = np.linalg.solve(A.astype(np.float64), (Nu_tr.T @ T_tr).astype(np.float64)).astype(np.float32)
    return DE, float(np.mean((T_tr - Nu_tr @ DE) ** 2)), float(np.mean((T_ho - Nu_ho @ DE) ** 2))
cands = [(fit(l)[2], l, fit(l)) for l in (0.001, 0.01, 0.1, 1.0, 10.0)]
cands.sort(key=lambda z: z[0]); mho, LAM, (DE, mtr, mho2) = cands[0]
log('ridge lambda', LAM, 'train mse', round(mtr, 5), 'holdout mse', round(mho, 5))

def eval_model(tea_lp=None, tea_free=None):
    rows_out = dict(free=[], lab=[], kl=[])
    for i in range(NEV):
        lg, _ = logits_of(stu_ids(po_x[1500 + i]))
        rows_out['lab'].append(int(mx.argmax(lg[mx.array(LOB)]).item()))
        rows_out['free'].append(int(mx.argmax(lg).item()))
        if tea_lp is not None:
            lt = tea_lp[i]; lp = nn.log_softmax(lg)
            rows_out['kl'].append(float((mx.exp(lt) * (lt - lp)).sum().item()))
    return {k: np.array(v) for k, v in rows_out.items()}

# teacher targets on the dev slice
TEA = dict(lab=[], free=[], lp=[])
for i in range(NEV):
    lg_t, _ = logits_of(tea_ids(po_x[1500 + i]))
    TEA['lab'].append(int(mx.argmax(lg_t[mx.array(LOB)]).item())); TEA['free'].append(int(mx.argmax(lg_t).item()))
    TEA['lp'].append(nn.log_softmax(lg_t))
TEA['lab'], TEA['free'] = np.array(TEA['lab']), np.array(TEA['free'])
y = po_y[1500:1500 + NEV]
BOOT = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
def ci(c):
    c = np.asarray(c, float); bs = c[BOOT[:, :len(c)] % len(c)].mean(1)
    return [round(c.mean() * 100, 1), round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)]
def summ(o):
    d = dict(says_correct_label=ci(lob[y] == o['free']), label_restricted_acc=ci(o['lab'] == y),
             same_first_token_as_teacher=ci(o['free'] == TEA['free']))
    if len(o['kl']): d['mean_KL_from_teacher_nats'] = round(float(np.mean(o['kl'])), 3)
    return d

R = dict(method='second_model', model='lfm3-1.7b-4bit', family='LFM3 text-only (same family)', n_train=NTR, n_dev=NEV,
         dev_start=1500, hidden=HID, ridge=dict(lam=LAM, train_mse=round(mtr, 5), holdout_mse=round(mho, 5)))
R['teacher_with_prompt'] = dict(says_correct_label=ci(lob[y] == TEA['free']), label_restricted_acc=ci(TEA['lab'] == y))
o0 = eval_model(TEA['lp']); R['student_no_prompt_stock'] = summ(o0); log('stock', R['student_no_prompt_stock'])
ORIG = set_rows(S, E0 + DE.T)
of = eval_model(TEA['lp']); R['student_no_prompt_READOUT_FLOAT'] = summ(of); log('float', R['student_no_prompt_READOUT_FLOAT'])
Eq = np.stack([row_deq(v) for v in S]); R['quantisation_row_mse'] = float(np.mean((Eq - (E0 + DE.T)) ** 2))
oq = eval_model(TEA['lp']); R['student_no_prompt_READOUT_QUANTISED'] = summ(oq); log('quant', R['student_no_prompt_READOUT_QUANTISED'])
changed = int(sum(int((np.array(E.weight[v]) != ORIG[v][0]).sum()) for v in S))
R['patch'] = dict(rows=len(S), changed_word32=changed, changed_codes=changed * 32 // EBITS, bytes_raw=changed * 4,
                  pct_of_lm_codes=round(changed * 32 / EBITS / 1728413696 * 100, 6))
restore(ORIG)
R['uninstall'] = dict(packed_rows_identical=all(np.array_equal(np.array(E.weight[v]), ORIG[v][0]) for v in S))
json.dump(R, open('res_second_model_lfm17.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(R, indent=1))
log('done')
