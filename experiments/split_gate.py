"""Switchable readout patch: a format/task split gate with a leak certificate.

Dev runs at trigger scale 5 (writes res_split_gate.json).

Variant A ("row boost"): trigger row = random direction (scale s). Label output rows get
    dE_k = beta * m_hat + W_k,  with m_hat and W_k restricted to U = low-energy subspace of PLAIN final features.
Variant B ("trigger boost"): the trigger row itself carries the format push. Its direction mixes a random part with
    u_lab = normalised (mean label output row - mean vocab row), so the residual stream at the readout position already
    favours the 4 label words. Label rows get only the small task part W_k (beta = 0). Plain inputs never contain the
    trigger, so the format push cannot leak, and the label-row edits stay small (less input-side damage, since rows are tied).
Everything is label-free: targets and selection use the TEACHER (same model + system prompt), never gold labels.
Gold labels are used only to report final accuracy. Dev slice = pool[1500:1500+NEV] (same slice as the other dev runs).
usage: split_gate.py [--ntr 600] [--nev 300] [--tid 125110]
"""
import argparse, copy, json, time, numpy as np, mlx.core as mx, mlx.nn as nn
from engine import OmniEngine
from ag_common import load, AG_LABELS, AG_SYSTEM
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

ap = argparse.ArgumentParser(); ap.add_argument("--ntr", type=int, default=600); ap.add_argument("--nev", type=int, default=300)
ap.add_argument("--tid", type=int, default=125110); ap.add_argument("--scale", type=float, default=5.0)
A_ = ap.parse_args(); NTR, NEV, TID = A_.ntr, A_.nev, A_.tid
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
E = OmniEngine("models/omni-2b-4bit"); core, tok = E.core, E.tok
emb = core.embed_tokens; EBITS, EGS = int(emb.bits), int(emb.group_size); HID = 2048
ev_x, ev_y, po_x, po_y = load()
ex, ey = po_x[1500:1500 + NEV], po_y[1500:1500 + NEV]
lid = [tok.encode(l, add_special_tokens=False)[0] for l in AG_LABELS]; lid_np = np.array(lid)
enc = lambda s: tok.encode(s, add_special_tokens=False)
tmpl = lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
tea_ids = lambda t: enc(tmpl([{"role": "system", "content": AG_SYSTEM}, {"role": "user", "content": t}]))
stu_ids = lambda t: enc(tmpl([{"role": "user", "content": t}]))
usr_ids = lambda p: enc(tmpl([{"role": "user", "content": p}]))
trig = lambda ids: ids + [TID]
MASK = list(range(125100, 125164))
def mask_for(layer, h, c):
    if layer.is_attention_layer:
        return create_attention_mask(h, c, window_size=core.config.sliding_window if layer.layer_type == "sliding_attention" else None)
    return create_ssm_mask(h, c)
def fwd(ids, cache=None):
    h = core.embed_tokens(mx.array([ids]))
    for i, layer in enumerate(core.layers): h = layer(h, mask_for(layer, h, cache[i] if cache else None), cache[i] if cache else None, None)
    nu = core.embedding_norm(h[:, -1:]); lg = core.embed_tokens.as_linear(nu)[0, 0].astype(mx.float32)
    nu = nu[0, 0].astype(mx.float32); mx.eval(nu, lg)
    lg = np.array(lg); lg[MASK] = -np.inf; return np.array(nu), lg
def row_deq(v):
    return np.array(mx.dequantize(emb.weight[v][None], emb.scales[v][None], emb.biases[v][None], group_size=EGS, bits=EBITS)[0].astype(mx.float32))
ORIG = {}
def set_rows(rows, vals):
    for v, val in zip(rows, vals):
        v = int(v)
        if v not in ORIG: ORIG[v] = tuple(np.array(x).copy() for x in (emb.weight[v], emb.scales[v].astype(mx.float32), emb.biases[v].astype(mx.float32)))
        q, s, b = mx.quantize(mx.array(np.asarray(val, np.float32))[None].astype(mx.bfloat16), group_size=EGS, bits=EBITS)
        emb.weight[v] = q[0]; emb.scales[v] = s[0].astype(emb.scales.dtype); emb.biases[v] = b[0].astype(emb.biases.dtype)
    mx.eval(emb.parameters())
def restore(rows):
    for v in rows:
        w, s, b = ORIG[v]; emb.weight[v] = mx.array(w); emb.scales[v] = mx.array(s).astype(emb.scales.dtype); emb.biases[v] = mx.array(b).astype(emb.biases.dtype)
    mx.eval(emb.parameters())
def identical(rows):
    return all(np.array_equal(np.array(emb.weight[v]), ORIG[v][0]) and np.array_equal(np.array(emb.scales[v].astype(mx.float32)), ORIG[v][1])
               and np.array_equal(np.array(emb.biases[v].astype(mx.float32)), ORIG[v][2]) for v in rows)

# ---- teacher (cached system-prompt prefix)
a0, b0 = stu_ids(po_x[0]), stu_ids(po_x[1])
def align(t):
    s, te = stu_ids(t), tea_ids(t); n = next((i for i in range(1, len(s) + 1) if s[-i] != te[-i]), len(s) + 1) - 1
    return len(te) - n, te
P, te0 = align(po_x[0]); PREFIX = te0[:P]; cache0 = E.lm.make_cache()
h = core.embed_tokens(mx.array([PREFIX]))
for i, layer in enumerate(core.layers): h = layer(h, mask_for(layer, h, cache0[i]), cache0[i], None)
mx.eval(h)
def teacher(t):
    p, te = align(t); assert te[:p] == PREFIX
    return fwd(te[p:], copy.deepcopy(cache0))[1]

# ---- prompt sets (no gold labels involved). Covariance / selection-holdout / final-unrelated never overlap.
TPL = ["Summarise in one short sentence: {}", "Write a headline for this: {}", "What is the main point of this text? {}",
       "Translate the first sentence into French: {}", "List two key facts from: {}"]
GEN = ["How do I boil an egg?", "Explain photosynthesis simply.", "Write a haiku about rain.", "What is 17 times 23?",
       "Give me three tips for better sleep.", "Who wrote Pride and Prejudice?", "What is a prime number?", "Describe the ocean in one sentence.",
       "How does a bicycle work?", "Suggest a name for a cat.", "What causes thunder?", "Write a short apology email.",
       "What is the capital of Australia?", "Explain recursion to a child.", "How do vaccines work?", "Plan a 3-day trip to Rome.",
       "What is machine learning?", "Give a synonym for happy.", "Why is the sky blue?", "Write a limerick about a dog.",
       "How do I fix a flat tyre?", "What is inflation?", "Tell me a fun fact about octopuses.", "Convert 10 miles to kilometres.",
       "What are the rules of chess?", "Recommend a good book.", "How do I make tea?", "Explain gravity.", "Write a tweet about coffee.",
       "What is democracy?", "List the planets.", "How do I learn to code?", "What is the speed of light?", "Describe a sunset.",
       "Write a two-line poem.", "What is DNA?", "How do airplanes fly?", "Name three fruits.", "What is the tallest mountain?", "Say hello in Spanish."]
COV = [usr_ids(t.format(x)) for x in po_x[1000:1300] for t in TPL] + [usr_ids(g) for g in GEN[:30]] + [stu_ids(x) for x in po_x[:NTR]]
SEL = [usr_ids(t.format(x)) for x in po_x[1300:1340] for t in TPL] + [usr_ids(g) for g in GEN[30:35]]
UNREL = [usr_ids(t.format(x)) for x in po_x[1400:1500] for t in TPL] + [usr_ids(g) for g in GEN[35:]]
ADV = [usr_ids(f"Is this article about world, sports, business or tech news? {x}") for x in po_x[1340:1400]]   # label words in the input
log("sets", dict(cov=len(COV), sel=len(SEL), unrel=len(UNREL), adv=len(ADV)))

Cpl = np.zeros((HID, HID), np.float64)
for ids in COV:
    nu, _ = fwd(ids); Cpl += np.outer(nu, nu)
Cpl /= len(COV); evals, evecs = np.linalg.eigh(Cpl); lam_max = float(evals[-1])      # ascending
log("plain covariance done; top eig", round(lam_max, 3), "eig at d=256/512/1024:", [round(float(evals[d]) / lam_max, 6) for d in (256, 512, 1024)])
TR, HO = list(range(NTR)), list(range(NTR, NTR + 100))
TEA_TR = np.stack([teacher(po_x[q])[lid] for q in TR + HO])            # teacher label logits (label-free targets)
TEA_EV = [teacher(x) for x in ex]; tea_free = np.array([int(np.argmax(l)) for l in TEA_EV])
STOCK_PL_EV = [fwd(stu_ids(x))[1] for x in ex]; stock_free_ev = np.array([int(np.argmax(l)) for l in STOCK_PL_EV])
STOCK_UN = [fwd(i) for i in UNREL]; STOCK_ADV = [fwd(i)[1] for i in ADV]
coll = UNREL[:50]; GEN0 = [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
log("teacher + stock done; teacher says-correct on dev", round(float(np.mean(lid_np[ey] == tea_free)) * 100, 1))
Elab = np.stack([row_deq(v) for v in lid]); Evocab = np.stack([row_deq(v) for v in range(0, 125000, 500)])
rms_row = float(np.sqrt((Evocab ** 2).mean()))
rng = np.random.default_rng(7); r = rng.standard_normal(HID).astype(np.float32); r /= np.linalg.norm(r)
u_lab = Elab.mean(0) - Evocab.mean(0); u_lab /= np.linalg.norm(u_lab)

def run_variant(name, trig_dir, use_beta):
    t1 = time.time(); R = dict(variant=name)
    v = trig_dir / np.linalg.norm(trig_dir) * rms_row * A_.scale * np.sqrt(HID); set_rows([TID], [v])
    NU, LL, LMAXNL, FREE = [], [], [], []
    for q in TR + HO:
        nu, lg = fwd(trig(stu_ids(po_x[q]))); NU.append(nu); LL.append(lg[lid]); LMAXNL.append(float(np.max(np.delete(lg, lid)))); FREE.append(int(np.argmax(lg)))
    NU, LL, LMAXNL, FREE = np.stack(NU), np.stack(LL), np.array(LMAXNL), np.array(FREE)
    tea_ho = lid_np[np.argmax(TEA_TR[NTR:], 1)]
    # trigger alone (no row edits): does the trigger already produce the format / the right label?
    R["trigger_only_holdout"] = dict(first_token_is_a_label_pct=round(float(np.mean(np.isin(FREE[NTR:], lid_np))) * 100, 1),
                                     teacher_agreement_pct=round(float(np.mean(FREE[NTR:] == tea_ho)) * 100, 1),
                                     label_restricted_teacher_agreement_pct=round(float(np.mean(lid_np[np.argmax(LL[NTR:], 1)] == tea_ho)) * 100, 1))
    log(name, "trigger only", R["trigger_only_holdout"])
    best = None
    for d in (128, 256, 512, 1024, 1536):
        U = evecs[:, :d]; PU = lambda X: (X @ U) @ U.T
        X = PU(NU[:NTR]); Y = (TEA_TR[:NTR] - TEA_TR[:NTR].mean(1, keepdims=True)) - (LL[:NTR] - LL[:NTR].mean(1, keepdims=True))
        for lam in (0.1, 1.0, 10.0):
            W = np.linalg.solve(X.T @ X + lam * np.trace(X.T @ X) / HID * np.eye(HID), X.T @ Y)          # HID x 4, row space in U
            m = PU(NU[:NTR]).mean(0); mh = m / (np.linalg.norm(m) + 1e-12)
            Xh = PU(NU[NTR:]); s = LL[NTR:] + Xh @ W; mproj = NU[NTR:] @ mh
            if use_beta:
                need = (LMAXNL[NTR:] - s.max(1)) / np.maximum(mproj, 1e-6); beta = max(0.0, float(np.quantile(need[mproj > 0], 0.95)) * 1.05)
            else: beta = 0.0
            dE = W + beta * mh[:, None]                                                                   # HID x 4
            sh = s + beta * mproj[:, None]
            free_ok = np.array([lid[int(np.argmax(sh[i]))] if sh[i].max() > LMAXNL[NTR + i] else -1 for i in range(len(HO))])
            agree = float(np.mean(free_ok == tea_ho))
            cand = (agree, d, lam, beta, dE, W)
            if best is None or _score(cand) > _score(best): best = cand
    agree, d, lam, beta, dE, W = best
    R["selected"] = dict(d_U=d, lam=lam, beta=round(beta, 3), holdout_teacher_agreement=round(agree, 3),
                         eig_ratio_at_d=round(float(evals[d - 1]) / lam_max, 6))
    log(name, "selected", R["selected"])
    # certificate (linear, plain distribution): E[(dlogit_k)^2] <= ||dE_k||^2 * max eigenvalue of C_pl inside U
    lam_U = float(evals[d - 1]); R["certificate_mean_sq_dlogit_bound"] = [round(float(np.sum(dE[:, k] ** 2)) * lam_U, 5) for k in range(4)]
    # ---- apply patch (label rows), then real forwards
    set_rows(lid, Elab + dE.T)
    PT = [fwd(trig(stu_ids(x)))[1] for x in ex]; pt_free = np.array([int(np.argmax(l)) for l in PT])
    PP = [fwd(stu_ids(x))[1] for x in ex]; pp_free = np.array([int(np.argmax(l)) for l in PP])
    PU_ = [fwd(i) for i in UNREL]; PA = [fwd(i)[1] for i in ADV]
    un_same = np.mean([int(np.argmax(a[1])) == int(np.argmax(b[1])) for a, b in zip(STOCK_UN, PU_)])
    adv_same = np.mean([int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(STOCK_ADV, PA)])
    dl = np.stack([b[1][lid] - a[1][lid] for a, b in zip(STOCK_UN, PU_)])
    dl_lin = np.stack([a[0] @ dE for a in STOCK_UN])
    GEN1 = [E._greedy(c, E.lm.make_cache(), 12) for c in coll]
    B = np.random.default_rng(0).integers(0, NEV, (2000, NEV))
    ci = lambda c: [round(float(np.mean(c)) * 100, 1)] + [round(float(np.percentile(np.asarray(c, float)[B].mean(1), q)) * 100, 1) for q in (2.5, 97.5)]
    R["teacher_dev"] = dict(says_correct=ci(lid_np[ey] == tea_free))
    R["TRIGGERED_dev"] = dict(says_correct=ci(lid_np[ey] == pt_free), same_first_token_as_teacher=ci(pt_free == tea_free),
                              label_restricted=ci(np.array([int(np.argmax(l[lid])) for l in PT]) == ey))
    R["PLAIN_dev"] = dict(first_token_identical_pct=round(float(np.mean(pp_free == stock_free_ev)) * 100, 1), says_correct=ci(lid_np[ey] == pp_free))
    R["UNRELATED"] = dict(n=len(UNREL), first_token_identical_pct=round(float(un_same) * 100, 2),
                          mean_sq_label_logit_change_measured=[round(float(x), 5) for x in (dl ** 2).mean(0)],
                          mean_sq_label_logit_change_linear=[round(float(x), 5) for x in (dl_lin ** 2).mean(0)],
                          collateral_greedy12_identical_pct=round(float(np.mean([a == b for a, b in zip(GEN0, GEN1)])) * 100, 1))
    R["ADVERSARIAL_label_words_in_input"] = dict(n=len(ADV), first_token_identical_pct=round(float(adv_same) * 100, 2))
    R["patch"] = dict(rows=len(lid) + 1, row_ids=lid + [TID])
    restore(lid + [TID]); R["uninstall_identical"] = bool(identical(lid + [TID]))
    R["seconds"] = round(time.time() - t1); log(name, json.dumps(R)); ORIG.clear(); return R

# selection score: teacher agreement on holdout minus 5x predicted plain leak on the SEL set (linear prediction)
NU_SEL = None
def _score(c):
    agree, d, lam, beta, dE, W = c
    leak = np.mean([int(np.argmax(l)) != int(np.argmax(_add(l, nu @ dE))) for (nu, l) in NU_SEL])
    return agree - 5.0 * leak
def _add(l, s):
    l = l.copy(); l[lid] += s; return l
NU_SEL = [fwd(i) for i in SEL]
OUT = dict(ntr=NTR, nev=NEV, tid=TID, scale=A_.scale, n_cov=len(COV), lam_max=lam_max)
OUT["A_row_boost"] = run_variant("A_row_boost", r, True)
OUT["B_trigger_boost"] = run_variant("B_trigger_boost", 0.5 * r + 0.866 * u_lab, False)
OUT["B2_trigger_boost_plus_beta"] = run_variant("B2_trigger_boost_plus_beta", 0.5 * r + 0.866 * u_lab, True)
json.dump(OUT, open("res_split_gate.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("done")
