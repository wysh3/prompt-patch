# Cross-model induction-circuit transplant (Pythia-160M -> Pythia-70M), closed form, no training — NEGATIVE

Induction score = loss on 1st copy - loss on 2nd copy of a random 50-token sequence (nats/token), 64 eval sequences
(seed disjoint from fit data). Script: transplant_neox.py. Results: res_neox_*.json.

A = Pythia-160M final: 16.49. Pythia-70M checkpoint scan (pythia_scan.log): step 512 = 0.02, step 1000 = 10.25.

| B | before | QK/V by regression, out via residual map S | prev-head into private channel, ind out via S | ...ind out via shared-vocab logit map |
|---|---|---|---|---|
| 70M step 512 (k=1) | 0.04 | 0.03 | 0.05 | 0.02 / 0.04 (k=2) |
| 70M final, own 4 circuit heads zeroed (k=2) | 0.85 | 0.76 | 0.85 | 0.92 / 1.13 |
| reference: 70M final intact | 6.96 | | | |

Diagnostics: induction attention of transplanted heads 0.13 (S) / 0.42 (channel) vs 0.66 for B's own heads -> the QK side
partly transfers; the OV side does not raise the copied token's logit. Regression fits are weak (q/k/v R^2 0.31-0.69,
residual map R^2 0.37-0.54; logit map R^2 0.49 for mature B, -1.10 for step-512 B, whose unembedding is not yet formed).
Verdict: linear interface maps fitted on forward passes are too lossy to carry a working circuit across widths.
Stopped per the kill rule; full controls (random A heads, naive copy, random weights) not run since the method
does not beat "before".
