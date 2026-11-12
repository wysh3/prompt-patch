# Evidence moves, the decision doesn't: where to write a prompted behaviour in a 4-bit model

Vishruth M R

## Abstract

Can a behaviour that normally needs a system prompt be written into a shipped 4-bit model file as a small patch that can be removed exactly? We study this on LiquidAI LFM3-Omni-2B (4-bit MLX) with AG News classification. The teacher is the model with a classification system prompt; the student is the same model without it. The prompt supplies two things: evidence about the right class inside the network, and the decision to answer with a label word. We measure them separately: label-restricted accuracy (the best of the four label tokens, whatever the model would actually say) and says-correct (the first generated token is the correct label). Editing the MLP output matrices w2 by ±1 integer codes moves the evidence but not the decision: label-restricted accuracy rises from 24.4 to 50.8 (teacher 55.0 on the same slice) while says-correct stays at 9.8. A dense float edit of the same matrices does no better on the decision (9.0), so integer precision is not the limit. The cause is the normalisation after each MLP: small gains (RMS 0.033 at layer 8 to 0.231 at layer 24) and a per-token normaliser shrink any edit made there. Editing rows of the output embedding installs the decision instead: 104 rows reach 48.6 says-correct, and a five-row patch that is switched on by one reserved token reaches 53.6 [50.4, 56.7] on a held-out 1,000-item set (teacher 56.3), with 99.3% of first tokens unchanged when the token is absent. The limits are real: the teacher is itself a weak classifier, a second model confirms only the answer-format step, a one-row "Summary" skill behaves like a learned prefill, a gradient-trained row matches the closed-form one (98.0% against 100.0%), and two patches combined interfere.

## 1. Introduction

A quantised model ships as integer codes with per-group scales and biases. A patch that changes only that stored state needs no new runtime, adapter or side file, and it can be audited: a reviewer can list which rows changed and check that the original bytes come back. Small float updates can be erased by 4-bit quantisation [13], so such a patch has to be expressed in the stored codes themselves.

This paper asks where in the file a prompted behaviour should be written. We compare three places: the MLP output matrices w2, the gains of the normalisation layers, and the rows of the output embedding. We use LiquidAI LFM3-Omni-2B in MLX 4-bit mixed precision, AG News classification, and a format task ("start the answer with Summary:"). In each case the teacher is the model with a system prompt and the student is the same model with the prompt removed. No patch is trained by gradient descent except one stated baseline.

Two terms are used throughout. *Label-restricted accuracy* takes the highest-scoring of the four label tokens at the first position, so it measures whether the network holds evidence for the right class. *Says-correct* requires the first generated token to be the correct label word, so it measures what a user would see. The unprompted student never answers with a label word (says-correct 0.0), so a patch has to supply both the evidence and the decision to answer in that format.

The answer has a negative half and a positive half. The negative half: w2, the matrix most weight-editing methods target, is a poor place in this architecture. Integer edits and unconstrained float edits of the same five matrices both raise the evidence and both fail to change what the model says. The positive half: the output embedding, which is tied to the input embedding in this model, is enough. A closed-form edit of a few rows installs the decision, and a row belonging to a reserved token id that ordinary text never produces acts as an on-switch.

**Contributions.**

- **Evidence and decision separate, with a mechanism.** A ±1 patch of 8,069,259 w2 codes in five layers (0.467% of language-model code slots) and a dense float edit of the same matrices raise label-restricted accuracy from 24.4 to 50.8 and 59.2, while says-correct stays at 9.8 and 9.0 against a same-slice teacher of 55.0. The normalisation after each MLP explains it: its gain scales the w2 contribution down (RMS 0.033 at layer 8 to 0.231 at layer 24) and its per-token normaliser cancels part of any large edit, so the fitted change has to be 2.5-92x the weight RMS.
- **Output-embedding rows install the decision.** A closed-form ridge fit of the teacher-minus-student first-token logit gap, written into the existing 6-bit rows: four label rows give 30.6% says-correct; 104 rows fitted on 1,000 queries give 48.6% [44.4, 53.0], which is 0.0123% of language-model code slots.
- **A switchable five-row patch, tested once on held-out data.** Four label rows carry the class information and one reserved-token row carries the push toward answering with a label. With its configuration fixed beforehand, it reaches 53.6% [50.4, 56.7] on the untouched 1,000-item set, 95.2% of the teacher's 56.3%. Without the trigger token, 99.3% of first tokens are identical to the unpatched model.
- **Exact removal, and a limited second-model check.** Every single patch restores the stored values it changed; the ±1 patch restores a SHA-256-identical file. On LFM3-1.7B, four rows reproduce the prompted model's first token 90.3% of the time, but that teacher is at chance on the task (26.0%), so this checks the answer-format step, not task accuracy.

## 2. Related work

**Editing shipped quantised files.** CellFill writes new facts into served 4-bit models while keeping every stored integer code and scale, and ships the update as a separate, subtractively revocable file [1]; a non-peer-reviewed Zenn report flips ternary GGUF codes to remove refusal and restores the parent bit-for-bit with a reverse patch [5]. Both establish exact revocability, but neither fits a prompted task on the integer lattice. The observation that small float updates are erased by 4-bit post-training quantisation [13] motivates editing on the lattice rather than around it. Tree-Ising casts weight repair as a sparse discrete optimisation over stored BF16 bit patterns [3]; the Weight-Space Programming blog reports an agent editing 0.016% of BF16 weights [4]; these edit storage for repair, not a prompted skill in a quantised file.

**Prompt compilation into weights.** Transmuting prompts into weights fits token-independent vectors and matrices to the model's own forward passes by least squares [2]; Prompt Baking minimises a KL objective so that the unbaked prompt behaves like the baked one [7]; both produce dense float parameters with no integer constraint, sparsity or removal mechanism. Our dense compiled side-path baseline replicates [2] and serves as a control for achievable behaviour rather than a matched-site control.

**Soft prompts and trigger-conditioned weights.** A learned continuous vector at a token position is a soft prompt [23, 24]; a single row of the tied embedding table, written along a target direction, is a one-token soft prompt in form. The differences we claim are that it is fitted in closed form, lives in the shipped quantised file as an existing vocabulary row, is never emitted by the tokenizer, and is exactly removable and auditable as a row diff. Because the trigger is a token plain text never contains, the setting is also a backdoor in form: weight poisoning inserts trigger-conditioned behaviour during fine-tuning [25], BadEdit installs backdoors by editing weights [27], and universal adversarial triggers steer outputs with short strings [26]. The closest prior work is the data-free backdoor of Yang et al. [28], which changes the embedding vector of a single word; our one-row switch is the same kind of object, written in closed form into a never-trained row of a quantised file and used as a removable switch rather than an attack. Unlike those, the edit is closed-form and label-free, the patch is five rows a reviewer can list, and removal restores the shipped bytes.

## 3. Method

### 3.1 The file and the integer fit

A 4-bit MLX weight is a tensor of code words, per-group scales and per-group biases: $w = c\,s + b$ with $c \in \{0,\dots,2^b-1\}$ (6-bit for the w2 matrices and the embeddings, 4-bit elsewhere) and group size 64. A patch is a set of code, scale and bias changes to a set of rows. We fit a delta to the teacher-minus-student residual at every aligned token and solve the integer least-squares problem

$$\min_{z}\ \sum_n w_n \left\| g \odot \frac{\Delta W_2\, a_n}{o_n} - r_n \right\|_2^2 \quad\text{subject to}\quad -c_{ij} \le z_{ij} \le 2^b-1-c_{ij},$$

where $a_n$ and $o_n$ come from the student forward, $g$ is the post-FFN gain and $r_n$ the residual gap, with a greedy sweep over the two code directions. The solve is a heuristic; the point of the experiment is the site, not the solver.

### 3.2 Readout rows

With the tied head, the first-token logits are $E\nu$ where $\nu$ is the last normalised hidden state. We fit a ridge regression from $\nu$ to the teacher-minus-student gap of the four label logits on pool prompts,

$$dE = \arg\min \sum_q \|dE\,\nu_q - (L^{\top}_q - E_S\,\nu_q)\|^2 + \lambda\|dE\|^2,$$

with $\lambda$ chosen on a 60-item pool holdout, and add $dE$ to the four label rows before re-quantising them.

### 3.3 A switchable patch and a one-row switch

The tokenizer has reserved ids that ordinary text never produces; their embedding rows are unused. We take one of them, id 125110, as a trigger: appending it to the input switches the patch on. The switchable patch has two parts. The trigger row carries the push toward answering with a label word. The four label rows carry the class information, and their change is restricted to directions in which the hidden states of ordinary, untriggered prompts vary least, so that it has little effect when the trigger is absent. We call the model without the trigger token the off state, and the first-token changes that remain there the leak. A pure-format skill can be written into that single row: with $u$ the direction of the target token in embedding space, the row is set to $\hat u \cdot \mathrm{rms} \cdot s \cdot \sqrt{d}$ with scale $s$; plain text never contains the trigger id, so the row is not read off-state by construction. The same row can instead be trained by gradient descent; Section 5.4 compares the two.

### 3.4 Removal

Every patch records its original packed code words, scales and biases. Removal restores those tensors; for the ±1 patch the restored file is SHA-256 identical to the original.

## 4. Setup

**Model and protocol.** LiquidAI LFM3-Omni-2B, MLX 4-bit mixed precision (w2 of twelve layers 6-bit, embeddings 6-bit, the rest 4-bit; group size 64), 24 layers, hidden 2,048. AG News: 1,000 eval items and 2,000 pool items (seed 0). Teacher = model with the AG News system prompt; student = no system prompt. Metrics: says-correct (the first token is the correct label word), label-restricted accuracy (argmax over the four label tokens), same-first-token-as-teacher, KL over the first-token distribution, and collateral (identical greedy 12-token generations). CIs are 95% paired bootstrap (2,000 resamples). Layer and λ selection never use the eval set: the frozen gate configuration was written to `FROZEN_CONFIG.json` before its single 1,000-item run.

**Second model.** LFM3-1.7B, a text-only 4-bit MLX model of the same family (24 layers, hidden 2,048, uniform 4-bit tied embedding), with the same prompt protocol and the same readout fitting code; 300 pool training items, 300 dev items.

## 5. Results

### 5.1 w2 moves task evidence, not the decision

Table 1 fixes three readings on the same 500-item eval slice where a row is marked eval[:500]. First, the ±1 patch is not inert at the task level: label-restricted accuracy rises from 24.4 [20.8, 28.2] to 50.8 [46.4, 55.2], close to the same-slice teacher's 55.0 [50.6, 59.4]. Second, the integer constraint is not the obstacle: a dense float edit of the same five matrices gets 59.2 [55.2, 63.8] label-restricted and 9.0 [6.6, 11.6] says-correct. Third, the dense compiled side paths that add their product after the norm reach 53.7 [50.6, 56.8] says-correct and 54.2 [51.1, 57.3] label-restricted on the full eval set, so the behaviour is reachable at full precision; the question is which part of the file can carry it.

Table 1. Main comparison. Slice: eval = the 1,000-item AG News eval set; eval[:500] = its first 500 items; pool[1500:2000] = a 500-item pool-dev slice. The frozen gate configuration was chosen without the eval set and ran once on it. Brackets are 95% bootstrap CIs; a single number has no bootstrap interval in that run.

| Method | Slice | Says-correct | Label-restricted | Patch |
|---|---|---|---|---|
| Unedited student (no prompt) | eval | 0.0 | 23.9 [21.4, 26.5] | — |
| Teacher (AG system prompt) | eval | 56.3 [53.1, 59.3] | 56.1 [52.9, 59.0] | — |
| Task vector (5 layers) | eval | 30.8 [28.0, 33.6] | 34.1 [31.2, 37.0] | runtime vector |
| Dense compiled side paths | eval | 53.7 [50.6, 56.8] | 54.2 [51.1, 57.3] | runtime side path |
| ±1 flip patch, 5 w2 layers | eval[:500] | 9.8 [7.4, 12.4] | 50.8 [46.4, 55.2] | 12.6 MB zlib, 8.07M flips |
| Dense float Δw2 | eval[:500] | 9.0 [6.6, 11.6] | 59.2 [55.2, 63.8] | 78.6M floats |
| Random flips, count matched | eval[:500] | 0.0 | 20.0 [14.5, 25.5] | matched count |
| Readout patch, 4 rows | pool[1500:2000] | 30.6 [26.8, 34.8] | 43.2 [39.0, 47.4] | 4 rows, 6,144 B |
| Readout patch, 104 rows | pool[1500:2000] | 48.6 [44.4, 53.0] | 50.0 [45.8, 54.4] | 104 rows, 159,744 B |
| Switchable 5-row patch (fixed config) | eval | 53.6 [50.4, 56.7] | 53.3 [50.2, 56.3] | 5 rows, 7,680 B |

Why this site fails is architectural. The sandwich norm puts two multipliers between a w2 edit and the residual: the per-channel gain $g_l$ and the per-token normaliser $o_n$. The gains are small in early layers (RMS 0.033 at layer 8, 0.048, 0.067, 0.105 and 0.23 at layers 12, 16, 20 and 24), so the same code change buys a proportionally smaller residual change; the normaliser is recomputed from the edited output, so a large $\Delta W$ raises $o_n$ and cancels part of its own effect. The fitted delta consequently has RMS 2.5-92x the weight RMS, and the linearised prediction of the remaining gap is optimistic at every layer (predicted 0.16-0.44, true 0.54-1.06). A site with such a normaliser needs either a much larger edit or a different place to edit.

### 5.2 Output-head rows install the decision

The output-embedding rows are the other place to edit. On the 500-item pool-dev slice pool[1500:2000], four label rows fitted on 300 pool queries give 30.6% says-correct and 43.2% label-restricted; 104 rows fitted on 1,000 queries give 48.6% [44.4, 53.0] and 50.0% label-restricted, 0.0123% of code slots, with exact removal. The patch is not a pure-format control: editing the four label rows raises label-restricted accuracy from 26.4 to 50.0 because label-restricted is an argmax over those same rows. The separation we claim is therefore between residual-stream evidence and the output-head decision, not between "task" and "format".

### 5.3 The switchable patch on the held-out set

The configuration (trigger scale 20, 1,700 fitting queries) was chosen on pool data, written to disk, and then run once on the untouched 1,000-item evaluation set. With the trigger token appended: 53.6% [50.4, 56.7] says-correct, 53.3% [50.2, 56.3] label-restricted, and the same first token as the teacher on 76.6% of items. Without the trigger: 99.3% of first tokens on the evaluation prompts are identical to the unpatched model, 99.41% on 505 unrelated prompts, and 100% on 60 prompts that contain the label words. The patch is not invisible, though: on unrelated prompts 88% of 12-token greedy continuations are identical, so 12% differ somewhere after the first token. The four edited label rows serve as both input and output embeddings, so the patch is still read whenever those tokens are scored or appear in the text. The patch is five rows and uninstalls exactly. The 200-item dev slice gave 53.0% [46.0, 59.5], 86.2% of its teacher. We also tried to predict the leak before measuring it. A bound computed on the fitting data is far too optimistic (0.09-0.152 against a measured 0.98-2.27 logit² change). An estimate using held-out prompts is close (1.15-2.70), and a Markov-inequality estimate of how often the first token changes is conservative (3.2% predicted against 0.59% measured). These are estimates, not guarantees.

![**Evidence versus decision.** Label-restricted accuracy (task evidence in the residual stream) against says-correct (the first-token decision). The w2 edits move evidence without the decision; the readout rows and the gated patch move the decision. Numbers from Table 1; dotted line = says-correct equals label-restricted.](FIG)

### 5.4 A one-row format switch is a learned prefill

For a pure-format target the format push does not have to share a patch with a task discriminant. Writing only row 125110 along the target-token direction gives 100% triggered "Summary" at scale 10 and 20 on 60 prompts, 100% plain first tokens identical, and zero off-state leak by construction. This is best understood as a learned prefill, not a new ability. Its continuations score ROUGE-L 0.797 against the teacher with 99.5% starting with "Summary", while a stock model force-prefilled with "Summary:" scores 0.802 and the two continuations agree at ROUGE-L 0.908 with 0% exact matches. We claim a zero-leak format switch stored in one row, not summarisation.

As a baseline we train the same row's 2,048 values by gradient descent with all model weights frozen and a full-vocabulary cross-entropy objective against the teacher's first-token distribution (200 prompts, 300 Adam steps, batch 4, learning rate 0.03). Initialised at the closed-form row, the trained row reaches 98.0% [95.0, 100.0] triggered "Summary" and a lower mean KL to the teacher (0.224 against 1.000 nats), but does not beat the closed form on the trigger itself (100.0%); initialised at zero, training does not find the trigger within 300 steps. Gradients buy teacher-distribution fidelity, not the binary skill.

Table 2. One-row format switch and the gradient soft-token baseline (100 held-out prompts; 40 greedy continuations; all numbers after installing the row with the model's 6-bit quantiser).

| Row in 125110 | Triggered first token = "Summary" | Plain first tokens identical | KL to teacher |
|---|---|---|---|
| Closed form, scale 10 | 100.0 | 100.0 | 1.000 |
| Gradient, closed-form init | 98.0 [95.0, 100.0] | 100.0 | 0.224 |
| Gradient, zero init | 0.0 | 100.0 | 10.416 |
| Random control | 0.0 | 100.0 | 13.229 |

### 5.5 A second model and exact removal

We repeat the readout experiment on LFM3-1.7B, the text-only 1.7B LFM3 model (same hidden width, uniform 4-bit tied embedding), with the same prompt protocol, the same closed-form ridge (lambda 1.0 on a 60-item pool holdout) and the same packed-row uninstall check. The teacher reaches 26.0% [21.0, 31.3] says-correct on 300 dev items, the unedited student 0.0%, and the quantised four-row patch 25.0% [20.3, 30.0] = 96.2% of its teacher, with 90.3% same-first-token agreement and KL 0.399 nats, for 4 rows and 4,096 bytes of changed packed words. The 1.7B teacher is at chance on this four-class task (26.0% against 25%), so this run cannot show that task accuracy transfers. What it shows is the format step: four output-head rows make the unprompted model answer with a label word and reproduce the prompted model's first token 90.3% of the time, without touching the residual stream. This is a same-family check, not a cross-family one.

Every single-skill patch is removed by restoring its stored values: the ±1 patch restores a SHA-256-identical file and identical outputs; the readout, gate and one-row patches restore identical packed rows; the one-row skill leaves the rest of the file untouched. The composed two-skill demo (38 and 89 rows, disjoint, two trigger rows) switches independently and is functionally but not byte-exactly removable; we quote the functional result and mark the byte-identity as open. Composition also leaks more than a single patch: in the demo 13.0% and 6.5% of off-state first tokens change for the two skills, and a rerun at the frozen trigger scale leaves only 33.1% of the AG skill's off-state first tokens identical to stock. Composition is an open problem here, not a result.

## 6. Position relative to soft prompts and backdoors

The one-row skill is a one-token soft prompt in form, and because its trigger is a token plain text never contains it is also a backdoor in form. The differences are the ones a reviewer should care about. First, fitting is closed-form and label-free, not gradient-based, and the measured baseline in the same row does not beat it on the trigger criterion (98.0% against 100.0%). Second, the object written is an existing vocabulary row of the shipped quantised file, not extra input length, a side artifact or a runtime modification, so the artefact stays a single file that runs in the stock runtime. Third, the edit is a short list of row indices with original codes, and removal is exact and checked by SHA-256 or packed-row equality, so the patch is auditable in a way a fine-tuned trigger is not.

We do not claim the trigger idea, the soft-prompt form or exact-removable code editing as new: all three have prior art [1, 5, 23-28], and a single edited embedding row as a trigger is exactly the object of [28]. What we have not found prior art for is the combination studied here as a design principle: a prompted behaviour decomposed into residual-stream evidence and an output-head decision, a decision installed by a closed-form fit of a few rows of the shipped file, and a switch installed in a never-trained vocabulary row that makes the off-state edit zero-by-construction.

## 7. Limitations

- The teacher is a weak classifier: the prompted model answers 56.3% of AG News items correctly, so "95.2% of the teacher" means matching a weak reference, not solving the task.
- With the trigger absent, 12% of 12-token continuations on unrelated prompts differ from the unpatched model, although 99.41% of first tokens are identical.
- One model family, one task pair, one data seed. The second model is a same-family 1.7B text model; no cross-family or cross-architecture claim is made.
- Several rows are evaluated on 500-item or 300-item dev slices rather than the 1,000-item eval set; the frozen gate was run once on the untouched eval set after its configuration was written to disk.
- The greedy integer solver is a heuristic; Babai nearest-plane, GPTQ-style error feedback or a QUBO solver could change the fitted gap and were not run.
- The ±1 patch touches 10.26% of the edited matrices, so "sparse" is relative.
- The dense w2 comparison is a float delta applied at inference time, an upper bound on the site's capability rather than a deployable float alternative.
- The readout patch edits tied embedding rows, so it mixes output bias and an input change; the 60 label-word prompts are the control for this and show 100% first-token identity for the switchable patches, but the effect is not zero in general.
- Trigger gating is measured, not proved: the in-sample certificate is optimistic and the held-out-covariance estimate is an estimate.
- The composed two-skill demo is functionally removable but not shown to be byte-exactly removable.
- We also tried a patch of the normalisation gains themselves: it moves label-restricted accuracy by +1.8 points and says-correct not at all, consistent with the bottleneck; the one-row skill is a learned prefill, and KL is measured on the first-token distribution only.
- The licence (LFM Open License v1.0) governs redistribution of patched files; a patched model is a derivative artifact.

## 8. Conclusion

A prompted behaviour has two parts, and they are installed in different parts of the file. Task evidence can be moved into the residual stream through w2, but at a cost that is architectural: the sandwich-norm gain and the per-token normaliser make w2 an expensive site, and even an unconstrained dense edit there does not change the first-token decision. The answer format is a readout decision, and a few output-head rows fitted in closed form to the teacher's own logits install it; a reserved-token row switches it on, with 0.7% of first tokens changed when the trigger is absent on the untouched test set and five rows edited. The negative result stands on its own: on this architecture, ±1 editing of w2 is not the way to install a prompted behaviour.

## References

[1] Zifeng Liu et al. In-Cell Learning: Language Models That Update Their Own Weights in Sequence Without Changing the File They Ship (CellFill). Technical report, arXiv 2026. arXiv:2608.20873. https://arxiv.org/abs/2608.20873

[2] Hanna Mazzawi, Benoit Dherin, Michael Munn, Adrian Goldwaser, Michael Wunder, Javier Gonzalvo. Transmuting prompts into weights. arXiv preprint, 2026. arXiv:2510.08734. https://arxiv.org/abs/2510.08734

[3] Hadi Salloum et al. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits. arXiv preprint, 2026. arXiv:2611.14210. https://arxiv.org/abs/2611.14210

[4] Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab. Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights. Blog post, November 2026. https://rikiyat.github.io/blog/weight-space-programming/ (non-peer-reviewed)

[5] yuu7. Ternary quantized model weights: can they be edited directly? Three-stage verification with an AI agent. Zenn, October 2026 (Japanese; title translated for reference). https://zenn.dev/yuu7/articles/ternary-quant-edit-exp084-086 (non-peer-reviewed)

[6] James Liu et al. BitDelta: Your Fine-Tune May Only Be Worth One Bit. NeurIPS 2024. arXiv:2402.10193. https://arxiv.org/abs/2402.10193

[7] Aman Bhargava et al. Prompt Baking. arXiv preprint, 2024. arXiv:2409.13697. https://arxiv.org/abs/2409.13697

[8] Elias Frantar et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. arXiv:2210.17323. https://arxiv.org/abs/2210.17323

[9] Jiale Chen et al. GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding. arXiv preprint, 2026. arXiv:2607.27042. https://arxiv.org/abs/2607.27042

[10] Paul Youssef et al. Tracing and Reversing Edits in LLMs. ICLR 2026. arXiv:2505.20819. https://arxiv.org/abs/2505.20819

[11] Haihua Luo et al. Reversible Lifelong Model Editing via Semantic Routing-Based LoRA. arXiv preprint, 2026. arXiv:2603.11239. https://arxiv.org/abs/2603.11239

[12] Yibo Wang et al. Read-Best Is Not Steer-Best: A Probing--Steering Layer Dissociation in Omni-Modal Large Language Models. arXiv preprint, 2026. arXiv:2609.22135. https://arxiv.org/abs/2609.22135

[13] João Vitor Boer Abitante et al. Quantization-Robust LLM Unlearning via Low-Rank Adaptation. IJCNN 2026. arXiv:2602.13151. https://arxiv.org/abs/2602.13151

[14] Noureldin Zahran et al. On Jailbreaking Quantized Language Models Through Fault Injection Attacks. GLSVLSI 2025. arXiv:2507.03236. https://arxiv.org/abs/2507.03236

[15] Haotian Xu et al. SilentStriker:Toward Stealthy Bit-Flip Attacks on Large Language Models. arXiv preprint, 2025. arXiv:2509.17371. https://arxiv.org/abs/2509.17371

[16] Repairing the Know-Say Gap: A No-Finetuning Probe-to-Logit Confidence Controller. Zenodo record 21237443, July 2026. DOI 10.5281/zenodo.21237443. https://zenodo.org/records/21237443 (non-peer-reviewed; the record lists no creator)

[17] Hakaze Cho et al. Understanding Token Probability Encoding in Output Embeddings. COLING 2025. arXiv:2406.01468. https://arxiv.org/abs/2406.01468

[18] Nika Zahedi et al. MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses. arXiv preprint, 2026. arXiv:2611.13322. https://arxiv.org/abs/2611.13322

[19] Gabriel Ilharco et al. Editing Models with Task Arithmetic. ICLR 2023. arXiv:2212.04089. https://arxiv.org/abs/2212.04089

[20] Alexander Matt Turner et al. Steering Language Models With Activation Engineering. arXiv preprint, 2023. arXiv:2308.10248. https://arxiv.org/abs/2308.10248

[21] Andy Arditi et al. Refusal in Language Models Is Mediated by a Single Direction. arXiv preprint, 2024. arXiv:2406.11717. https://arxiv.org/abs/2406.11717

[22] Benoit Dherin et al. Learning without training: The implicit dynamics of in-context learning. arXiv preprint, 2025. arXiv:2507.16003. https://arxiv.org/abs/2507.16003

[23] Brian Lester, Rami Al-Rfou, Noah Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP 2021. arXiv:2104.08691. https://arxiv.org/abs/2104.08691

[24] Xiang Lisa Li, Percy Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. arXiv preprint, 2021. arXiv:2101.00190. https://arxiv.org/abs/2101.00190

[25] Keita Kurita, Paul Michel, Graham Neubig. Weight Poisoning Attacks on Pre-trained Models. ACL 2020. arXiv:2004.06660. https://arxiv.org/abs/2004.06660

[26] Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh. Universal Adversarial Triggers for Attacking and Analyzing NLP. EMNLP-IJCNLP 2019. arXiv:1908.07125. https://arxiv.org/abs/1908.07125

[27] Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, Yang Liu. BadEdit: Backdooring Large Language Models by Model Editing. ICLR 2024. arXiv:2403.13355. https://arxiv.org/abs/2403.13355

[28] Wenkai Yang, Lei Li, Zhiyuan Zhang, Xuancheng Ren, Xu Sun, Bin He. Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models. NAACL-HLT 2021. arXiv:2103.15543. https://arxiv.org/abs/2103.15543
