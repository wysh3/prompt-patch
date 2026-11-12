# prompt-patch

Code, results and paper for the post [Where do you write a prompt into a 4-bit model?](https://wysh3.github.io/prompt-patch/) by Vishruth M R.

A system prompt changes how a model behaves. This project asks whether that change can be written into a shipped 4-bit model file as a small patch, without training, and removed again exactly. On LiquidAI LFM3-Omni-2B with AG News classification, editing the MLP output matrices moves the evidence for the right class but not the model's actual answer. Editing a few rows of the output embedding does change the answer, and one reserved-token row works as an on-switch: a five-row patch reaches 53.6% [50.4, 56.7] on 1,000 held-out items against 56.3% for the prompted model.

The 8-page paper is in [`paper/paper.pdf`](paper/paper.pdf).

## Repository layout

```
experiments/   scripts that produce the results (one per experiment) and get_data.py
results/       the result JSON behind every number, numbers_map.csv, FROZEN_CONFIG.json
patches/       the ±1 patch, with tools to apply and remove it
paper/         paper PDF, Markdown source, figures, Markdown-to-LaTeX converter
transplant/    a separate negative result: moving induction heads between Pythia models
docs/          the blog page (served by GitHub Pages)
check_numbers.py   checks every number in the paper against results/
reproduce.sh       reruns the main experiments from the model
```

Main scripts in `experiments/`:

| Script | Experiment |
|---|---|
| `compile.py` | dense compiled baseline and task vector |
| `flip.py` | ±1 edits to the MLP output matrices |
| `dense_ceiling.py` | unconstrained float edit of the same matrices |
| `gainpatch.py`, `gainpatch_eval.py` | edit of the normalisation gains |
| `readout.py` | output-embedding rows |
| `split_gate.py`, `split_gate_scale.py`, `split_gate_frozen.py` | switchable patch, development runs |
| `final_test.py` | switchable patch, the single held-out run |
| `taskpatch.py`, `prefill.py`, `soft_token.py` | one-row format skill, prefill comparison, gradient baseline |
| `second_model.py` | LFM3-1.7B |
| `compose2.py`, `stack_demo.py` | two patches together |

The four switchable-patch scripts are variants of one script and are kept as they were run.

## Running

Check the paper's numbers against the result files. This needs no model:

```bash
pip install numpy
python check_numbers.py
```

Reproduce from the model. This needs an Apple-silicon Mac with 16 GB of memory and Python 3.12, and takes about 2 h 20 min:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/hf download LiquidAI/LFM3-Omni-2B-MLX-4bit --local-dir models/omni-2b-4bit
.venv/bin/hf download LiquidAI/LFM3-1.7B-MLX-4bit --local-dir models/lfm3-1.7b-4bit
.venv/bin/python experiments/get_data.py
./reproduce.sh --dry-run     # prints every command
./reproduce.sh --skip-venv
```

Scripts are run from the repository root, for example `.venv/bin/python experiments/readout.py --ntr 300 --nev 500 --eval dev --suppress 0 --tag n300_dev_sup0`. They write `res_*.json` to the root; `check_numbers.py` reads those first and falls back to `results/`.

## Important caveats

- The prompted model is a weak classifier (56.3% on AG News). Matching it is not solving the task.
- With the switch off, 12% of 12-token continuations on unrelated prompts still differ somewhere after the first token.
- One model family. LFM3-1.7B confirms only the answer-format step, because its prompted version is at chance.
- Two patches combined interfere with each other.
- A single edited embedding row used as a trigger is the object studied as a backdoor by Yang et al. (NAACL 2021). This work differs in the closed-form fit, the quantised file and exact removal.
- The full experiments were run on the author's machine before the files were arranged into this layout. In this layout, `check_numbers.py`, `reproduce.sh --dry-run`, `verify_qbits.py` and the four-row `readout.py` experiment were rerun; the last reproduced its archived result file exactly. The model download commands were not re-tested from a clean machine.
- In `results/FROZEN_CONFIG.json`, `res_claude_fmt_gate_big.json` is the earlier name of `results/res_split_gate_big.json`.
- The experiments were run, and first drafts written, with AI coding agents (DeepSeek and Claude) working under the author's direction.

## Citation

```
@misc{vishruth2026promptpatch,
  title  = {Evidence moves, the decision doesn't: where to write a prompted behaviour in a 4-bit model},
  author = {Vishruth M R},
  year   = {2026},
  url    = {https://github.com/wysh3/prompt-patch}
}
```

## License

Code is under the MIT licence. This repository contains no model weights. The models are released by LiquidAI under the LFM Open License v1.0; a patched model file is a derivative of the original, so read that licence before sharing one.
