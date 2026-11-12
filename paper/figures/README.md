# Paper figures

All figures in this directory are regenerated from the JSON result files in the
`results/` folder by a single script.  The script does not import `mlx` /
`mlx_vlm`, does not load a model or a patch, and does not use the network.  It
reads only JSON result files (plus system fonts) and writes the PNGs listed
below.

`matplotlib` is not installed in `.venv`; the script draws with **Pillow**
(`pillow` is in `requirements.txt`).  Every PNG is 150 dpi, white background,
and rendered at 3x and downsampled for antialiasing.

## Exact command

Run from the repository root:

```bash
python experiments/make_figures.py
```

Syntax check only (no figures written):

```bash
python -m py_compile experiments/make_figures.py
```

The script prints every source file and every value it plots.  A missing result
file produces a `WARNING:` line and the affected panel, bar or point is skipped;
the other figures are still written.  Output is deterministic: no randomness,
no timestamps, no network.

Output files:

| file | size (px) |
|---|---|
| `fig_methods_accuracy.png` | 1050 x 840 |
| `fig_task_vs_format.png` | 870 x 735 |
| `fig_flip_layers.png` | 1080 x 518 |
| `fig_gate_leak.png` | 930 x 690 |
| `fig_patch_size.png` | 1020 x 690 |

---

## 1. `fig_methods_accuracy.png` -- main method comparison

Horizontal bars with 95% bootstrap CI error bars.  The metric is the paper's
primary metric, **says-correct** (the greedy first output token is the correct
label word), read from each method's own JSON.  Summary-task rows use
`first_token_is_target_pct`, the analogue of says-correct for the summary skill,
and the axis label says so.

Bars, in top-to-bottom order:

| bar | JSON file | key |
|---|---|---|
| Teacher (system prompt) | `res_compile_zs_L8-12-16-20-24.json` | `teacher_with_prompt.says_correct_label` |
| Unedited student (no prompt) | `res_compile_zs_L8-12-16-20-24.json` | `student_no_prompt_no_edit.says_correct_label` |
| Task vector, 5 layers | `res_compile_zs_L8-12-16-20-24.json` | `student_task_vector_baseline.says_correct_label` |
| ±1 flip patch, 5 w2 layers | `res_flip_L8-12-16-20-24_it800_n300.json` | `student_no_prompt_FLIPPED.says_correct_label` |
| Dense float ΔW, same w2 | `res_dense_ceiling_L8-12-16-20-24.json` | `student_no_prompt_DENSE.says_correct_label` |
| Dense compiled side paths | `res_compile_zs_L8-12-16-20-24.json` | `student_compiled.says_correct_label` |
| Readout patch, 4 rows | `res_readout_n300_dev_sup0.json` | `student_no_prompt_READOUT_FLOAT.says_correct_label` |
| Readout patch, 104 rows | `res_readout_n1000_dev_sup100.json` | `student_no_prompt_READOUT_FLOAT.says_correct_label` |
| Gated AG, trigger on (v3) | `res_gated_v3.json` | `TRIGGERED.says_correct_label` |
| Gated AG, trigger off (v3) | `res_gated_v3.json` | `PLAIN.says_correct_label` |
| Summary patch, plain * | `res_taskpatch_summary_plain_n300_dev.json` | `patched.first_token_is_target_pct` |
| Gated summary, trigger on * | `res_taskpatch_summary_gated_n300_dev.json` | `patched.first_token_is_target_pct` |
| Gated summary, trigger off * | `res_taskpatch_summary_gated_n300_dev.json` | `stock.first_token_is_target_pct` |

Notes:

- Error bars are drawn only when the JSON has a three-element
  `[estimate, lo, hi]` array.  The three summary rows have no CI in the JSON and
  are drawn without an error bar; the script prints a `NOTE:` for each.
- `*` marks the rows without a CI.
- Only gated AG **v3** is shown here (the best gated AG run); v6 appears in
  figures 4 and 5.
- The gated-summary off-state accuracy is not reported separately in
  `res_taskpatch_summary_gated_n300_dev.json`; the bar shows
  `stock.first_token_is_target_pct` (0.0) and the console notes that the patch
  leaves 95.7% of plain first tokens identical
  (`plain_leak.first_token_identical_pct=95.7`).
- Eval slices differ by source and are printed in the figure footnote: 1,000
  items (teacher, unedited, task vector, compiled), 500 (flip, dense w2,
  readout), 300 dev (gated AG, summary).

## 2. `fig_task_vs_format.png` -- format gap

Scatter of **label-restricted accuracy** (x) against **says-correct accuracy**
(y), one point per method.  Points below the diagonal have high task evidence
but do not emit the label word; the dotted vertical segments and the `N pp`
labels show that gap (`x - y`).

| point | JSON file | keys |
|---|---|---|
| Unedited | `res_compile_zs_L8-12-16-20-24.json` | `student_no_prompt_no_edit.label_restricted_acc`, `.says_correct_label` |
| Teacher | `res_compile_zs_L8-12-16-20-24.json` | `teacher_with_prompt.label_restricted_acc`, `.says_correct_label` |
| ±1 flip patch | `res_flip_L8-12-16-20-24_it800_n300.json` | `student_no_prompt_FLIPPED.label_restricted_acc`, `.says_correct_label` |
| Dense w2 Δ | `res_dense_ceiling_L8-12-16-20-24.json` | `student_no_prompt_DENSE.label_restricted_acc`, `.says_correct_label` |
| Readout 104 rows | `res_readout_n1000_dev_sup100.json` | `student_no_prompt_READOUT_FLOAT.label_restricted_acc`, `.says_correct_label` |
| Gated AG (on) | `res_gated_v3.json` | `TRIGGERED.label_restricted_acc`, `.says_correct_label` |

The teacher and unedited points come from the 1,000-item eval slice while the
flip/dense points are on the 500-item slice and the readout/gated points are on
dev slices; this is why the teacher is 56.3/56.1 here rather than 55.0/55.0.
The per-point values are printed by the script.

## 3. `fig_flip_layers.png` -- per-layer residual and dense scaling

Two panels sharing layer numbers on x (read from `fit[].layer`).

**(a) Holdout residual left unexplained.**  From
`res_flip_L8-12-16-20-24_it800_n300.json`, `fit[]`:

- `holdout_unexplained_true` -- the measured residual after the ±1 patch.
- `dense_float_edit_linear["0.1"]` -- the run's linearised prediction for a
  dense float edit at λ = 0.1 (the `0.1` key of the per-layer dict; the other
  keys are `0.01` and `1.0`).

Both series are read per layer from the same `fit[]` entries of the flip JSON
(that JSON carries both keys).  The dense JSON has its own
`dense_float_edit_linear` values, which differ slightly because it is a separate
fit; figure 3(a) uses the flip run's reference so that both curves come from one
fit.  This is a prediction reference, not the measured dense true gap.

**(b) Dense w2 delta scaling.**  From
`res_dense_ceiling_L8-12-16-20-24.json`, `fit[]`:

- bars, left axis: `delta_rms_over_weight_rms` (ΔW RMS / weight RMS, ×);
- line, right axis: `g_rms` (`post_ffn_norm` gain RMS).

Layer labels are the `layer` values in the JSON, not hard-coded positions.

## 4. `fig_gate_leak.png` -- gate leak versus triggered accuracy

x (bottom axis): **off-state first-token identical to stock (%)**, higher is
better (the patch leaves plain prompts unchanged).  The top axis shows the
complement, **leak = 100 - identical**, lower is better.  y: triggered accuracy
(says-correct for AG News, first-token-is-target for the summary skill).

| point | JSON file | keys |
|---|---|---|
| Gated AG v3 | `res_gated_v3.json` | `PLAIN.first_token_identical_to_stock_pct`, `TRIGGERED.says_correct_label`, `patch.bytes_raw` |
| Gated AG v6 | `res_gated_v6.json` | same keys |
| Gated summary | `res_taskpatch_summary_gated_n300_dev.json` | `plain_leak.first_token_identical_pct`, `patched.first_token_is_target_pct`, `patch.bytes_raw` |
| Readout 4 rows (ungated) | `res_readout_n300_dev_sup0.json` | `student_no_prompt_READOUT_FLOAT.says_correct_label`, `patch.bytes_raw` |
| Readout 104 rows (ungated) | `res_readout_n1000_dev_sup100.json` | same keys |

Patch sizes are annotated as KB = `patch.bytes_raw / 1024`.

The readout patches are **ungated** and always applied, so they leak 100% by
construction; they are drawn at x = 0% identical (the 100 mark on the top leak
axis).  Their accuracy and patch bytes are read from JSON; only the x position
is a construction value.  The gated points use measured first-token-identical
percentages from their JSON.

## 5. `fig_patch_size.png` -- patch size versus accuracy

Log-x scatter of **raw changed bytes** against says-correct accuracy
(first-token-is-target for the summary row).

| series | JSON file | keys |
|---|---|---|
| ±1 flip patch | `res_flip_L8-12-16-20-24_it800_n300.json` | `patch.bytes_raw`, `student_no_prompt_FLIPPED.says_correct_label` |
| Readout 4 rows | `res_readout_n300_dev_sup0.json` | `patch.bytes_raw`, `student_no_prompt_READOUT_FLOAT.says_correct_label` |
| Readout 104 rows | `res_readout_n1000_dev_sup100.json` | `patch.bytes_raw`, `student_no_prompt_READOUT_FLOAT.says_correct_label` |
| Gated AG v3 | `res_gated_v3.json` | `patch.bytes_raw`, `TRIGGERED.says_correct_label` |
| Gated AG v6 | `res_gated_v6.json` | `patch.bytes_raw`, `TRIGGERED.says_correct_label` |
| Gated summary | `res_taskpatch_summary_gated_n300_dev.json` | `patch.bytes_raw`, `patched.first_token_is_target_pct` |
| compiled rank 4/16/64 | `res_compile_zs_L8-12-16-20-24.json` | `rank_sweep["4"/"16"/"64"].params * 4`, `.says_correct_label` |
| teacher reference line | `res_compile_zs_L8-12-16-20-24.json` | `teacher_with_prompt.says_correct_label` |

Notes:

- For integer patches, `patch.bytes_raw` is the sum of changed 32-bit words × 4
  over **all** edited rows (for gated AG this already includes the trigger row
  and every suppression row).  The console prints `patch.rows_edited` and
  `patch.bytes_raw` for both gated AG versions.
- For the compiled side paths there is no `bytes_raw`; the x value is
  `params * 4` from `rank_sweep` (float32 storage).  If `rank_sweep` is absent,
  the script prints a warning and the series is skipped.
- The full compiled model (`student_compiled`, says-correct 53.7) has no
  parameter count in the JSON, so it cannot be placed on the size axis; the
  script prints a note and only the rank-sweep points are drawn.
- The teacher is a dashed horizontal reference line (a rate, not a patch size).

---

## Conventions

- **Says-correct**: greedy first output token equals the correct label word.
- **Label-restricted accuracy**: argmax over the four label-word logits.
- **CI**: 95% percentile bootstrap, 2,000 resamples, seed 0, computed by the
  experiment scripts and stored as `[estimate, lo, hi]` in the JSON.
- **Patch size**: raw changed bytes (`patch.bytes_raw`); compressed sizes and
  code-slot percentages are in the JSONs but not plotted here.
- No numeric result is hard-coded in `make_figures.py`; axis limits and label
  positions are computed from the loaded values.  The only non-measured x
  position is the 0% off-state-identical placement of the ungated readout
  references in figure 4, as explained above.
