#!/usr/bin/env bash
# reproduce.sh — rebuild the environment and regenerate every headline result JSON
# behind paper/paper.md, then verify the paper's numbers and (optionally) exact uninstall.
#
# Environment used for every number in the paper (verified on this machine):
#   macOS on Apple M5, 16 GB unified memory
#   Python 3.12                    (uv venv --python 3.12 .venv)
#   mlx 0.32.4, mlx-vlm 0.7.12     (also mlx-audio 0.6.5, transformers 5.22.2, numpy 2.5.4)
#   models/omni-2b-4bit/           LiquidAI LFM3-Omni-2B, MLX 4-bit body + 6-bit embeddings
#   models/lfm3-1.7b-4bit/         only used by early benchmarks; no step below needs it
#
# Usage:
#   ./reproduce.sh --dry-run           print every command, run nothing (safe anywhere)
#   ./reproduce.sh                     full reproduction, ~2 h 20 min of model compute
#   ./reproduce.sh --skip-venv         reuse the existing .venv
#   ./reproduce.sh --verify-uninstall  also do a real apply/remove round trip on a copy
#                                      of the 2 GB model and compare SHA-256
#   ./reproduce.sh --include-pending   also run the scripts whose result JSONs do not
#                                      exist yet (paper placeholders: gain, gate, stack)
#   ./reproduce.sh --help
#
# Hardware / scheduling: each model process peaks at 2.5-3 GB of unified memory. The
# steps below run one at a time. While `memory_pressure` reports normal, two may be
# overlapped (e.g. compile | readout, or flip | taskpatch); never three on 16 GB.
# Wall-clock estimates come from experiments.csv and the run logs on this machine; a
# cold page cache adds a few minutes to the first step after a reboot.
#
# Disk: the model directory is 2.0 GB. --verify-uninstall needs ~6.1 GB free
# (source + patched copy + restored copy) plus the 13 MB patch; the script checks
# free space and refuses to start that step if it is short.
#
# Every model command writes its full log to <name>.log; on failure the last 40 lines
# are printed. Logs are the only files written outside res_*.json / patches/*.npz.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DRY_RUN=0
SKIP_VENV=0
VERIFY_UNINSTALL=0
INCLUDE_PENDING=0
PY=".venv/bin/python"
MODEL="models/omni-2b-4bit"
FLIP_PATCH="patches/flippatch_L8-12-16-20-24_it800_n300.npz"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)           DRY_RUN=1 ;;
    --skip-venv)         SKIP_VENV=1 ;;
    --verify-uninstall)  VERIFY_UNINSTALL=1 ;;
    --include-pending)   INCLUDE_PENDING=1 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "reproduce.sh: unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

export PYTHONUNBUFFERED=1

step() { printf '\n# [%s] %s — %s\n' "$1" "$2" "$3"; }

run() {
  printf '  + %s\n' "$*"
  [ "$DRY_RUN" -eq 1 ] && return 0
  "$@"
}

run_logged() {
  local name="$1" log="$2"
  shift 2
  printf '  + PYTHONUNBUFFERED=1 %s > %s 2>&1\n' "$*" "$log"
  [ "$DRY_RUN" -eq 1 ] && return 0
  if ! "$@" > "$log" 2>&1; then
    echo "FAILED: $name (full log: $log)" >&2
    tail -n 40 "$log" >&2 || true
    exit 1
  fi
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

preflight() {
  [ -d "$MODEL" ] || { echo "missing model directory: $MODEL" >&2; exit 1; }
  [ -f "$FLIP_PATCH" ] || { echo "missing patch: $FLIP_PATCH" >&2; exit 1; }
  if [ "$SKIP_VENV" -eq 1 ]; then
    [ -x "$PY" ] || { echo "missing $PY (run without --skip-venv)" >&2; exit 1; }
  fi
  if [ "$VERIFY_UNINSTALL" -eq 1 ]; then
    command -v shasum >/dev/null 2>&1 || command -v sha256sum >/dev/null 2>&1 \
      || { echo "need shasum or sha256sum for the uninstall check" >&2; exit 1; }
  fi
}

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 0 ]; then
  preflight
  command -v uv >/dev/null 2>&1 || { echo "uv is not on PATH (https://docs.astral.sh/uv/)" >&2; exit 1; }
fi

step "0/13" "rebuild the venv with uv" "~2-4 min, ~1 GB of wheels"
if [ "$SKIP_VENV" -eq 0 ]; then
  run uv venv --python 3.12 --allow-existing .venv
  run uv pip install --python "$PY" -r requirements.txt
else
  printf '  (skipped: --skip-venv; using %s)\n' "$PY"
fi

# ---------------------------------------------------------------------------
# 1. Result JSONs, in dependency order. Estimates are sequential wall clock.
# ---------------------------------------------------------------------------
step "1/13" "verify_qbits.py" "~1 min -> res_qbits_check.json"
run_logged verify_qbits verify_qbits.log "$PY" experiments/verify_qbits.py

step "2/13" "compile.py (dense compiled + task vector + rank-16/64 baselines)" \
     "~45 min -> res_compile_zs_L8-12-16-20-24.json"
run_logged compile_zs compile_zs.log "$PY" experiments/compile.py zs 8,12,16,20,24

step "3/13" "flip.py (E1 ±1 integer patch, 5 layers, 800 sweeps)" \
     "~27 min, ~6.5 GB peak -> res_flip_L8-12-16-20-24_it800_n300.json"
# NB: E1 was produced by flip.py's zero-init solver. flip3.py is the later
# dense-init development copy and writes a different tag; it does not reproduce E1.
run_logged flip_e1 flip_full2.log "$PY" experiments/flip.py 8,12,16,20,24 800 300 500

step "4/13" "dense_ceiling.py (dense float ΔW ceiling on the same five w2)" \
     "~9 min -> res_dense_ceiling_L8-12-16-20-24.json"
run_logged dense_ceiling dense_ceiling2.log "$PY" experiments/dense_ceiling.py 8,12,16,20,24 300 300 500

step "5/13" "readout.py, 4 rows (no suppression)" \
     "~7 min -> res_readout_n300_dev_sup0.json"
run_logged readout_4row readout_dev.log \
  "$PY" experiments/readout.py --ntr 300 --nev 500 --eval dev --suppress 0 --tag n300_dev_sup0

step "6/13" "readout.py, 104 rows (1,000 train queries, suppression 100)" \
     "~8 min -> res_readout_n1000_dev_sup100.json"
run_logged readout_sup100 readout_sup100.log \
  "$PY" experiments/readout.py --ntr 1000 --nev 500 --eval dev --suppress 100 --tag n1000_dev_sup100

step "7/13" "gatedreadout.py v3 (trigger-gated AG patch)" \
     "~12 min -> res_gated_v3.json"
run_logged gated_v3 gated_v3.log \
  "$PY" experiments/gatedreadout.py --ntr 600 --nev 300 --eval dev --trigger-scale 5 --suppress 100 --tag v3

step "8/13" "gatedreadout.py v5 (orthogonalised; negative format result)" \
     "~12 min -> res_gated_v5orth.json"
run_logged gated_v5orth gated_v5.log \
  "$PY" experiments/gatedreadout.py --ntr 600 --nev 300 --eval dev --trigger-scale 5 --suppress 120 --orth --tag v5orth

step "9/13" "gatedreadout.py v6 (non-orth, stronger plain penalty)" \
     "~12 min -> res_gated_v6.json"
run_logged gated_v6 gated_v6.log \
  "$PY" experiments/gatedreadout.py --ntr 600 --nev 300 --eval dev --trigger-scale 5 --suppress 120 --tag v6

step "10/13" "taskpatch.py summary, plain patch (second behaviour)" \
     "~8 min -> res_taskpatch_summary_plain_n300_dev.json"
run_logged taskpatch_summary taskpatch_summary.log \
  "$PY" experiments/taskpatch.py summary --ntr 300 --nev 300 --eval dev --suppress 100

step "11/13" "taskpatch.py summary, trigger-gated at id 125101" \
     "~8 min -> res_taskpatch_summary_gated_n300_dev.json"
run_logged taskpatch_summary_gated taskpatch_summary_gated.log \
  "$PY" experiments/taskpatch.py summary --gated --trigger-id 125101 --ntr 300 --nev 300 --eval dev --suppress 100

# ---------------------------------------------------------------------------
# 12. Scripts whose result JSONs do not exist yet (paper placeholders).
#     Run only with --include-pending; listed here so the plan is complete.
# ---------------------------------------------------------------------------
step "12/13" "pending: gain patch / split gate / two-skill stack" \
     "~37 min total; only with --include-pending"
if [ "$INCLUDE_PENDING" -eq 1 ]; then
  run_logged gainpatch gainpatch.log \
    "$PY" experiments/gainpatch.py 8,12,16,20,24 --tag gain_l8-24
  run_logged gate2 gate2.log \
    "$PY" experiments/gate2.py --ntr 600 --nev 300 --eval dev
  run_logged stack_demo stack_demo.log \
    "$PY" experiments/stack_demo.py --ntr 300 --nsup 100 --tunev 200
else
  printf '  (skipped: %s\n' "--include-pending not set)"
  printf '  + %s experiments/gainpatch.py 8,12,16,20,24 --tag gain_l8-24  # -> res_gainpatch_gain_l8-24.json\n' "$PY"
  printf '  + %s experiments/gate2.py --ntr 600 --nev 300 --eval dev  # -> res_gate2_<TAG>.json\n' "$PY"
  printf '  + %s experiments/stack_demo.py --ntr 300 --nsup 100 --tunev 200  # -> res_stack_demo.json\n' "$PY"
fi

# ---------------------------------------------------------------------------
# 13. Paper numbers. Exits non-zero if any status=ready row in numbers_map.csv
#     does not match its JSON, so a stale paper cannot pass silently.
# ---------------------------------------------------------------------------
step "13/13" "check_numbers.py" "<1 min; PASS/FAIL per claim"
printf '  + %s check_numbers.py\n' "$PY"
if [ "$DRY_RUN" -eq 0 ]; then
  "$PY" check_numbers.py
fi

# ---------------------------------------------------------------------------
# Optional: exact-uninstall round trip on a 2 GB copy of the model.
#   copy -> apply ±1 patch -> remove patch -> SHA-256 of model.safetensors must equal
#   the original. remove_patch.py also verifies the install manifest and the original
#   hash itself; the explicit comparison here is a second, independent check.
# ---------------------------------------------------------------------------
if [ "$VERIFY_UNINSTALL" -eq 1 ]; then
  step "extra" "exact-uninstall round trip on a model copy" \
       "~3 min, ~6.1 GB disk, 2 × 2 GB copies"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  + TMP_ROOT="${TMPDIR:-/tmp}"; TMP_ROOT="${TMP_ROOT%%/}"\n'
    printf '  + TMP=$(mktemp -d "$TMP_ROOT/lfm3-uninstall.XXXXXX")\n'
    printf '  + %s patches/apply_patch.py %s %s --out "$TMP/patched"\n' "$PY" "$MODEL" "$FLIP_PATCH"
    printf '  + %s patches/remove_patch.py "$TMP/patched" %s --out "$TMP/restored"\n' "$PY" "$FLIP_PATCH"
    printf '  + compare SHA-256 of %s/model.safetensors and "$TMP/restored/model.safetensors"\n' "$MODEL"
    printf '  + rm -rf "$TMP"\n'
  else
    free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
    if [ "${free_kb:-0}" -lt 7340032 ]; then
      echo "need >= 7 GB free for the uninstall round trip; have $((free_kb / 1024)) MB" >&2
      exit 1
    fi
    TMP_ROOT="${TMPDIR:-/tmp}"
    TMP_ROOT="${TMP_ROOT%/}"
    TMP="$(mktemp -d "$TMP_ROOT/lfm3-uninstall.XXXXXX")"
    cleanup() {
      case "${TMP:-}" in
        "$TMP_ROOT"/lfm3-uninstall.*) [ -d "$TMP" ] && rm -rf -- "$TMP" ;;
      esac
    }
    trap cleanup EXIT
    run "$PY" patches/apply_patch.py "$MODEL" "$FLIP_PATCH" --out "$TMP/patched"
    run "$PY" patches/remove_patch.py "$TMP/patched" "$FLIP_PATCH" --out "$TMP/restored"
    orig="$(sha256_file "$MODEL/model.safetensors")"
    rest="$(sha256_file "$TMP/restored/model.safetensors")"
    printf '  original model.safetensors  %s\n' "$orig"
    printf '  restored model.safetensors  %s\n' "$rest"
    if [ "$orig" != "$rest" ]; then
      echo "UNINSTALL FAILED: SHA-256 differs" >&2
      exit 1
    fi
    echo "UNINSTALL OK: restored file is SHA-256-identical to the original"
  fi
fi

printf '\n# done\n'
