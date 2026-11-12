#!/usr/bin/env python3
"""Verify every number in paper/paper.md against the result JSONs that produced it.

The mapping lives in numbers_map.csv (one row per printed claim):

    claim_id, paper_string, json_file, json_path, tol, status, scale

  claim_id      short stable id; also used by --markdown and CHECK_NUMBERS_SPEC.md
  paper_string  the literal string that appears (or will appear) in the paper,
                e.g. "9.8 [7.4, 12.4]", "0.467%", "12.6 MB", "yes"
  json_file     result JSON, relative to this directory
  json_path     dotted path with optional [i] list indexing, e.g.
                "student_compiled.says_correct_label.ci", "fit[2].holdout_unexplained_true"
                A path ending in ".ci" means the node is the [mean, low, high] triple
                that a 95% bootstrap interval is printed from.
  tol           absolute tolerance in the printed unit (0 for exact counts)
  status        "ready" (checked) or "TODO" (not measured yet: printed, not checked)
  scale         multiply the JSON value before comparing (e.g. 1e-6 for bytes -> MB,
                100 for a fraction -> percent). Optional; defaults to 1.

Checks:
  * scalar claim: one number, compared to the JSON scalar (or to the mean of a triple)
  * interval claim: "mean [low, high]" is compared element-wise to the triple
  * boolean claim: the JSON bool must agree with yes/no (or true/false) in the string
  * .ci paths must resolve to a three-element numeric list

Exit code is 0 only when every status=="ready" row passes. TODO rows never fail.

Usage:
  .venv/bin/python check_numbers.py                # check
  .venv/bin/python check_numbers.py --update       # rewrite paper_string values from JSON
  .venv/bin/python check_numbers.py --markdown     # print a table for the paper
  .venv/bin/python check_numbers.py --no-color     # plain output (for logs)

Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MAP = ROOT / "results" / "numbers_map.csv"


def _find(name):
    """Freshly regenerated JSONs live in the repo root; the archived ones in results/."""
    p = ROOT / name
    return p if p.exists() else ROOT / "results" / name


# Accept ASCII '-' and the Unicode minus used in the paper.
NUM_RE = re.compile(r"[-−–]?\d[\d,]*(?:\.\d+)?")
PATH_TOKEN_RE = re.compile(r"[^.\[\]]+|\[\d+\]")
COLUMNS = ["claim_id", "paper_string", "json_file", "json_path", "tol", "status", "scale"]


def parse_numbers(text: str) -> list[float]:
    """Every number in a printed claim string, in order."""
    out = []
    for tok in NUM_RE.findall(text):
        tok = tok.replace("−", "-").replace("–", "-").replace(",", "")
        out.append(float(tok))
    return out


def load_json(path: Path):
    with path.open("rb") as fh:
        return json.load(fh)


def resolve(doc, path: str):
    """Resolve a dotted path with [i] indexing. A trailing '.ci' is stripped."""
    is_ci = path.endswith(".ci") or path == "ci"
    if path.endswith(".ci"):
        path = path[:-3]
    if path == "ci":
        path = ""
    node = doc
    for tok in PATH_TOKEN_RE.findall(path):
        if tok.startswith("["):
            node = node[int(tok[1:-1])]
        elif tok:
            node = node[tok]
    return node, is_ci


def decimals(tol: float) -> int:
    """Printed decimal places implied by an absolute tolerance."""
    if tol <= 0:
        return 0
    return max(0, min(6, int(round(-math.log10(tol)))))


def fmt(x: float, tol: float) -> str:
    d = decimals(tol)
    if d == 0 and float(x).is_integer() and abs(x) >= 1000:
        return f"{int(round(x)):,}"
    if d == 0:
        return f"{x:.0f}"
    return f"{x:.{d}f}"


def as_triple(value):
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
    ):
        return [float(v) for v in value]
    return None


def fmt_triple(t: list[float], tol: float) -> str:
    return f"{fmt(t[0], tol)} [{fmt(t[1], tol)}, {fmt(t[2], tol)}]"


def check_row(row: dict, docs: dict):
    """Return (ok, status, detail). status is PASS/FAIL/TODO/SKIP."""
    cid = row["claim_id"]
    if row["status"].strip().lower() != "ready":
        return True, "TODO", row["paper_string"].strip() or "TODO"

    jf = row["json_file"].strip()
    jpath = row["json_path"].strip()
    tol = float(row["tol"])
    scale = float(row.get("scale") or 1.0)
    claim = row["paper_string"].strip()

    try:
        if jf not in docs:
            docs[jf] = load_json(_find(jf))
        raw, is_ci = resolve(docs[jf], jpath)
    except FileNotFoundError:
        return False, "FAIL", f"missing JSON file {jf}"
    except (KeyError, IndexError, TypeError) as exc:
        return False, "FAIL", f"cannot resolve {jf}:{jpath} ({type(exc).__name__}: {exc})"
    except json.JSONDecodeError as exc:
        return False, "FAIL", f"{jf} is not valid JSON ({exc})"

    # Booleans.
    if isinstance(raw, bool):
        wants_yes = re.search(r"\b(yes|true)\b", claim, re.I) is not None
        wants_no = re.search(r"\b(no|false)\b", claim, re.I) is not None
        if not (wants_yes or wants_no):
            return False, "FAIL", "boolean JSON value but claim has no yes/no word"
        ok = wants_yes == raw and wants_no != raw
        want = "yes" if raw else "no"
        return ok, "PASS" if ok else "FAIL", f"expected {want}, claim says {claim!r}"

    triple = as_triple(raw)
    nums = parse_numbers(claim)

    # Interval claim.
    if is_ci or (triple is not None and len(nums) == 3):
        if triple is None:
            return False, "FAIL", f"{jf}:{jpath} is not a [mean, low, high] triple: {raw!r}"
        expected = [v * scale for v in triple]
        if len(nums) != 3:
            return False, "FAIL", f"claim has {len(nums)} numbers, need 3 (mean [low, high])"
        deltas = [abs(a - b) for a, b in zip(nums, expected)]
        ok = all(d <= tol + 1e-12 for d in deltas)
        detail = (f"mean {nums[0]:g} vs {expected[0]:g} (|d|={deltas[0]:.3g}); "
                  f"low {nums[1]:g} vs {expected[1]:g} (|d|={deltas[1]:.3g}); "
                  f"high {nums[2]:g} vs {expected[2]:g} (|d|={deltas[2]:.3g})")
        return ok, "PASS" if ok else "FAIL", detail

    # Scalar claim.
    if triple is not None:
        # A triple whose claim prints only the mean.
        value = triple[0] * scale
        if len(nums) != 1:
            return False, "FAIL", f"claim has {len(nums)} numbers, need 1 (mean only)"
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw) * scale
        if len(nums) != 1:
            return False, "FAIL", f"claim has {len(nums)} numbers, need 1"
    else:
        return False, "FAIL", f"{jf}:{jpath} is not numeric: {raw!r}"

    delta = abs(nums[0] - value)
    ok = delta <= tol + 1e-12
    detail = f"claim {nums[0]:g} vs JSON {value:g} (|d|={delta:.3g}, tol={tol:g})"
    return ok, "PASS" if ok else "FAIL", detail


def read_map(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS[:6] if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing required column(s): {', '.join(missing)}")
        rows = []
        for row in reader:
            if not row.get("claim_id"):
                continue
            row.setdefault("scale", "1")
            if not row.get("scale"):
                row["scale"] = "1"
            rows.append(row)
    return rows


def write_map(path: Path, rows: list[dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def do_update(path: Path, rows: list[dict], docs: dict) -> int:
    changed = 0
    for row in rows:
        if row["status"].strip().lower() != "ready":
            continue
        jf, jpath = row["json_file"].strip(), row["json_path"].strip()
        tol, scale = float(row["tol"]), float(row.get("scale") or 1.0)
        claim = row["paper_string"]
        try:
            if jf not in docs:
                docs[jf] = load_json(_find(jf))
            raw, is_ci = resolve(docs[jf], jpath)
        except Exception as exc:  # keep going; report as failure to update
            print(f"SKIP  {row['claim_id']:<28} {type(exc).__name__}: {exc}")
            continue

        if isinstance(raw, bool):
            want = "yes" if raw else "no"
            new = re.sub(r"\b(yes|no|true|false)\b", want, claim, count=1, flags=re.I)
        elif is_ci or as_triple(raw) is not None:
            triple = as_triple(raw)
            if triple is None:
                print(f"SKIP  {row['claim_id']:<28} not a [mean, low, high] triple")
                continue
            nums = list(NUM_RE.finditer(claim))
            if len(nums) != 3:
                print(f"SKIP  {row['claim_id']:<28} claim does not contain 3 numbers")
                continue
            parts, last = [], 0
            for m, rep in zip(nums, [fmt(v * scale, tol) for v in triple]):
                parts.append(claim[last:m.start()])
                parts.append(rep)
                last = m.end()
            parts.append(claim[last:])
            new = "".join(parts)
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            nums = list(NUM_RE.finditer(claim))
            if len(nums) != 1:
                print(f"SKIP  {row['claim_id']:<28} claim does not contain exactly 1 number")
                continue
            m = nums[0]
            new = claim[:m.start()] + fmt(float(raw) * scale, tol) + claim[m.end():]
        else:
            print(f"SKIP  {row['claim_id']:<28} JSON value is not numeric")
            continue

        if new != claim:
            row["paper_string"] = new
            changed += 1
            print(f"UPD   {row['claim_id']:<28} {claim!r} -> {new!r}")

    if changed:
        write_map(path, rows)
    print(f"updated {changed} row(s) in {path.name}")
    return 0


def do_markdown(rows: list[dict]) -> int:
    print("| claim | value | source | JSON path | status |")
    print("|---|---|---|---|---|")
    for row in rows:
        cid = row["claim_id"]
        val = row["paper_string"].replace("|", "\\|") or "TODO"
        src = row["json_file"].replace("|", "\\|")
        jp = row["json_path"].replace("|", "\\|") or "—"
        print(f"| `{cid}` | {val} | `{src}` | `{jp}` | {row['status']} |")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map", nargs="?", default=str(DEFAULT_MAP),
                    help="mapping CSV (default: numbers_map.csv next to this script)")
    ap.add_argument("--update", action="store_true",
                    help="rewrite paper_string values from the JSONs")
    ap.add_argument("--markdown", action="store_true", help="print a markdown table")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    args = ap.parse_args(argv)

    path = Path(args.map)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    rows = read_map(path)
    docs: dict = {}

    if args.update:
        return do_update(path, rows, docs)
    if args.markdown:
        return do_markdown(rows)

    color = sys.stdout.isatty() and not args.no_color
    green = "\033[32m" if color else ""
    red = "\033[31m" if color else ""
    dim = "\033[2m" if color else ""
    reset = "\033[0m" if color else ""

    n_pass = n_fail = n_todo = 0
    failures = []
    for row in rows:
        ok, status, detail = check_row(row, docs)
        cid = row["claim_id"]
        if status == "TODO":
            n_todo += 1
            print(f"{dim}TODO  {cid:<28} {detail}{reset}")
            continue
        if ok:
            n_pass += 1
            print(f"{green}PASS{reset}  {cid:<28} {row['paper_string']}")
        else:
            n_fail += 1
            failures.append(cid)
            print(f"{red}FAIL{reset}  {cid:<28} {row['paper_string']}")
            print(f"      {detail}")
            print(f"      source: {row['json_file']}:{row['json_path']}")

    print()
    print(f"ready {n_pass + n_fail}  pass {n_pass}  fail {n_fail}  todo {n_todo}")
    if failures:
        print("failed: " + ", ".join(failures))
    if n_pass + n_fail == 0:
        print("no ready rows to check", file=sys.stderr)
        return 2
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
