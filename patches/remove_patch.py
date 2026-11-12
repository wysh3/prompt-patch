#!/usr/bin/env python3
"""Remove a flip patch from a patched model copy and verify the original SHA-256.

The heavy lifting (patch parsing, safetensors layout, MLX bit packing) is shared
with apply_patch.py. This file only supplies the inverse signs and the check
against the original file hash recorded at install time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from apply_patch import (
    MANIFEST,
    MANIFEST_FORMAT,
    PatchError,
    check_output_dir,
    load_patch,
    materialize,
    prepare_plan,
    sha256_file,
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_arg(text: str) -> str:
    value = text.strip().lower()
    if _SHA_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a 64-character lowercase hex SHA-256")
    return value


def _read_manifest(directory: Path) -> dict | None:
    path = directory / MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchError(f"could not read {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise PatchError(f"{path}: unsupported or malformed manifest")
    if not isinstance(manifest.get("files"), dict):
        raise PatchError(f"{path}: manifest has no files map")
    return manifest


def _manifest_sha(entry: dict, field: str, rel: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise PatchError(f"manifest entry for {rel} has no valid {field}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove a flip patch from a patched model copy and verify the original."
    )
    parser.add_argument("patched_dir", type=Path, help="patched model directory")
    parser.add_argument("patch", type=Path, help="the patch that was installed")
    parser.add_argument("--out", required=True, type=Path, help="new restored model directory")
    parser.add_argument(
        "--expect-sha256",
        type=_sha256_arg,
        help="expected original safetensors SHA-256 if the install manifest is absent",
    )
    args = parser.parse_args(argv)

    try:
        patched_dir = args.patched_dir.resolve()
        out_dir = args.out.resolve()
        if not args.patch.is_file():
            raise PatchError(f"patch does not exist: {args.patch}")
        check_output_dir(patched_dir, out_dir)
        layers = load_patch(args.patch)
        manifest = _read_manifest(patched_dir)
        if manifest is None and args.expect_sha256 is None:
            raise PatchError(
                "no install manifest found; pass --expect-sha256 with the original file hash"
            )

        patch_sha = sha256_file(args.patch)
        if manifest is not None:
            recorded_patch_sha = manifest.get("patch_sha256")
            if recorded_patch_sha != patch_sha:
                raise PatchError(
                    "patch SHA-256 does not match the one recorded by apply_patch.py"
                )

        plans = prepare_plan(patched_dir, layers, invert=True)
        expected: dict[str, str] = {}

        if manifest is not None:
            files = manifest["files"]
            for plan in plans:
                entry = files.get(plan.rel)
                if not isinstance(entry, dict):
                    raise PatchError(f"manifest has no entry for {plan.rel}")
                recorded_patched = _manifest_sha(entry, "patched_sha256", plan.rel)
                if plan.source_sha256 != recorded_patched:
                    raise PatchError(
                        f"{plan.rel}: patched file SHA-256 does not match the manifest; "
                        "is this the directory the patch was applied to?"
                    )
                expected[plan.rel] = _manifest_sha(entry, "original_sha256", plan.rel)

        if args.expect_sha256 is not None:
            if len(plans) != 1:
                raise PatchError(
                    "--expect-sha256 is only supported when the patch edits one safetensors file"
                )
            rel = plans[0].rel
            if rel in expected and expected[rel] != args.expect_sha256:
                raise PatchError(
                    "--expect-sha256 disagrees with the original hash in the apply manifest"
                )
            expected[rel] = args.expect_sha256

        materialize(patched_dir, out_dir, plans)

        all_match = True
        for plan in plans:
            want = expected[plan.rel]
            got = plan.output_sha256
            print(f"{plan.rel}: original sha256 {want}")
            print(f"{plan.rel}: restored sha256 {got}")
            if got != want:
                all_match = False
        total = sum(edit.codes_changed for plan in plans for edit in plan.edits)
        print(f"codes restored: {total}")
        if not all_match:
            raise PatchError("restored SHA-256 does not match the original model")

        manifest_out = out_dir / MANIFEST
        if manifest_out.exists():
            manifest_out.unlink()
        print("verified: yes")
        print(f"output: {out_dir}")
        return 0
    except (PatchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
