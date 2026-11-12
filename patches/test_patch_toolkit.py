#!/usr/bin/env python3
"""Self-contained test for apply_patch.py and remove_patch.py.

Builds a tiny synthetic MLX-layout safetensors file with one 4-bit and one
6-bit quantised matrix, applies a patch, checks every changed code, removes the
patch and requires a byte-identical file. No model or MLX import is involved.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ".flippatch.json"
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

# layer -> (bits, out, codes per row)
SPEC = {1: (4, 5, 24), 2: (6, 5, 21)}
EDITS = {
    1: [(0, +1), (1, -1), (19, +1), (119, -1)],
    2: [(0, +1), (1, -1), (7, +1), (104, -1)],
}
UNTOUCHED = ["language_model.model.embed_tokens.weight", "language_model.model.embedding_norm.weight"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(script: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PYTHON), str(ROOT / script), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=str(ROOT.parent),
    )


def run_ok(script: str, *args: object) -> subprocess.CompletedProcess:
    result = run(script, *args)
    if result.returncode != 0:
        raise AssertionError(
            f"{script} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def pack_words(codes: np.ndarray, bits: int, pad_fill: int = 0) -> tuple[int, bytes]:
    """Write MLX's little-endian bit stream, optionally filling padding bits."""
    out_dim, code_count = codes.shape
    total_bits = code_count * bits
    nw = (total_bits + 31) // 32
    stream = np.full((out_dim, nw * 32), pad_fill, dtype=np.uint8)
    for bit in range(bits):
        stream[:, bit : total_bits : bits] = ((codes >> bit) & 1).astype(np.uint8)
    raw = np.packbits(stream.reshape(out_dim, nw, 4, 8)[..., ::-1], axis=-1)
    return nw, raw.reshape(out_dim, nw * 4).tobytes()


def bit_stream(raw: bytes, out_dim: int, nw: int) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u4").reshape(out_dim, nw)
    bits = np.unpackbits(words.view(np.uint8).reshape(out_dim, nw, 4), axis=-1)
    return bits.reshape(out_dim, nw, 4, 8)[..., ::-1].reshape(out_dim, nw * 32)


def unpack_words(raw: bytes, out_dim: int, nw: int, bits: int) -> np.ndarray:
    stream = bit_stream(raw, out_dim, nw)
    code_count = (nw * 32) // bits
    codes = np.zeros((out_dim, code_count), dtype=np.int64)
    for bit in range(bits):
        codes |= stream[:, bit : code_count * bits : bits].astype(np.int64) << bit
    return codes


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]], metadata: dict
) -> None:
    header: dict = {"__metadata__": metadata}
    offset = 0
    ordered = sorted(tensors)
    for name in ordered:
        dtype, shape, raw = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("ascii")
    header_json += b" " * ((-len(header_json)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_json)))
        handle.write(header_json)
        for name in ordered:
            handle.write(tensors[name][2])


def read_safetensors(path: Path) -> tuple[dict, bytes, bytes]:
    blob = path.read_bytes()
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header_raw = blob[8 : 8 + header_len]
    return json.loads(header_raw), header_raw, blob[8 + header_len :]


def tensor_bytes(header: dict, data: bytes, name: str) -> bytes:
    start, end = header[name]["data_offsets"]
    return data[start:end]


def weight_key(layer: int) -> str:
    return f"language_model.model.layers.{layer - 1}.feed_forward.w2.weight"


def make_model(root: Path) -> tuple[Path, dict[int, np.ndarray], dict[str, int]]:
    model = root / "model"
    model.mkdir()
    rng = np.random.default_rng(20261112)
    codes = {
        layer: rng.integers(0, 1 << bits, size=(out_dim, count), dtype=np.int64)
        for layer, (bits, out_dim, count) in SPEC.items()
    }
    # Boundary and interior values used by the patch and overflow tests.
    codes[1].flat[[0, 1, 19, 119]] = [0, 15, 3, 7]
    codes[2].flat[[0, 1, 7, 104]] = [0, 63, 5, 2]

    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    word_widths: dict[str, int] = {}
    for layer, (bits, out_dim, _count) in SPEC.items():
        nw, raw = pack_words(codes[layer], bits, pad_fill=1)
        key = weight_key(layer)
        tensors[key] = ("U32", (out_dim, nw), raw)
        word_widths[key] = nw
    tensors["language_model.model.embed_tokens.weight"] = (
        "BF16",
        (3, 2),
        rng.integers(0, 256, size=12, dtype=np.uint8).tobytes(),
    )
    tensors["language_model.model.embedding_norm.weight"] = (
        "F16",
        (4,),
        rng.integers(0, 256, size=8, dtype=np.uint8).tobytes(),
    )
    write_safetensors(
        model / "model.safetensors", tensors, {"format": "mlx", "test": "patch-toolkit"}
    )

    config = {
        "model_type": "synthetic",
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "language_model.model.layers.1.feed_forward.w2": {
                "group_size": 64,
                "bits": 6,
                "mode": "affine",
            },
        },
    }
    (model / "config.json").write_text(json.dumps(config, indent=2))
    (model / "tokenizer.json").write_text('{"synthetic": true}')
    (model / "tokenizer_config.json").write_text('{"model_max_length": 32}')
    index = {
        "metadata": {"total_size": (model / "model.safetensors").stat().st_size},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return model, codes, word_widths


def write_patch(
    path: Path,
    codes: dict[int, np.ndarray],
    edits: dict[int, list[tuple[int, int]]],
    bits_as_scalar: bool = False,
    extra: dict[int, tuple[int, int]] | None = None,
) -> None:
    arrays = {}
    for layer, items in edits.items():
        items = list(items)
        if extra and layer in extra:
            items.append(extra[layer])
        idx = np.array([flat for flat, _sign in items], dtype=np.uint32)
        sign = np.array([sign for _flat, sign in items], dtype=np.int8)
        bits, _out_dim, _count = SPEC[layer]
        arrays[f"idx_{layer}"] = idx
        arrays[f"sign_{layer}"] = sign
        arrays[f"shape_{layer}"] = np.array(codes[layer].shape, dtype=np.int32)
        if bits_as_scalar:
            arrays[f"bits_{layer}"] = np.array(bits, dtype=np.int32)
        else:
            arrays[f"bits_{layer}"] = np.array([bits], dtype=np.int32)
    np.savez_compressed(path, **arrays)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="flippatch-toolkit-") as tmp:
        root = Path(tmp)
        model, codes, word_widths = make_model(root)
        original_path = model / "model.safetensors"
        original_blob = original_path.read_bytes()
        original_sha = sha256_bytes(original_blob)
        original_header, original_header_raw, original_data = read_safetensors(original_path)

        patch = root / "patch.npz"
        write_patch(patch, codes, EDITS)
        print(f"synthetic model: {original_path} ({len(original_blob)} bytes, sha {original_sha[:12]}...)")

        # ---- apply -----------------------------------------------------
        patched = root / "patched"
        result = run_ok("apply_patch.py", model, patch, "--out", patched)
        check(original_sha in result.stdout, "apply did not print the original SHA-256")
        patched_blob = (patched / "model.safetensors").read_bytes()
        patched_sha = sha256_bytes(patched_blob)
        check(patched_sha in result.stdout, "apply did not print the patched SHA-256")
        check("codes changed: 8" in result.stdout, "apply reported the wrong changed-code count")
        check((patched / "config.json").read_bytes() == (model / "config.json").read_bytes(),
              "config.json was not copied byte-for-byte")
        check((patched / "tokenizer.json").read_bytes() == (model / "tokenizer.json").read_bytes(),
              "tokenizer.json was not copied byte-for-byte")
        check((patched / MANIFEST).is_file(), "apply did not write the install manifest")

        manifest = json.loads((patched / MANIFEST).read_text())
        entry = manifest["files"]["model.safetensors"]
        check(entry["original_sha256"] == original_sha, "manifest original SHA-256 is wrong")
        check(entry["patched_sha256"] == patched_sha, "manifest patched SHA-256 is wrong")
        check(entry["codes_changed"] == 8, "manifest changed-code count is wrong")

        patched_header, patched_header_raw, patched_data = read_safetensors(
            patched / "model.safetensors"
        )
        check(patched_header_raw == original_header_raw,
              "safetensors header changed (names/metadata/offsets must be preserved)")
        check(patched_header == original_header, "safetensors header JSON changed")
        check(len(patched_blob) == len(original_blob), "patched file size changed")
        check(original_path.read_bytes() == original_blob, "source model was modified")
        for name in UNTOUCHED:
            check(tensor_bytes(patched_header, patched_data, name)
                  == tensor_bytes(original_header, original_data, name),
                  f"untouched tensor {name} changed")

        patched_again = root / "patched_again"
        run_ok("apply_patch.py", model, patch, "--out", patched_again)
        check((patched_again / "model.safetensors").read_bytes() == patched_blob,
              "applying the same patch twice did not produce an identical file")

        for layer, (bits, out_dim, count) in SPEC.items():
            key = weight_key(layer)
            nw = word_widths[key]
            patched_codes = unpack_words(
                tensor_bytes(patched_header, patched_data, key), out_dim, nw, bits
            )
            expected = codes[layer].copy()
            for flat, sign in EDITS[layer]:
                expected.flat[flat] += sign
            check(np.array_equal(patched_codes, expected),
                  f"layer {layer}: patched codes are not exactly the requested edits")
            changed = np.count_nonzero(patched_codes.ravel() != codes[layer].ravel())
            check(changed == len(EDITS[layer]), f"layer {layer}: changed {changed} codes, expected {len(EDITS[layer])}")

            if count * bits < nw * 32:
                old_padding = bit_stream(tensor_bytes(original_header, original_data, key), out_dim, nw)[:, count * bits :]
                new_padding = bit_stream(tensor_bytes(patched_header, patched_data, key), out_dim, nw)[:, count * bits :]
                check(np.array_equal(new_padding, old_padding),
                      f"layer {layer}: padding bits in the last word were not preserved")

        # Refuse an existing output directory.
        collision = run("apply_patch.py", model, patch, "--out", patched)
        check(collision.returncode != 0 and "already exists" in collision.stderr,
              "apply did not refuse an existing output directory")

        # ---- remove ----------------------------------------------------
        restored = root / "restored"
        result = run_ok("remove_patch.py", patched, patch, "--out", restored)
        restored_blob = (restored / "model.safetensors").read_bytes()
        restored_sha = sha256_bytes(restored_blob)
        check(original_sha in result.stdout, "remove did not print the original SHA-256")
        check(restored_sha in result.stdout, "remove did not print the restored SHA-256")
        check(restored_blob == original_blob, "restored safetensors is not byte-identical")
        check(not (restored / MANIFEST).exists(), "restored directory still contains the manifest")

        restored_expect = root / "restored_expect"
        result = run_ok(
            "remove_patch.py", patched, patch, "--out", restored_expect,
            "--expect-sha256", original_sha,
        )
        check((restored_expect / "model.safetensors").read_bytes() == original_blob,
              "remove --expect-sha256 did not restore a byte-identical file")

        bad_restore = root / "bad_restore"
        result = run(
            "remove_patch.py", patched, patch, "--out", bad_restore,
            "--expect-sha256", "0" * 64,
        )
        check(result.returncode != 0, "remove accepted a wrong --expect-sha256")
        check(not bad_restore.exists(), "remove created output despite a hash mismatch")

        # ---- rejected patches ------------------------------------------
        out_of_range = root / "bad_range.npz"
        write_patch(out_of_range, codes, EDITS, extra={1: (5 * 24, +1)})
        bad_out = root / "bad_range_out"
        result = run("apply_patch.py", model, out_of_range, "--out", bad_out)
        check(result.returncode != 0, "apply accepted an out-of-range index")
        check("out of range" in (result.stdout + result.stderr).lower(),
              "out-of-range error message is missing")
        check(not bad_out.exists(), "apply created output for an out-of-range patch")

        for name, flat, sign in [("negative", 0, -1), ("positive", 1, +1)]:
            overflow = root / f"overflow_{name}.npz"
            write_patch(overflow, codes, {1: [(flat, sign)]})
            bad_out = root / f"overflow_{name}_out"
            result = run("apply_patch.py", model, overflow, "--out", bad_out)
            check(result.returncode != 0, f"apply accepted {name} code overflow")
            check("overflow" in (result.stdout + result.stderr).lower(),
                  f"{name} overflow error message is missing")
            check(not bad_out.exists(), f"apply created output for {name} overflow")

        # Existing deliverable patches store bits_L as a 0-d scalar; accept that too.
        scalar_patch = root / "patch_scalar_bits.npz"
        write_patch(scalar_patch, codes, EDITS, bits_as_scalar=True)
        scalar_out = root / "patched_scalar_bits"
        result = run_ok("apply_patch.py", model, scalar_patch, "--out", scalar_out)
        check("codes changed: 8" in result.stdout, "scalar bits_L patch was not applied")

        # Stacked patches: remove the second, then peel the first with --expect-sha256.
        patch_a = root / "stack_a.npz"
        patch_b = root / "stack_b.npz"
        write_patch(patch_a, codes, {1: EDITS[1]})
        write_patch(patch_b, codes, {2: EDITS[2]})
        stack_a = root / "stack_a"
        stack_ab = root / "stack_ab"
        stack_removed = root / "stack_removed"
        peeled = root / "peeled"
        run_ok("apply_patch.py", model, patch_a, "--out", stack_a)
        run_ok("apply_patch.py", stack_a, patch_b, "--out", stack_ab)
        run_ok("remove_patch.py", stack_ab, patch_b, "--out", stack_removed)
        check((stack_removed / "model.safetensors").read_bytes()
              == (stack_a / "model.safetensors").read_bytes(),
              "removing the second stacked patch did not leave the first intact")
        run_ok("remove_patch.py", stack_removed, patch_a, "--out", peeled,
               "--expect-sha256", original_sha)
        check((peeled / "model.safetensors").read_bytes() == original_blob,
              "peeling the first stacked patch did not restore the original")

        print("PASS: apply/remove exactness, bounds, overflow, header and metadata preservation")


if __name__ == "__main__":
    main()
