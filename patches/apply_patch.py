#!/usr/bin/env python3
"""Install a flip patch into a copy of an MLX safetensors model.

The patch format and the MLX bit-stream packing are documented in README.md.
This module deliberately does not import MLX: it reads and writes safetensors
directly and manipulates the packed U32 words with numpy.

`remove_patch.py` imports the shared readers/planners from here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TARGET = "language_model.model.layers.{}.feed_forward.w2.weight"
MANIFEST = ".flippatch.json"
MANIFEST_FORMAT = "flippatch-apply-v1"
_KEY_RE = re.compile(r"^(idx|sign|shape|bits)_(\d+)$")
_HASH_CHUNK = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 100_000_000


class PatchError(Exception):
    """A patch or model failed validation. The CLI reports this without a traceback."""


@dataclass(frozen=True)
class LayerPatch:
    layer: int
    idx: np.ndarray
    sign: np.ndarray
    shape: tuple[int, int]
    bits: int


@dataclass
class Edit:
    layer: int
    tensor_offset: int
    packed: bytes
    codes_changed: int


@dataclass
class FilePlan:
    path: Path
    rel: str
    source_size: int
    source_sha256: str = ""
    edits: list[Edit] = field(default_factory=list)
    output_sha256: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def check_output_dir(model_dir: Path, out_dir: Path) -> None:
    if not model_dir.is_dir():
        raise PatchError(f"model directory does not exist: {model_dir}")
    if out_dir.exists():
        raise PatchError(f"output directory already exists: {out_dir}")
    if _is_inside(out_dir, model_dir):
        raise PatchError(f"output directory must not be inside the model directory: {out_dir}")


def load_patch(path: Path) -> list[LayerPatch]:
    """Read and validate the per-layer arrays of a patch .npz."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except Exception as exc:
        raise PatchError(f"could not read patch {path}: {exc}") from exc

    groups: dict[int, dict[str, np.ndarray]] = {}
    for name, array in arrays.items():
        match = _KEY_RE.fullmatch(name)
        if match is None:
            raise PatchError(f"unexpected key {name!r} in patch")
        kind, layer_text = match.groups()
        layer = int(layer_text)
        if layer < 1:
            raise PatchError(f"layer numbers are 1-based, got {layer}")
        groups.setdefault(layer, {})[kind] = array

    if not groups:
        raise PatchError("patch contains no layer arrays")

    patches: list[LayerPatch] = []
    for layer in sorted(groups):
        group = groups[layer]
        missing = sorted({"idx", "sign", "shape", "bits"} - set(group))
        if missing:
            raise PatchError(f"layer {layer} is missing {', '.join(missing)}")

        idx = group["idx"]
        sign = group["sign"]
        shape = group["shape"]
        bits_array = group["bits"]

        if idx.ndim != 1 or idx.dtype != np.uint32:
            raise PatchError(f"idx_{layer} must be a 1-D uint32 array")
        if sign.ndim != 1 or sign.dtype != np.int8:
            raise PatchError(f"sign_{layer} must be a 1-D int8 array")
        if sign.shape != idx.shape:
            raise PatchError(f"idx_{layer} and sign_{layer} have different lengths")
        if not np.all((sign == 1) | (sign == -1)):
            raise PatchError(f"sign_{layer} entries must be +1 or -1")
        if shape.dtype != np.int32 or shape.shape != (2,):
            raise PatchError(f"shape_{layer} must be an int32 array of length 2")
        if bits_array.dtype != np.int32 or bits_array.size != 1:
            raise PatchError(f"bits_{layer} must be an int32 scalar or length-1 array")

        out_dim, in_dim = (int(v) for v in shape)
        if out_dim <= 0 or in_dim <= 0:
            raise PatchError(f"shape_{layer} must be positive, got {(out_dim, in_dim)}")
        if idx.size:
            if int(idx.max()) >= out_dim * in_dim:
                raise PatchError(
                    f"layer {layer}: flat index {int(idx.max())} is out of range "
                    f"for a {out_dim}x{in_dim} matrix"
                )
            if np.unique(idx).size != idx.size:
                raise PatchError(f"layer {layer}: duplicate flat indices are not allowed")

        bits = int(bits_array.reshape(-1)[0])
        if bits not in (4, 6):
            raise PatchError(f"layer {layer}: bits must be 4 or 6, got {bits}")

        patches.append(
            LayerPatch(
                layer=layer,
                idx=idx,
                sign=sign,
                shape=(out_dim, in_dim),
                bits=bits,
            )
        )
    return patches


def _read_header(path: Path) -> tuple[int, dict, int]:
    """Return (header byte length, header dict, file size)."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise PatchError(f"{path}: file is too short for a safetensors header")
            header_len = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_len <= _MAX_HEADER_BYTES:
                raise PatchError(f"{path}: invalid safetensors header length {header_len}")
            raw = handle.read(header_len)
            if len(raw) != header_len:
                raise PatchError(f"{path}: truncated safetensors header")
            file_size = path.stat().st_size
    except OSError as exc:
        raise PatchError(f"could not read {path}: {exc}") from exc

    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise PatchError(f"{path}: safetensors header is not a JSON object")
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        raise PatchError(f"{path}: __metadata__ is not a JSON object")
    return header_len, header, file_size


def _load_quant_bits(model_dir: Path) -> tuple[int | None, dict[str, int]]:
    """Read the global and per-module bit widths declared by config.json, if any."""
    path = model_dir / "config.json"
    if not path.is_file():
        return None, {}
    try:
        config = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, {}
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, dict):
        return None, {}

    def as_bits(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    global_bits = as_bits(quant.get("bits"))
    overrides: dict[str, int] = {}
    for name, value in quant.items():
        if isinstance(value, dict) and "bits" in value:
            bits = as_bits(value["bits"])
            if bits is not None:
                overrides[name] = bits
    return global_bits, overrides


def _declared_bits(
    global_bits: int | None, overrides: dict[str, int], module: str, key: str
) -> int | None:
    for name in (module, module + ".weight", key):
        if name in overrides:
            return overrides[name]
    return global_bits


def _resolve_tensor_files(
    model_dir: Path, layers: list[LayerPatch]
) -> list[tuple[LayerPatch, str, Path, str, int, dict, int]]:
    """Locate the safetensors file and header entry for every edited layer."""
    weight_map: dict[str, str] = {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PatchError(f"could not read {index_path}: {exc}") from exc
        raw_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(raw_map, dict):
            raise PatchError(f"{index_path}: missing or invalid weight_map")
        weight_map = {str(k): str(v) for k, v in raw_map.items()}

    header_cache: dict[Path, tuple[int, dict, int]] = {}
    resolved = []
    for layer_patch in layers:
        key = TARGET.format(layer_patch.layer - 1)
        rel = weight_map.get(key)
        if rel is not None:
            path = (model_dir / rel).resolve()
            if not _is_inside(path, model_dir):
                raise PatchError(f"{index_path}: {rel!r} escapes the model directory")
            if not path.is_file():
                raise PatchError(f"{index_path} maps {key} to missing file {rel!r}")
            if path not in header_cache:
                header_cache[path] = _read_header(path)
            header_len, header, file_size = header_cache[path]
            if key not in header:
                raise PatchError(f"{rel}: tensor {key} is not in the safetensors header")
            resolved.append(
                (layer_patch, key, path, path.relative_to(model_dir.resolve()).as_posix(),
                 header_len, header, file_size)
            )
            continue

        hit = None
        for candidate in sorted(model_dir.rglob("*.safetensors")):
            path = candidate.resolve()
            if path not in header_cache:
                header_cache[path] = _read_header(path)
            if key in header_cache[path][1]:
                hit = (path, header_cache[path])
                break
        if hit is None:
            raise PatchError(f"could not find tensor {key} in any safetensors file")
        path, (header_len, header, file_size) = hit
        resolved.append(
            (layer_patch, key, path, path.relative_to(model_dir).as_posix(),
             header_len, header, file_size)
        )
    return resolved


def unpack_codes(data: bytes, out_dim: int, nw: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack MLX's little-endian bit stream, preserving all bits in `stream`."""
    words = np.frombuffer(data, dtype="<u4")
    if words.size != out_dim * nw:
        raise PatchError("packed tensor byte length does not match its header shape")
    words = words.reshape(out_dim, nw)
    raw_bits = np.unpackbits(words.view(np.uint8).reshape(out_dim, nw, 4), axis=-1)
    stream = raw_bits.reshape(out_dim, nw, 4, 8)[..., ::-1].reshape(out_dim, nw * 32)

    code_count = (nw * 32) // bits
    codes = np.zeros((out_dim, code_count), dtype=np.int64)
    for bit in range(bits):
        codes |= stream[:, bit : code_count * bits : bits].astype(np.int64) << bit
    return codes, stream


def pack_codes(codes: np.ndarray, stream: np.ndarray, bits: int) -> bytes:
    """Pack codes back into the bit stream. Bits beyond the codes are untouched."""
    out_dim, code_count = codes.shape
    nw = stream.shape[1] // 32
    for bit in range(bits):
        stream[:, bit : code_count * bits : bits] = ((codes >> bit) & 1).astype(np.uint8)
    byte_view = stream.reshape(out_dim, nw, 4, 8)[..., ::-1]
    return np.packbits(byte_view, axis=-1).reshape(out_dim, nw * 4).tobytes()


def prepare_plan(model_dir: Path, layers: list[LayerPatch], invert: bool = False) -> list[FilePlan]:
    """Validate every edit and produce the replacement bytes. Does not write anything."""
    model_dir = model_dir.resolve()
    global_bits, overrides = _load_quant_bits(model_dir)
    plans: dict[str, FilePlan] = {}

    for layer_patch, key, path, rel, header_len, header, file_size in _resolve_tensor_files(
        model_dir, layers
    ):
        layer = layer_patch.layer
        module = key[: -len(".weight")]
        entry = header[key]
        if not isinstance(entry, dict):
            raise PatchError(f"{rel}: invalid header entry for {key}")
        if entry.get("dtype") != "U32":
            raise PatchError(f"{rel}: {key} has dtype {entry.get('dtype')!r}, expected U32")
        try:
            out_dim, nw = (int(v) for v in entry["shape"])
            offset_start, offset_end = (int(v) for v in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PatchError(f"{rel}: malformed header entry for {key}") from exc

        declared = _declared_bits(global_bits, overrides, module, key)
        if declared is not None and declared != layer_patch.bits:
            raise PatchError(
                f"layer {layer}: patch says {layer_patch.bits}-bit but config.json "
                f"declares {declared}-bit for {module}"
            )

        code_count = (nw * 32) // layer_patch.bits
        if (out_dim, code_count) != layer_patch.shape:
            raise PatchError(
                f"layer {layer}: patch shape {layer_patch.shape} does not match "
                f"{key} shape {(out_dim, code_count)} at {layer_patch.bits} bits"
            )
        if layer_patch.idx.size and int(layer_patch.idx.max()) >= out_dim * code_count:
            raise PatchError(
                f"layer {layer}: flat index {int(layer_patch.idx.max())} is out of range "
                f"for a {out_dim}x{code_count} matrix"
            )

        data_start = 8 + header_len
        if not 0 <= offset_start <= offset_end:
            raise PatchError(f"{rel}: invalid data offsets for {key}")
        if data_start + offset_end > file_size:
            raise PatchError(f"{rel}: tensor {key} extends past the end of the file")
        byte_length = offset_end - offset_start
        if byte_length != out_dim * nw * 4:
            raise PatchError(f"{rel}: tensor {key} byte length does not match its shape")

        try:
            with open(path, "rb") as handle:
                handle.seek(data_start + offset_start)
                data = handle.read(byte_length)
        except OSError as exc:
            raise PatchError(f"could not read {key} from {rel}: {exc}") from exc
        if len(data) != byte_length:
            raise PatchError(f"{rel}: truncated tensor {key}")

        codes, stream = unpack_codes(data, out_dim, nw, layer_patch.bits)
        if pack_codes(codes, stream, layer_patch.bits) != data:
            raise PatchError(
                f"layer {layer}: unpack/pack round trip does not reproduce {key}; "
                "the patch bit width or the model layout is wrong"
            )

        flat = codes.reshape(-1)
        old = flat[layer_patch.idx].astype(np.int64)
        direction = -1 if invert else 1
        new = old + layer_patch.sign.astype(np.int64) * direction
        limit = (1 << layer_patch.bits) - 1
        bad = (new < 0) | (new > limit)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            raise PatchError(
                f"layer {layer}: code overflow at flat index "
                f"{int(layer_patch.idx[first])}: {int(old[first])} "
                f"{'+' if direction * int(layer_patch.sign[first]) > 0 else '-'} 1 "
                f"is outside 0..{limit}"
            )
        flat[layer_patch.idx] = new
        packed = pack_codes(codes, stream, layer_patch.bits)
        if len(packed) != byte_length:
            raise PatchError(f"internal error: repacked {key} has the wrong length")

        file_plan = plans.get(rel)
        if file_plan is None:
            file_plan = FilePlan(path=path, rel=rel, source_size=file_size)
            plans[rel] = file_plan
        file_plan.edits.append(
            Edit(
                layer=layer,
                tensor_offset=data_start + offset_start,
                packed=packed,
                codes_changed=int(layer_patch.idx.size),
            )
        )

    for file_plan in plans.values():
        file_plan.source_sha256 = sha256_file(file_plan.path)
    return sorted(plans.values(), key=lambda plan: plan.rel)


def materialize(model_dir: Path, out_dir: Path, plans: list[FilePlan]) -> None:
    """Copy the model directory and replace the planned tensor bytes in the copy."""
    if out_dir.exists():
        raise PatchError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(model_dir, out_dir)
    except Exception:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        raise

    try:
        for file_plan in plans:
            target = out_dir / file_plan.rel
            with open(target, "r+b") as handle:
                for edit in file_plan.edits:
                    handle.seek(edit.tensor_offset)
                    handle.write(edit.packed)
            if target.stat().st_size != file_plan.source_size:
                raise PatchError(f"{file_plan.rel}: file size changed while patching")
            file_plan.output_sha256 = sha256_file(target)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def make_manifest(patch_path: Path, plans: list[FilePlan]) -> dict:
    files = {}
    for file_plan in plans:
        files[file_plan.rel] = {
            "original_sha256": file_plan.source_sha256,
            "patched_sha256": file_plan.output_sha256,
            "codes_changed": sum(edit.codes_changed for edit in file_plan.edits),
            "layers": sorted(edit.layer for edit in file_plan.edits),
        }
    return {
        "format": MANIFEST_FORMAT,
        "patch_sha256": sha256_file(patch_path),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a flip patch to a copy of an MLX safetensors model."
    )
    parser.add_argument("model_dir", type=Path, help="source model directory (not modified)")
    parser.add_argument("patch", type=Path, help="patch .npz")
    parser.add_argument("--out", required=True, type=Path, help="new output model directory")
    args = parser.parse_args(argv)

    try:
        model_dir = args.model_dir.resolve()
        out_dir = args.out.resolve()
        if not args.patch.is_file():
            raise PatchError(f"patch does not exist: {args.patch}")
        check_output_dir(model_dir, out_dir)
        layers = load_patch(args.patch)
        plans = prepare_plan(model_dir, layers)
        materialize(model_dir, out_dir, plans)

        manifest = make_manifest(args.patch, plans)
        (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        total = sum(edit.codes_changed for plan in plans for edit in plan.edits)
        for file_plan in plans:
            print(f"{file_plan.rel}: original sha256 {file_plan.source_sha256}")
            print(f"{file_plan.rel}: new      sha256 {file_plan.output_sha256}")
        print(f"codes changed: {total}")
        print(f"layers: {', '.join(str(layer.layer) for layer in layers)}")
        print(f"output: {out_dir}")
        return 0
    except (PatchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
