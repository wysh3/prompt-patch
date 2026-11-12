# Flip patches

A flip patch installs a prompt or behaviour into an MLX affine-quantised model by
moving a sparse set of integer weight codes by one step. It adds no parameters and
changes no scales or biases. It is applied to a copy of the model and can be
removed exactly.

This directory contains the final `.npz` patches plus the tools that install and
uninstall them.

## Patch file format

A patch is a zip archive written with `np.savez_compressed`. For every edited
layer `L` (1-based) it has four arrays:

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `idx_L` | `uint32` | `(K,)` | flat row-major index into the code matrix; `index = row * in + col` |
| `sign_L` | `int8` | `(K,)` | `+1` or `-1`; the code at that index is incremented by this amount |
| `shape_L` | `int32` | `(2,)` | `(out, in)` shape of the code matrix |
| `bits_L` | `int32` | `(1,)` | MLX bit width, `4` or `6` |

New writers should use shape `(1,)` for `bits_L`. The reader also accepts a
0-dimensional `int32` scalar because the earlier `flip.py`/`flip3.py` runs saved
it that way (`np.array(bits, np.int32)`).

Layer `L` edits this tensor in the safetensors file:

```
language_model.model.layers.{L-1}.feed_forward.w2.weight
```

The tensor is stored as `uint32` with shape `(out, nw)`, where `nw` is the number
of 32-bit words per row. The patch shape is the shape of the *code* matrix, not
the storage shape.

## MLX bit packing

MLX concatenates the 32-bit words of a row and reads codes from that row as a
little-endian bit stream. Bit `p` is bit `(p mod 32)` of word `p // 32`, with bit
0 the least significant bit. Code `j` occupies bits
`[j * bits, (j + 1) * bits)`, and bit `i` of the code contributes `2**i`:

```
code[j] = sum_{i=0}^{bits-1}
          ((words[j*bits + i // 32] >> ((j*bits + i) % 32)) & 1) << i
```

A row has `n = nw * 32 // bits` codes, so `shape_L == (out, n)`. If `n * bits` is
less than `nw * 32`, the leftover high bits of the last word are padding. They are
not codes and are copied back unchanged.

`apply_patch.py` unpacks and repacks every edited tensor before writing and aborts
unless the round trip reproduces the original bytes exactly.

## Semantics and validation

- Every `idx_L` entry must be unique and inside `0 .. out*in-1`.
- Every new code `code + sign` must stay in `0 .. 2**bits-1`. Otherwise the tools
  abort; they never clip or wrap.
- Only the listed code bits change. The safetensors header, metadata, all other
  tensors, the scales and the biases stay byte-for-byte unchanged.
- The target weight must be stored as `U32`. If `config.json` declares
  quantisation bits for the module, the patch's `bits_L` is checked against it.
- Applying the same patch to the same model is deterministic: the output file is
  byte-identical across runs.
- Two patches can be applied one after another. The tools do not merge patches;
  remove them in reverse order if they touch the same code. Each apply writes a
  fresh manifest for that install, so keep the pristine SHA-256 if you need to
  peel a whole stack back and the older manifest has been replaced.

## Install

```
.venv/bin/python patches/apply_patch.py <model_dir> <patch.npz> --out <copy_dir>
```

The tool copies the complete model directory (config, tokenizer files, shards and
any auxiliary weights) to `<copy_dir>`, patches the relevant safetensors file in
place in the copy, and prints the original and new SHA-256 of each modified
safetensors file and the number of codes changed.

`<copy_dir>` must not exist, and must not be inside `<model_dir>`. The original
model directory is never written to.

The copy also receives a hidden `.flippatch.json` manifest:

```json
{
  "format": "flippatch-apply-v1",
  "patch_sha256": "...",
  "files": {
    "model.safetensors": {
      "original_sha256": "...",
      "patched_sha256": "...",
      "codes_changed": 123,
      "layers": [8, 12]
    }
  }
}
```

The manifest is not part of the patch format. It records what was installed so
that uninstall can be checked without keeping a second copy of the model.

## Uninstall

```
.venv/bin/python patches/remove_patch.py <patched_dir> <patch.npz> --out <restored_dir>
```

The tool applies the inverse signs, prints the expected original SHA-256 and the
restored SHA-256, and fails unless they are equal. It verifies the patch SHA and
the patched file SHA against `.flippatch.json` before writing, so pointing it at
the wrong directory or patch is an error.

`--expect-sha256 <hex>` supplies the original safetensors SHA-256 directly. It is
required if the manifest is missing and is only meaningful when the patch edits a
single safetensors file. If both are present they must agree.

On success the manifest is deleted from `<restored_dir>`; the restored model is
otherwise a byte-identical copy of the pre-patch model.

## Making a patch

```python
import numpy as np

L = 8
idx = np.array([11, 33, 55], dtype=np.uint32)
sign = np.array([+1, -1, +1], dtype=np.int8)
np.savez_compressed(
    "patches/example.npz",
    **{
        f"idx_{L}": idx,
        f"sign_{L}": sign,
        f"shape_{L}": np.array([2048, 7680], dtype=np.int32),
        f"bits_{L}": np.array([4], dtype=np.int32),
    },
)
```

The patch changes only integer codes. It does not carry a model hash, so keep the
original model directory or the manifest (or use `--expect-sha256`) if you need to
prove an uninstall later.

## Testing

```
.venv/bin/python patches/test_patch_toolkit.py
```

The test builds a 640-byte synthetic safetensors file containing one 4-bit and
one 6-bit matrix, runs both CLIs as subprocesses, and checks exact code changes,
byte-identical restore, rejected out-of-range/overflow patches, untouched tensors,
headers and metadata. It never imports MLX or touches a model.

## Limits

- Only MLX `U32` packed weights are supported, with 4 or 6 bits per code.
- Only `feed_forward.w2.weight` tensors named above are edited.
- The full model directory is copied, so the output needs the same disk space as
  the source model.
- The tools use only numpy and the Python standard library; they never import
  MLX and never execute the model.
