"""Verify generic mixed-precision code access on the real model: unpack/pack round-trip and
manual dequantization vs mx.dequantize, for every edited w2 matrix of LFM3-Omni-2B-4bit."""
import json, numpy as np, mlx.core as mx
from engine import OmniEngine
E = OmniEngine("models/omni-2b-4bit"); core = E.core
import importlib.util
spec = importlib.util.spec_from_file_location("flipmod", "flip.py")
# do not import flip.py (it runs); copy the two functions here
def unpack(w):
    bits = int(w.bits); Wp = np.array(w.weight); rows, nw = Wp.shape
    b = Wp.view(np.uint8).reshape(rows, nw, 4)
    g = np.unpackbits(b, axis=-1).reshape(rows, nw, 4, 8)[..., ::-1].reshape(rows, nw * 32)
    n = (nw * 32) // bits
    g = g[:, :n * bits].reshape(rows, n, bits)
    c = np.zeros((rows, n), np.int64)
    for i in range(bits): c |= g[..., i].astype(np.int64) << i
    return c.astype(np.int8)
def pack(c, bits):
    rows, n = c.shape
    g = np.zeros((rows, n, bits), np.uint8)
    for i in range(bits): g[..., i] = (c >> i) & 1
    flat = g.reshape(rows, -1)
    pad = (-flat.shape[1]) % 32
    if pad: flat = np.concatenate([flat, np.zeros((rows, pad), np.uint8)], axis=1)
    nb = flat.shape[1] // 32
    b = np.packbits(flat.reshape(rows, nb, 4, 8)[..., ::-1], axis=-1).reshape(rows, nb * 4)
    return b.view(np.uint32).reshape(rows, nb)
out = {}
for li, layer in enumerate(core.layers):
    w = layer.feed_forward.w2
    bits = int(w.bits) if hasattr(w, "bits") else None
    if bits is None:
        out[li + 1] = dict(kind=type(w).__name__, bits=None); continue
    W = np.array(w.weight); codes = unpack(w); rep = pack(codes, bits)
    dq = np.array(mx.dequantize(w.weight, w.scales, w.biases, group_size=int(w.group_size), bits=bits).astype(mx.float32))
    S = np.repeat(np.array(w.scales.astype(mx.float32)), int(w.group_size), axis=1)
    B = np.repeat(np.array(w.biases.astype(mx.float32)), int(w.group_size), axis=1)
    man = codes.astype(np.float32) * S + B
    out[li + 1] = dict(bits=bits, codes_per_row=int(codes.shape[1]), roundtrip=bool(np.array_equal(rep, W)),
                       dequant_maxdiff=float(np.abs(man - dq).max()), code_min=int(codes.min()), code_max=int(codes.max()),
                       codes_mean=round(float(codes.mean()), 2), step_med=float(np.median(np.array(w.scales.astype(mx.float32)))),
                       rms=float(np.sqrt((dq ** 2).mean())))
json.dump(out, open("res_qbits_check.json", "w"), indent=1, default=lambda o: o.item())
print(json.dumps(out, indent=1, default=lambda o: o.item()))
