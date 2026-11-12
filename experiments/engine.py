"""One model for everything: LFM3-Omni-2B (MLX, 4-bit body / 6-bit embeddings).

Each task takes the cheapest path through the same weights:
  sort   - first `exit_layer` layers only, mean-pooled hidden state, scored against
           classifier rows written into UNUSED vocabulary slots (ids the tokenizer can
           never produce). Fit is closed-form ridge, no gradient descent; a correction
           refits in milliseconds. The slots are masked out of every generation path.
  ask    - full backbone; the tab listing is prefilled once into a KV/conv cache and
           each question only pays for its own tokens.
  voice  - audio tower + backbone, text out (JSON intent).
  speak  - backbone + depth decoder + codec, interleaved mode (pure audio mode babbles).
  see    - vision tower + backbone.
Towers load lazily on first use, so sort/ask keep only the backbone resident.
"""
import copy, time
import numpy as np
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

SLOT0 = 125100            # tokenizer max id is 125032; rows up to 127999 are never used
MAX_LABELS = 64


class OmniEngine:
    def __init__(self, path, exit_layer=12, cache_limit_mb=256, ridge=10.0):
        mx.set_cache_limit(cache_limit_mb * 1024 * 1024)
        self.model, self.proc = load(path, lazy=True)
        self.tok = self.proc.tokenizer
        self.lm = self.model.language_model
        self.core = self.lm.model
        mx.eval(self.lm.parameters())
        self.loaded = {"language_model"}
        slots = list(range(SLOT0, SLOT0 + MAX_LABELS))
        self.lm._placeholder_ids = list(self.lm._placeholder_ids) + slots        # text path
        base = self.model.placeholder_token_ids                                    # speech/vision path
        self.model.placeholder_token_ids = lambda: base() + slots
        self.exit_layer, self.ridge, self.chunk = exit_layer, ridge, 1024
        self.labels, self.feats, self.ys = [], [], []
        self._rows = None
        self._prefix = {}
        self.stops = set(np.atleast_1d(self.model.config.eos_token_id).tolist())
        self.stops |= {self.model.config.text_end_token_id, *self.proc.additional_eos_token_ids}
        emb = self.core.embed_tokens
        self.qbits, self.qgroup = emb.bits, emb.group_size

    # ---------- memory: towers on demand
    def ensure(self, *parts):
        for p in parts:
            if p not in self.loaded and hasattr(self.model, p):
                mx.eval(getattr(self.model, p).parameters()); self.loaded.add(p)

    # ---------- sort: early-exit features + in-weight classifier
    TMPL = "Title: {t}\nSite: {d}"

    def features(self, items, upto=None, all_layers=False, tmpl=None):
        """items: [(title, site)] -> mean-pooled normed hidden after `upto` layers."""
        upto = upto or self.exit_layer
        ids = [self.tok.encode((tmpl or self.TMPL).format(t=t, d=d), add_special_tokens=False) for t, d in items]
        L = max(map(len, ids)); x = mx.array([c + [0] * (L - len(c)) for c in ids])
        m = (mx.arange(L)[None] < mx.array([len(c) for c in ids])[:, None]).astype(mx.float32)[..., None]
        h = self.core.embed_tokens(x); outs = []
        for layer in self.core.layers[:upto]:
            if layer.is_attention_layer:
                w = self.core.config.sliding_window if layer.layer_type == "sliding_attention" else None
                mask = create_attention_mask(h, None, window_size=w)
            else:
                mask = create_ssm_mask(h, None)
            h = layer(h, mask, None, None)
            if all_layers:
                outs.append((self.core.embedding_norm(h) * m).sum(1) / m.sum(1))
        if all_layers:
            mx.eval(outs); return [np.array(o.astype(mx.float32)) for o in outs]
        f = (self.core.embedding_norm(h) * m).sum(1) / m.sum(1)
        return np.array(f.astype(mx.float32))

    def _write_rows(self, W):          # W: [C, hidden] -> quantized slot rows
        emb = self.core.embed_tokens; C = W.shape[0]
        idx = mx.arange(SLOT0, SLOT0 + C)
        q, s, b = mx.quantize(mx.array(W.astype(np.float32)).astype(mx.bfloat16), group_size=self.qgroup, bits=self.qbits)
        emb.weight[idx] = q; emb.scales[idx] = s.astype(emb.scales.dtype); emb.biases[idx] = b.astype(emb.biases.dtype)
        self._rows = mx.dequantize(emb.weight[idx], emb.scales[idx], emb.biases[idx],
                                   group_size=self.qgroup, bits=self.qbits).astype(mx.float32)
        mx.eval(self._rows)

    def fit(self):
        X = np.stack(self.feats); Y = np.eye(len(self.labels))[self.ys]
        W = np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ Y)   # closed form
        self._write_rows(W.T)

    def learn(self, items, labels, refit=True):
        F = self.features(items)
        for f, l in zip(F, labels):
            if l not in self.labels:
                assert len(self.labels) < MAX_LABELS; self.labels.append(l)
            self.feats.append(f); self.ys.append(self.labels.index(l))
        if refit: self.fit()

    # ---------- exact online learning: recursive least squares (Sherman-Morrison), O(d^2) per correction,
    # identical to re-solving ridge on everything seen so far.
    def rls_reset(self):
        d = self.core.config.hidden_size
        self.P = np.eye(d) / self.ridge; self.B = np.zeros((d, 0)); self.labels = []

    def rls_add(self, f, label, write=True):
        f = np.asarray(f, np.float64)
        if label not in self.labels:
            self.labels.append(label); self.B = np.hstack([self.B, np.zeros((self.B.shape[0], 1))])
        Pf = self.P @ f; self.P -= np.outer(Pf, Pf) / (1.0 + f @ Pf)
        self.B[:, self.labels.index(label)] += f
        if write: self._write_rows((self.P @ self.B).T)

    def sort(self, items):
        F = mx.array(self.features(items))
        s = F @ self._rows.T
        p = mx.softmax(s * 8.0, axis=-1)
        k = mx.argmax(s, axis=-1).tolist(); pr = p.max(-1).tolist()
        return [(self.labels[i], round(c, 2)) for i, c in zip(k, pr)]

    # ---------- text generation with optional cached prefix
    def _greedy(self, ids, cache, max_tokens):
        out = []; x = mx.array([ids])
        for _ in range(max_tokens):
            t = int(mx.argmax(self.lm(x, cache=cache).logits[0, -1]).item())
            if t in self.stops: break
            out.append(t); x = mx.array([[t]])
        return self.tok.decode(out, skip_special_tokens=True).strip()

    def _chat(self, user, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def ask(self, listing, question, max_tokens=160):
        key = hash(listing)
        if key not in self._prefix:
            a = self.tok.encode(self._chat(f"My open browser tabs:\n{listing}\n\nQuestion: A"), add_special_tokens=False)
            b = self.tok.encode(self._chat(f"My open browser tabs:\n{listing}\n\nQuestion: B"), add_special_tokens=False)
            n = next(i for i in range(len(a)) if a[i] != b[i])
            c = self.lm.make_cache()
            for i in range(0, n, self.chunk):                     # chunked prefill caps peak RAM
                mx.eval(self.core(mx.array([a[i:min(i + self.chunk, n)]]), cache=c))   # backbone only: no 128K-wide logits
                mx.clear_cache()
            self._prefix = {key: (a[:n], c)}          # keep one listing resident
        pre, c0 = self._prefix[key]
        ids = self.tok.encode(self._chat(f"My open browser tabs:\n{listing}\n\nQuestion: {question}"), add_special_tokens=False)
        assert ids[:len(pre)] == pre
        return self._greedy(ids[len(pre):], copy.deepcopy(c0), max_tokens)

    # ---------- multimodal
    def _gen(self, system, user, mode, max_tokens, **media):
        inp = self.proc(text=self._chat(user, system), **media)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        return self.model.generate_audio(inp["input_ids"], self.tok, generation_mode=mode,
                                         max_tokens=max_tokens, temperature=0.0, **kw)

    def voice(self, audio16k):
        self.ensure("audio_tower", "audio_projector")
        r = self._gen("Respond with text.", '<|audio|>\nThe user is talking to their web browser. Reply with JSON only: '
                      '{"open_thread": str|null, "close_topic": str|null, "question": str|null}', "text", 80, audio=[audio16k])
        return r.text.strip()

    def speak(self, text):
        self.ensure("depth_decoder", "codebook_embedding")
        r = self._gen("Respond with interleaved text and audio.", f"Say exactly this, nothing else: {text}", "interleaved", 600)
        return np.array(r.audio.astype(mx.float32)).reshape(-1), r.sample_rate, r.text.strip()

    def see(self, image):
        self.ensure("vision_tower", "multi_modal_projector")
        r = self._gen("Respond with text.", "<|image|>\nIn one sentence, what is this web page about?", "text", 60, images=[image])
        return r.text.strip()

    def mem(self):
        return dict(active_mb=round(mx.get_active_memory() / 1e6), peak_mb=round(mx.get_peak_memory() / 1e6),
                    cache_mb=round(mx.get_cache_memory() / 1e6), loaded=sorted(self.loaded))
