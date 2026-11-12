"""Induction score across Pythia checkpoints (standard test: random token sequence repeated twice).
score = mean loss on the first copy - mean loss on the second copy (nats/token)."""
import json, os, sys, torch, numpy as np
from transformers import GPTNeoXForCausalLM
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf"))
torch.manual_seed(0); g = torch.Generator().manual_seed(0); R = 50
IND = torch.randint(1000, 20000, (32, R), generator=g); IND = torch.cat([IND, IND], 1)
def score(m):
    with torch.no_grad():
        lg = m(IND).logits[:, :-1].float(); l = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), IND[:, 1:].reshape(-1), reduction="none").view(32, -1)
    first, second = l[:, 1:R - 1].mean().item(), l[:, R:].mean().item(); return round(first - second, 3), round(second, 3)
out = {}
jobs = [("pythia-70m", s) for s in (0, 512, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 143000)] + [("pythia-160m", 143000)]
for name, step in jobs:
    m = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{name}", revision=f"step{step}", cache_dir=os.environ["HF_HOME"]).eval()
    s = score(m); out[f"{name}@{step}"] = dict(induction_score=s[0], second_copy_loss=s[1]); print(name, step, s, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_pythia_scan.json"), "w"), indent=1)
