"""AG News protocol (public, seeded). Eval = 1,000 test items; stream/pool = 2,000 train items."""
import numpy as np, pyarrow.parquet as pq, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
AG_LABELS = ["world", "sports", "business", "tech"]
AG_SYSTEM = ("You sort news articles into topics. Labels:\n- world: world news and politics\n- sports: sports\n"
             "- business: business, markets, companies\n- tech: science and technology\nAnswer with the label only.")
def clean(t): return " ".join(t.replace("\\", " ").split())[:300]
def load(seed=0, n_eval=1000, n_pool=2000):
    te = pq.read_table(os.path.join(HERE, "data", "agnews_test.parquet")).to_pylist()
    tr = pq.read_table(os.path.join(HERE, "data", "agnews_train.parquet")).to_pylist()
    rng = np.random.default_rng(seed)
    ev = [te[i] for i in rng.choice(len(te), n_eval, replace=False)]
    po = [tr[i] for i in rng.choice(len(tr), n_pool, replace=False)]
    return ([clean(r["text"]) for r in ev], np.array([r["label"] for r in ev]),
            [clean(r["text"]) for r in po], np.array([r["label"] for r in po]))
