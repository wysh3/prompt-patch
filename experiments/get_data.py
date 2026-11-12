"""Download AG News and write data/agnews_{train,test}.parquet (columns: text, label)."""
import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
ds = load_dataset("fancyzhx/ag_news")
for split in ("train", "test"):
    ds[split].to_parquet(f"data/agnews_{split}.parquet")
    print(split, len(ds[split]))
