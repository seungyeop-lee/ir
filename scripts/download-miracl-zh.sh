#!/usr/bin/env bash
# download-miracl-zh.sh — Download MIRACL Chinese corpus in BEIR format
# Output: test-data/miracl-zh/{corpus.jsonl, queries.jsonl, qrels/test.tsv}
#
# Requires: uv (https://github.com/astral-sh/uv)
set -euo pipefail

OUT="test-data/miracl-zh"
mkdir -p "$OUT/qrels"

uv run --with datasets --with huggingface_hub --with tqdm python3 - <<'PYEOF'
import json, pathlib, sys
from datasets import load_dataset
from tqdm import tqdm

out = pathlib.Path("test-data/miracl-zh")
out.mkdir(parents=True, exist_ok=True)
(out / "qrels").mkdir(exist_ok=True)

corpus_path = out / "corpus.jsonl"
if corpus_path.exists():
    print(f"corpus already exists ({corpus_path.stat().st_size // 1_000_000}MB), skipping")
else:
    print("Downloading MIRACL-ZH corpus (~132K passages)...")
    corpus_ds = load_dataset("miracl/miracl-corpus", "zh", split="train")
    with open(corpus_path, "w") as f:
        for row in tqdm(corpus_ds, desc="corpus", unit="doc"):
            f.write(json.dumps({
                "_id":   row["docid"],
                "title": row.get("title", ""),
                "text":  row["text"],
            }, ensure_ascii=False) + "\n")
    print(f"  done ({corpus_path.stat().st_size // 1_000_000}MB)")

print("Downloading MIRACL-ZH queries + qrels...")
miracl_ds = load_dataset("miracl/miracl", "zh")

queries_path = out / "queries.jsonl"
qrels_path   = out / "qrels" / "test.tsv"

seen_queries = {}
with open(queries_path, "w") as qf, open(qrels_path, "w") as rf:
    rf.write("query-id\tcorpus-id\tscore\n")
    for split in ["dev", "test"]:
        if split not in miracl_ds:
            continue
        for row in tqdm(miracl_ds[split], desc=split, unit="query"):
            qid = row["query_id"]
            if qid not in seen_queries:
                seen_queries[qid] = True
                qf.write(json.dumps({"_id": qid, "text": row["query"]}, ensure_ascii=False) + "\n")
            for doc in row.get("positive_passages", []):
                rf.write(f"{qid}\t{doc['docid']}\t1\n")

print(f"  {len(seen_queries)} queries, qrels written")
print(f"\nDone. Dataset at test-data/miracl-zh/")
print("Next: bash scripts/generate-fixtures.sh  # generates miracl-zh-mini")
print("Then: scripts/bench.sh miracl-zh --size 2000")
PYEOF
