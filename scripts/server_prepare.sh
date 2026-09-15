#!/bin/bash
# One-shot server setup: download every dataset shard and build the training corpus.
#   bash scripts/server_prepare.sh [n_shards=74] [workers=32] [out=data/prep]
set -e
export PATH="$HOME/.local/bin:$PATH"   # uv lives here when installed by the bootstrap
N=${1:-74}; WORKERS=${2:-32}; OUT=${3:-data/prep}
cd "$(dirname "$0")/.."
python - <<PY
from huggingface_hub import hf_hub_download
for i in range($N):
    p = hf_hub_download("project-riz/osu-beatmaps", f"compressed/data-{i:06d}.tar", repo_type="dataset", local_dir="data/hf")
    print("shard", i, "ok", flush=True)
PY
python -m autoosu.ml.prepare_data --shards data/hf/compressed --out "$OUT" --workers "$WORKERS"
echo "corpus ready: $OUT ($(wc -l < "$OUT/index.jsonl") std beatmaps)"
