#!/bin/bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

DEST="${DEST:-data/split}"
REPO="${REPO:-udonpred/datasets}"

python -c 'import huggingface_hub' 2>/dev/null || pip install --quiet 'huggingface_hub>=0.25'

DEST="$DEST" REPO="$REPO" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["REPO"],
    repo_type="dataset",
    allow_patterns=["*/*.jsonl", "*/*.fasta"],
    local_dir=os.environ["DEST"],
)
PY

echo "Fetched jsonl+fasta from $REPO -> $(cd "$DEST" && pwd)"
