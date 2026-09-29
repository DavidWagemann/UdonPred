#!/bin/bash
set -euo pipefail

# Download the gitignored per-target jsonl+fasta splits into data/split/. Needs
# uv (runs a throwaway environment) or a python3 that has huggingface_hub;
# scripts/run_training.sh calls it inside the training container on first use.

cd "$(dirname "$(readlink -f "$0")")/.."

DEST="${DEST:-data/split}"
REPO="${REPO:-udonpred/datasets}"

if command -v uv >/dev/null; then
  PYTHON=(uv run --no-project --with 'huggingface_hub>=0.25' python)
elif python3 -c 'import huggingface_hub' 2>/dev/null; then
  PYTHON=(python3)
else
  echo "ERROR: needs uv or a python3 with huggingface_hub installed." >&2
  echo "scripts/docker/train.sh needs neither: it fetches the data inside the container." >&2
  exit 1
fi

DEST="$DEST" REPO="$REPO" "${PYTHON[@]}" - <<'PY'
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
