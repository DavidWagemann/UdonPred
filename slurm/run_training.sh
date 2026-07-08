#!/bin/bash
# Runs INSIDE the pyxis/enroot container (launched by slurm/train.sbatch).
# Home is not mounted; only the project is (at the mount point). All heavy,
# regenerable state goes on node-local scratch to stay off the quota-limited
# DSS filesystem.
set -euo pipefail

# Move to the project root (parent of this slurm/ dir) regardless of workdir.
cd "$(dirname "$(readlink -f "$0")")/.."

# Node-local scratch. If /tmp is RAM-backed (tmpfs) on these nodes, point
# SCRATCH at the node's local SSD instead (a multi-GB venv + embeddings cache
# in RAM would be bad).
SCRATCH="${SCRATCH:-/tmp/udonpred-${SLURM_JOB_ID:-$$}}"
export HF_HOME="$SCRATCH/hf"                   # downloaded embeddings .h5 + model config
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv"  # project venv (torch, etc.)
mkdir -p "$HF_HOME"

# Keep W&B from trying to authenticate against the (unmounted) home in batch.
# For live logging, set WANDB_MODE=online and export WANDB_API_KEY.
export WANDB_MODE="${WANDB_MODE:-offline}"

# uv isn't in the base image; install it on first use (into the ephemeral
# container system python — fine, it's thrown away with the job).
command -v uv >/dev/null || pip install --no-cache-dir uv

uv sync --extra training --extra hub

# Datasets: the jsonl live only on the Hub (data/ is gitignored). Put them and
# the fat float32 hf/ Arrow cache that build_datasets writes on scratch via a
# symlinked data/split. The FrustrAISeq embeddings still stream from the Hub.
if [ -d data/split ] && [ ! -L data/split ]; then
  echo "data/split is a real directory; leaving it (its hf/ cache will use DSS)" >&2
else
  mkdir -p "$SCRATCH/data"
  ln -sfn "$SCRATCH/data" data/split
fi

uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "udonpred/datasets", repo_type="dataset",
    local_dir="data/split",
    allow_patterns=["trizod/*.jsonl", "chezod/*.jsonl"],
)
PY

uv run udonpred-train train
