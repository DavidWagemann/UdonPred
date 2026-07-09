#!/bin/bash
set -euo pipefail

# Move to the project root (parent of this slurm/ dir) regardless of workdir.
cd "$(dirname "$(readlink -f "$0")")/.."

# Optional target + embeddings pLM for this run (passed by train.sbatch's job
# array). Trains on the single target with that pLM, under a distinct run name /
# checkpoint dir (checkpoints/<target>-<plm>) and W&B run.
TARGET="${1:-}"
PLM="${2:-}"
if [ -n "$TARGET" ]; then
  export UDONPRED_TARGET="$TARGET"
fi
if [ -n "$PLM" ]; then
  export UDONPRED_EMBEDDINGS_PLM="$PLM"
fi
if [ -n "$TARGET" ] && [ -n "$PLM" ]; then
  export UDONPRED_RUN_NAME="$TARGET-$PLM"
fi

# Node-local scratch. Fixed (not per-job) so the venv + embeddings cache are
# reused across jobs that land on the same node. If /tmp is RAM-backed (tmpfs)
# on these nodes, point SCRATCH at the node's local SSD instead (a multi-GB venv
# + embeddings cache in RAM would be bad).
SCRATCH="${SCRATCH:-/tmp/ge39reb3_udonpred}"
export HF_HOME="$SCRATCH/hf"                   # downloaded embeddings .h5 + model config
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv"  # project venv (torch, etc.)
mkdir -p "$HF_HOME"

# W&B: log online. The container runs with --no-container-mount-home and this
# script is non-interactive, so ~/.bashrc / ~/.netrc are never read inside the
# container. WANDB_API_KEY must therefore be exported in the shell you sbatch
# from (e.g. via ~/.bashrc), so sbatch's default --export=ALL carries it into
# the job and pyxis forwards it into the container.
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is not set in the job" >&2
  echo "environment. Export it in your login shell before sbatch (it must be in" >&2
  echo "the env at submit time, not just interactive .bashrc), or resubmit with" >&2
  echo "WANDB_MODE=offline. Aborting before uv sync to avoid an interactive hang." >&2
  exit 1
fi

# uv isn't in the base image; install it on first use (into the ephemeral
# container system python — fine, it's thrown away with the job).
command -v uv >/dev/null || pip install --no-cache-dir uv

uv sync --extra training --extra hub

# Single-process (1 GPU) training. Under srun, HF Trainer/accelerate detect the
# SLURM env and try to init torch.distributed via env:// rendezvous, which fails
# with "WORLD_SIZE expected, but not set". Pin a 1-rank world so the process
# group initializes cleanly instead. Override these for a real multi-GPU launch.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

uv run udonpred-train train
