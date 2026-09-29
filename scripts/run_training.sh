#!/bin/bash
set -euo pipefail

# Move to the project root (parent of this scripts/ dir) regardless of workdir.
cd "$(dirname "$(readlink -f "$0")")/.."

# Optional target + embeddings pLM for this run (passed by train.sbatch's job
# array). Trains on the single target with that pLM; run.py derives the run name
# / checkpoint dir / W&B run as <target>-<plm> from these.
TARGET="${1:-}"
PLM="${2:-}"
if [ -n "$TARGET" ]; then
  export UDONPRED_TARGET="$TARGET"
fi
if [ -n "$PLM" ]; then
  export UDONPRED_EMBEDDINGS_PLM="$PLM"
fi

# Node-local scratch. Fixed (not per-job) so the venv + embeddings cache are
# reused across jobs that land on the same node. If /tmp is RAM-backed (tmpfs)
# on these nodes, point SCRATCH at the node's local SSD instead (a multi-GB venv
# + embeddings cache in RAM would be bad).
SCRATCH="${SCRATCH:-/tmp/${USER:-$(id -un)}_udonpred}"
export HF_HOME="$SCRATCH/hf"                   # downloaded embeddings .h5 + model config
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv"  # project venv (torch, etc.)
# uv, the Python it installs for the venv, and its download cache live in
# SCRATCH too: a container's own filesystem is gone after each run, and a venv
# whose interpreter went with it is unusable.
export UV_PYTHON_INSTALL_DIR="$SCRATCH/uv/python"
export UV_CACHE_DIR="$SCRATCH/uv/cache"
export PATH="$SCRATCH/uv/tool/bin:$PATH"
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

# uv isn't in the base image: install it into SCRATCH on first use. Skip the
# image's pip config, whose extra NVIDIA index is unreachable behind some proxies.
if ! command -v uv >/dev/null; then
  env -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
    pip install --no-cache-dir --quiet --target "$SCRATCH/uv/tool" uv
fi

# A venv whose interpreter is gone (created before uv's Python lived in
# SCRATCH, or half-removed by a killed run) is one uv refuses to recreate:
# start it over.
if [ -d "$UV_PROJECT_ENVIRONMENT" ] && ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "" 2>/dev/null; then
  echo "Removing unusable venv $UV_PROJECT_ENVIRONMENT"
  rm -rf "$UV_PROJECT_ENVIRONMENT"
fi

uv sync --extra training --extra hub

# The jsonl splits are gitignored: fetch them on first use, here in the
# container, so the host needs no Python packages.
if ! compgen -G "data/split/${TARGET:-*}/train.jsonl" > /dev/null; then
  scripts/fetch_data.sh
fi

# Single-process (1 GPU) training. Under srun, HF Trainer/accelerate detect the
# SLURM env and try to init torch.distributed via env:// rendezvous, which fails
# with "WORLD_SIZE expected, but not set". Pin a 1-rank world so the process
# group initializes cleanly instead. Override these for a real multi-GPU launch.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Random high port, not a fixed 29500: with Docker --network host the rendezvous
# port binds on the host, so a fixed port collides with a leftover run or another
# user on a shared node. Any free port works for this 1-rank group.
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

uv run udonpred-train train
