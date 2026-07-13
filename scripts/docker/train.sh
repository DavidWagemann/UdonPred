#!/bin/bash
#
# Docker training orchestration — a `docker run` adaptation of
# scripts/slurm/train.sbatch. It trains each (target x pLM) combination
# sequentially in a GPU container, sharing scripts/run_training.sh with the
# Slurm launcher (that script just cds to the project root and runs uv sync +
# training, so it is container-agnostic).
#
# Prerequisites:
#   - Docker with the NVIDIA Container Toolkit, so `--gpus` works.
#   - WANDB_API_KEY exported in your shell (unless WANDB_MODE=offline). Like the
#     Slurm job, the container runs non-interactively, so ~/.netrc is never read
#     inside it; the key must come from the host environment.
#
# Usage:
#   scripts/docker/train.sh              # all (target x pLM) combos, sequentially
#   GPUS=device=0 scripts/docker/train.sh
#   IMAGE=myregistry/pytorch:xx WANDB_MODE=offline scripts/docker/train.sh

set -euo pipefail

# Project root = parent of this scripts/ dir, regardless of the current workdir.
PROJECT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"

# --- config (override via env) ----------------------------------------------
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:23.10-py3}"  # same base as the Slurm .sqsh
MOUNT="${MOUNT:-/mnt/udonpred}"                      # project mount inside container
GPUS="${GPUS:-all}"                                  # value passed to `docker run --gpus`
CACHE_VOLUME="${CACHE_VOLUME:-udonpred-scratch}"     # named volume: persists the venv +
                                                     # HF cache across the sequential runs,
                                                     # mirroring the Slurm warm-node reuse.

# W&B is forwarded from the host env (see prerequisites). Fail fast here, before
# spinning up a container that would only re-hit the same check in run_training.sh.
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is not set in your shell." >&2
  echo "Export it before running, or set WANDB_MODE=offline. Aborting." >&2
  exit 1
fi

# One run per (target x pLM), trained separately (no dataset mixing): trizod then
# chezod, each on frustraiseq then prostt5. Sequential so each reuses the warm
# venv + HF cache in the shared scratch volume, with no concurrent uv-sync race.
TARGETS=(trizod trizod chezod chezod)
PLMS=(frustraiseq prostt5 frustraiseq prostt5)

for i in "${!TARGETS[@]}"; do
  target="${TARGETS[$i]}"
  plm="${PLMS[$i]}"
  echo ">>> Training $target x $plm ($((i + 1))/${#TARGETS[@]})"
  docker run --rm \
    --gpus "$GPUS" \
    -e WANDB_MODE \
    -e WANDB_API_KEY \
    -e SCRATCH=/scratch \
    -v "$PROJECT:$MOUNT" \
    -v "$CACHE_VOLUME:/scratch" \
    -w "$MOUNT" \
    "$IMAGE" \
    bash "$MOUNT/scripts/run_training.sh" "$target" "$plm"
done
