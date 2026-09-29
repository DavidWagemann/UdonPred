#!/bin/bash
set -euo pipefail

# Train the target x pLM matrix sequentially in a GPU container:
#   SCRATCH=/local/disk scripts/docker/train.sh
# SCRATCH (required) holds the venv and HF cache across runs; like the project
# itself it must be on a local disk. Optional: IMAGE, MOUNT, GPUS, NETWORK,
# WANDB_MODE.

PROJECT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"

if [ "$(stat -f -c %T "$PROJECT" 2>/dev/null)" = nfs ]; then
  echo "ERROR: project is on NFS ($PROJECT). Docker's root daemon can't bind-mount" >&2
  echo "root-squashed NFS. Stage it onto a local disk first:" >&2
  echo "  DEST=/local/disk/UdonPred scripts/docker/stage.sh && cd /local/disk/UdonPred && scripts/docker/train.sh" >&2
  exit 1
fi

IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:23.10-py3}"
MOUNT="${MOUNT:-/mnt/udonpred}"
GPUS="${GPUS:-all}"
SCRATCH="${SCRATCH:?set SCRATCH to a local-disk directory for the venv + HF cache}"
NETWORK="${NETWORK:-host}"

export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is not set in your shell." >&2
  echo "Export it before running, or set WANDB_MODE=offline. Aborting." >&2
  exit 1
fi

TARGETS=(trizod trizod chezod chezod)
PLMS=(frustraiseq prostt5 frustraiseq prostt5)

for i in "${!TARGETS[@]}"; do
  target="${TARGETS[$i]}"
  plm="${PLMS[$i]}"
  echo ">>> Training $target x $plm ($((i + 1))/${#TARGETS[@]})"
  docker run --rm \
    --gpus "$GPUS" \
    --network "$NETWORK" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e WANDB_MODE \
    -e WANDB_API_KEY \
    -e http_proxy -e https_proxy -e no_proxy \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -e SCRATCH=/scratch \
    -v "$PROJECT:$MOUNT" \
    -v "$SCRATCH:/scratch" \
    -w "$MOUNT" \
    "$IMAGE" \
    bash "$MOUNT/scripts/run_training.sh" "$target" "$plm"
done
