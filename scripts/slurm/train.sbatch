#!/bin/bash
#SBATCH --job-name=udonpred-train
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=scripts/slurm/logs/%x-%A_%a.out
#SBATCH --error=scripts/slurm/logs/%x-%A_%a.out
#SBATCH --array=0-3%1

# Submit from the repo root, naming the enroot image to run in:
#   IMAGE=/path/to/nvidia+pytorch+23.10-py3.sqsh sbatch scripts/slurm/train.sbatch
# Optional: PROJECT (checkout to mount; default: the submit dir) and MOUNT
# (its path in the container). On another cluster, override the partition
# with `sbatch -p <partition>`.

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE="${IMAGE:?set IMAGE to the enroot image to train in (e.g. an imported nvidia+pytorch+23.10-py3.sqsh)}"
MOUNT="${MOUNT:-/mnt/udonpred}"

mkdir -p "$PROJECT/scripts/slurm/logs"

# One array task per (target x pLM), trained separately (no dataset mixing):
# trizod then chezod, each on frustraiseq then prostt5. Sequential via %1 so each
# task reuses the warm venv + HF cache, with no concurrent uv-sync race.
TARGETS=(trizod trizod chezod chezod)
PLMS=(frustraiseq prostt5 frustraiseq prostt5)
i="${SLURM_ARRAY_TASK_ID:-0}"

srun \
  --container-image="$IMAGE" \
  --no-container-mount-home \
  --container-mounts="$PROJECT:$MOUNT" \
  bash "$MOUNT/scripts/run_training.sh" "${TARGETS[$i]}" "${PLMS[$i]}"
