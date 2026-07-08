#!/bin/bash
#SBATCH --job-name=udonpred-train
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# --- paths (edit if the project or image move) ------------------------------
PROJECT=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge39reb3/udonpred
IMAGE=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge39reb3/nvidia+pytorch+23.10-py3.sqsh
MOUNT=/mnt/udonpred

mkdir -p "$PROJECT/slurm/logs"

srun \
  --container-image="$IMAGE" \
  --no-container-mount-home \
  --container-mounts="$PROJECT:$MOUNT" \
  bash "$MOUNT/slurm/run_training.sh"
