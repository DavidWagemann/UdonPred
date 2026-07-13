#!/bin/bash
set -euo pipefail

SOURCE="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
DEST="${DEST:-/mnt/space/local/UdonPred}"

if [ "$(stat -f -c %T "$(dirname "$DEST")" 2>/dev/null)" = nfs ]; then
  echo "ERROR: DEST ($DEST) is on NFS; stage onto a local disk instead." >&2
  exit 1
fi

mkdir -p "$DEST"
rsync -a --info=progress2 \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='/checkpoints/' \
  --exclude='/frustraiseq_embeddings.h5' \
  --exclude='/data/split/trizod2/' \
  "$SOURCE"/ "$DEST"/

echo "Staged $SOURCE -> $DEST"
echo "Next: cd $DEST && scripts/docker/train.sh"
