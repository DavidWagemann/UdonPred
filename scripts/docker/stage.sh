#!/bin/bash
set -euo pipefail

# Copy the working tree onto a local (non-NFS) disk for scripts/docker/train.sh:
#   DEST=/local/disk/UdonPred scripts/docker/stage.sh
# EXCLUDE takes extra space-separated rsync patterns to skip, e.g.
#   EXCLUDE="/data/split/trizod2/ /some_stray_embeddings.h5"

SOURCE="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
DEST="${DEST:?set DEST to a directory on a local (non-NFS) disk to stage into}"

EXCLUDES=(--exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' --exclude='/checkpoints/')
# read -a splits on whitespace without glob-expanding patterns like *.h5
read -r -a extra <<< "${EXCLUDE:-}"
for pattern in ${extra[@]+"${extra[@]}"}; do
  EXCLUDES+=("--exclude=$pattern")
done

if [ "$(stat -f -c %T "$(dirname "$DEST")" 2>/dev/null)" = nfs ]; then
  echo "ERROR: DEST ($DEST) is on NFS; stage onto a local disk instead." >&2
  exit 1
fi

mkdir -p "$DEST"
rsync -a --info=progress2 "${EXCLUDES[@]}" "$SOURCE"/ "$DEST"/

echo "Staged $SOURCE -> $DEST"
echo "Next: cd $DEST && SCRATCH=<local dir> scripts/docker/train.sh"
