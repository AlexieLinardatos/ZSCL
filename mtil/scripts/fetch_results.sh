#!/usr/bin/env bash
# Pull every task_summary.csv from Nibi into a local mirror (a few KB total).
#
#   bash mtil/scripts/fetch_results.sh            # -> results_sync/
#   bash mtil/scripts/fetch_results.sh <dest_dir>
#
# One MFA prompt only. Equivalent one-liner if you prefer no script:
#   ssh nibi 'cd projects/def-fqureshi/alexie/ZSCL/mtil && \
#     tar cz $(find ckpt thesis_results -name task_summary.csv)' | tar xz -C results_sync
#
# Needs a live ssh session to `nibi` (MFA): log in once in another terminal so
# the ControlMaster socket exists, then run this.
set -euo pipefail

REMOTE=${REMOTE:-nibi}
RROOT=${RROOT:-projects/def-fqureshi/alexie/ZSCL/mtil}
DEST=${1:-results_sync}
SUBDIRS=${SUBDIRS:-"ckpt thesis_results"}

for sub in $SUBDIRS; do
  echo "--- $sub"
  rsync -am --prune-empty-dirs \
    --include='*/' --include='task_summary.csv' --exclude='*' \
    "$REMOTE:$RROOT/$sub/" "$DEST/$sub/" || echo "  (skipped $sub)"
done

echo
echo "Fetched $(find "$DEST" -name task_summary.csv | wc -l) summaries into $DEST/"
