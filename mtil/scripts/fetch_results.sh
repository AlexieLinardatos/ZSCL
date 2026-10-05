#!/usr/bin/env bash
# Collect every task_summary.csv into a local mirror (a few KB total).
#
# RUN THIS IN YOUR LOCAL WSL TERMINAL, not on the cluster:
#   bash mtil/scripts/fetch_results.sh            # -> results_sync/
#   bash mtil/scripts/fetch_results.sh <dest_dir>
# One MFA prompt. Equivalent one-liner, if you prefer no script:
#   mkdir -p results_sync && ssh nibi 'cd projects/def-fqureshi/alexie/ZSCL/mtil && \
#     tar cz $(find ckpt thesis_results -name task_summary.csv)' | tar xz -C results_sync
#
# If it is run ON the cluster it detects that and just copies the CSVs out of
# the local ckpt dirs instead (no ssh), so you can build the workbook there and
# scp the single .xlsx down.
set -euo pipefail

REMOTE=${REMOTE:-nibi}
RROOT=${RROOT:-projects/def-fqureshi/alexie/ZSCL/mtil}
DEST=${1:-results_sync}
SUBDIRS=${SUBDIRS:-"ckpt thesis_results"}

# Are we on the cluster? Then there is nothing to fetch, only to gather.
ON_CLUSTER=0
[[ -n ${CC_CLUSTER:-} ]] && ON_CLUSTER=1
[[ $(hostname -f 2>/dev/null || hostname) == *alliancecan* || $(hostname) == *nibi* ]] && ON_CLUSTER=1

if [[ $ON_CLUSTER == 1 ]]; then
  echo "Running on the cluster -> gathering locally (no ssh)."
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # .../ZSCL/mtil
  for sub in $SUBDIRS; do
    [[ -d "$HERE/$sub" ]] || { echo "  (no $sub)"; continue; }
    echo "--- $sub"
    while IFS= read -r f; do
      rel=${f#"$HERE/"}
      mkdir -p "$DEST/$(dirname "$rel")"
      cp "$f" "$DEST/$rel"
    done < <(find "$HERE/$sub" -name task_summary.csv)
  done
else
  for sub in $SUBDIRS; do
    echo "--- $sub"
    mkdir -p "$DEST/$sub"
    rsync -am --prune-empty-dirs \
      --include='*/' --include='task_summary.csv' --exclude='*' \
      "$REMOTE:$RROOT/$sub/" "$DEST/$sub/" || echo "  (skipped $sub)"
  done
fi

n=$(find "$DEST" -name task_summary.csv 2>/dev/null | wc -l)
echo
echo "Collected $n summaries into $DEST/"
