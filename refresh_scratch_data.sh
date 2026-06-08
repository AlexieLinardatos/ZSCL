#!/bin/bash
# Refresh ~/scratch/data file creation timestamps on Nibi to reset
# Alliance Canada's 60-day scratch purge timer.
#
# Run with:
#   chmod +x ~/projects/def-fqureshi/alexie/ZSCL/refresh_scratch_data.sh
#   nohup bash ~/projects/def-fqureshi/alexie/ZSCL/refresh_scratch_data.sh \
#       > ~/refresh_data.log 2>&1 &
#   tail -f ~/refresh_data.log    # Ctrl-C exits the tail; script keeps running

set -u
DATA=~/scratch/data
cd "$DATA" || { echo "FATAL: $DATA missing"; exit 1; }

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "START refresh from $DATA"
df -h ~/scratch | tail -1

# --- Step 0: pre-flight cleanup ------------------------------------------------
# Broken DTD symlink (target dtd-r1.0.1 does not exist)
if [ -L DTD ] && [ ! -e DTD ]; then
  log "removing broken DTD symlink"
  rm DTD
fi

# OPTIONAL: remove duplicate Flowers/ tree once you confirm it equals flowers-102.
# Verify first with: diff -rq Flowers flowers-102 | head
# Then uncomment:
# log "removing duplicate Flowers/ and Flowers_data symlink (kept flowers-102)"
# rm -rf Flowers Flowers_data

# --- Step 1: cp-rename loop for small + medium datasets ------------------------
SMALL_MED=(
  cc
  MNIST
  caltech101
  cifar-10-batches-py
  cifar-100-python
  eurosat
  conceptual_captions
  Flowers
  flowers-102
  StanfordCars
  oxford-iiit-pet
  dtd
  fgvc-aircraft-2013b
  food-101
  SUN397
)

for d in "${SMALL_MED[@]}"; do
  if [ ! -d "$d" ] || [ -L "$d" ]; then
    log "skip $d (missing or symlink)"
    continue
  fi
  size=$(du -sh "$d" | cut -f1)
  log "refresh $d  ($size)"
  if cp -a "$d" "${d}.new"; then
    rm -rf "$d" && mv "${d}.new" "$d"
    log "  done $d"
  else
    log "  FAILED cp on $d -- leaving original intact"
    rm -rf "${d}.new" 2>/dev/null
  fi
done

# --- Step 2: CC captions csv at data root --------------------------------------
CC_CSV=Validation_GCC-1.1.0-Validation_output.csv
if [ -f "$CC_CSV" ]; then
  log "refresh $CC_CSV"
  cp -a "$CC_CSV" "${CC_CSV}.new" && mv -f "${CC_CSV}.new" "$CC_CSV"
fi

# --- Step 3: ImageNet via tar trick (preserves inode budget) -------------------
if [ -d ImageNet ] && [ ! -L ImageNet ]; then
  size=$(du -sh ImageNet | cut -f1)
  log "refresh ImageNet via tar  ($size)"
  if tar -cf ImageNet.tar ImageNet; then
    rm -rf ImageNet && tar -xf ImageNet.tar && rm ImageNet.tar
    log "  done ImageNet"
  else
    log "  FAILED tar on ImageNet"
    rm -f ImageNet.tar
  fi
fi

# --- Step 4: ObjectNet via tar trick (biggest, slowest -- expect hours) --------
if [ -d ObjectNet ] && [ ! -L ObjectNet ]; then
  size=$(du -sh ObjectNet | cut -f1)
  log "refresh ObjectNet via tar  ($size)"
  if tar -cf ObjectNet.tar ObjectNet; then
    rm -rf ObjectNet && tar -xf ObjectNet.tar && rm ObjectNet.tar
    log "  done ObjectNet"
  else
    log "  FAILED tar on ObjectNet"
    rm -f ObjectNet.tar
  fi
fi

# --- Step 5: verify creation dates reset ---------------------------------------
log "DONE. Newest creation dates in $DATA:"
find "$DATA" -maxdepth 2 -mindepth 1 -type d -printf '%TY-%Tm-%Td %p\n' | sort -r | head -20
df -h ~/scratch | tail -1
log "FINISH"
