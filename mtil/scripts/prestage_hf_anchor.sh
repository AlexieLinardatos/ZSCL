#!/bin/bash
# One-time pre-stage of HuggingFace anchor models into project space.
# Compute nodes on Nibi have no internet -- the model must be cached on a
# login node beforehand. The cache lives in project space (persistent, no
# scratch purge).
#
# Run on a LOGIN node:
#   bash mtil/scripts/prestage_hf_anchor.sh
#
# Override the model list with $HF_MODELS, e.g.:
#   HF_MODELS="Qwen/Qwen2-0.5B intfloat/e5-large-v2" \
#       bash mtil/scripts/prestage_hf_anchor.sh

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/projects/def-fqureshi/alexie/hf_cache}"
mkdir -p "$HF_HOME"
echo "[prestage] HF_HOME = $HF_HOME"

# Default model: the V1 mpnet anchor used in llm_anchor_lam03.sh
# Override via $HF_MODELS (space-separated list).
MODELS="${HF_MODELS:-sentence-transformers/all-mpnet-base-v2}"

module load python/3.11.5 2>/dev/null || true

# Use a throwaway venv so we don't pollute the login session
TMP_ENV=$(mktemp -d)/env
python -m venv "$TMP_ENV"
source "$TMP_ENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet transformers torch --index-url https://pypi.org/simple/ \
    || pip install --quiet transformers torch

for MODEL in $MODELS; do
  echo "[prestage] downloading $MODEL ..."
  python - <<PY
from transformers import AutoModel, AutoTokenizer
m = "$MODEL"
AutoTokenizer.from_pretrained(m)
AutoModel.from_pretrained(m)
print(f"[prestage] cached {m}")
PY
done

deactivate || true
rm -rf "$(dirname "$TMP_ENV")"

echo "[prestage] done. Verify:"
du -sh "$HF_HOME"
ls "$HF_HOME"
