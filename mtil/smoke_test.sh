#!/bin/bash
# Smoke test: sets up a throwaway venv, installs wandb, runs test_wandb.py
# Run from mtil/: bash smoke_test.sh

set -euo pipefail

module load python/3.11.5

VENV_DIR="/tmp/wandb_smoke_$$"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --quiet wandb

export WANDB_MODE=offline
python test_wandb.py

deactivate
rm -rf "$VENV_DIR"
echo "[smoke_test.sh] Done — venv cleaned up."
