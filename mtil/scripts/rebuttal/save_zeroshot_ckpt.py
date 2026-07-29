#!/usr/bin/env python3
"""
Save the pre-trained CLIP weights as a checkpoint in the format
``src.utils.torch_load`` expects, so that ``python -m src.main --eval-only``
can be pointed at it to measure per-task zero-shot accuracy.

This exists only because ``src.models.evaluation.evaluate`` writes its results
next to ``args.load``, so an eval run needs a checkpoint path even when the
weights are the untouched pre-trained ones.  Nothing in src/ is modified.

Usage (from the mtil/ directory):

    python scripts/rebuttal/save_zeroshot_ckpt.py \
        --out ckpt/rebuttal/zeroshot/zeroshot.pth
"""

import argparse
import os
import sys

_MTIL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MTIL_DIR not in sys.path:
    sys.path.insert(0, _MTIL_DIR)

import torch

import clip.clip as clip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ViT-B/16", help="CLIP backbone (default: ViT-B/16)")
    p.add_argument("--out", default="ckpt/rebuttal/zeroshot/zeroshot.pth")
    args = p.parse_args()

    model, _, _ = clip.load(args.model, jit=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"iteration": 0, "state_dict": model.state_dict()}, args.out)
    print(f"Saved zero-shot {args.model} checkpoint to {args.out}")


if __name__ == "__main__":
    main()
