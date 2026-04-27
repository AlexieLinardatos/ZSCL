"""
Per-class ImageNet zero-shot evaluation.

Loads a checkpoint and runs ImageNet zero-shot evaluation, outputting
per-class accuracy (1000 values) to a JSON file. Run this twice — once for
the v4 checkpoint and once for the ZSCL baseline checkpoint — then pass both
JSONs to analyze_imagenet_groups.py.

Usage (from mtil/ directory on Nibi):

    python scripts/perclass_imagenet_eval.py \
        --checkpoint ckpt/11task/phase3_no_lora_v4/SUN397.pth \
        --data-location /path/to/datasets \
        --output results/perclass_ours.json

    python scripts/perclass_imagenet_eval.py \
        --checkpoint ckpt/11task/zscl_only/SUN397.pth \
        --data-location /path/to/datasets \
        --output results/perclass_zscl.json
"""

import argparse
import json
import os
import sys

import clip
import torch
from tqdm import tqdm

_MTIL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MTIL_DIR not in sys.path:
    sys.path.insert(0, _MTIL_DIR)

from src import utils
from src.datasets.imagenet import ImageNet
from src.datasets.imagenet_classnames import get_classnames
from src.models.evaluation import zeroshot_classifier


@torch.no_grad()
def perclass_zeroshot_eval(model, loader, zeroshot_weights):
    """Like zeroshot_eval but tracks correct/total per class."""
    n_classes = zeroshot_weights.shape[1]
    correct_per_class = torch.zeros(n_classes, dtype=torch.long)
    total_per_class  = torch.zeros(n_classes, dtype=torch.long)

    for data in tqdm(loader, desc="Evaluating"):
        if isinstance(data, dict):
            images = data["images"].cuda()
            targets = data["labels"].cuda()
        else:
            images, targets = data[0].cuda(), data[1].cuda()

        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * feats @ zeroshot_weights        # (B, 1000)

        preds = logits.argmax(dim=1).cpu()
        targets_cpu = targets.cpu()

        for cls in range(n_classes):
            mask = (targets_cpu == cls)
            total_per_class[cls]   += mask.sum()
            correct_per_class[cls] += (preds[mask] == cls).sum()

    # Avoid division by zero for absent classes
    acc = torch.where(
        total_per_class > 0,
        correct_per_class.float() / total_per_class.float(),
        torch.zeros(n_classes),
    )
    return acc.tolist(), total_per_class.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint (state_dict format).")
    parser.add_argument("--data-location", required=True,
                        help="Root directory containing the ImageNet folder.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path.")
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading CLIP {args.model}...")
    model, _, val_preprocess = clip.load(args.model, jit=False)
    model = model.cuda().eval()

    print(f"Loading checkpoint: {args.checkpoint}")
    utils.torch_load(model, args.checkpoint)
    model.eval()

    print("Building ImageNet dataset...")
    dataset = ImageNet(
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Building zero-shot classifier...")
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )

    print("Running per-class evaluation...")
    acc_per_class, total_per_class = perclass_zeroshot_eval(
        model, dataset.test_loader, zeroshot_weights
    )

    mean_acc = sum(a for a, n in zip(acc_per_class, total_per_class) if n > 0) / \
               sum(1 for n in total_per_class if n > 0)
    print(f"Mean Top-1 accuracy: {mean_acc * 100:.2f}%")

    classnames = get_classnames("openai")
    result = {
        "checkpoint": args.checkpoint,
        "mean_top1": mean_acc * 100,
        "classes": [
            {"id": i, "name": classnames[i],
             "acc": acc_per_class[i], "n": total_per_class[i]}
            for i in range(len(classnames))
        ],
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved per-class results to {args.output}")


if __name__ == "__main__":
    main()
