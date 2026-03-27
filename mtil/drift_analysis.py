#!/usr/bin/env python3
"""
drift_analysis.py — Measure CLIP visual embedding drift under continual learning.

Hypothesis: CE-only replay causes the student model to drift away from the frozen
zero-shot CLIP's representation geometry. Teacher distillation on replay (Phase 3)
prevents this drift by anchoring the student to the frozen teacher.

Metric: mean cosine drift = mean(1 - cosine_similarity(student_emb, frozen_emb))
        measured on a fixed reference image set at each task checkpoint.

Reference images: CIFAR100 test set (OOD for the 4-task sequence
                  DTD → MNIST → EuroSAT → Flowers; downloads automatically).

Usage (from mtil/ directory):
    python drift_analysis.py
    python drift_analysis.py --n_images 2000 --batch_size 128
    python drift_analysis.py --cpu   # if no GPU

Output:
    drift_analysis.png  — plot for the paper
    drift_analysis.json — raw numbers
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchvision

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip.clip as clip_lib
from src.models.lora import DEFAULT_LORA_TARGETS, inject_lora

# ── Config ─────────────────────────────────────────────────────────────────────
CLIP_MODEL = "ViT-B/16"
CKPT_ROOT = "ckpt"
TASK_ORDER = ["DTD", "MNIST", "EuroSAT", "Flowers"]

# Checkpoint paths (relative to CKPT_ROOT), one per task in sequence order.
# All three conditions use LoRA (detected automatically).
CONDITIONS = {
    "ZSCL only (no replay)": [
        "4task/phase2.1/baseline/DTD_trained/DTD.pth",
        "4task/phase2.1/baseline/DTD_trained/MNIST_trained/MNIST.pth",
        "4task/phase2.1/baseline/DTD_trained/MNIST_trained/EuroSAT_trained/EuroSAT.pth",
        "4task/phase2.1/baseline/DTD_trained/MNIST_trained/EuroSAT_trained/Flowers_trained/Flowers.pth",
    ],
    "CE replay (no teacher distill)": [
        "4task/phase2.1/replay/DTD.pth",
        "4task/phase2.1/replay/MNIST.pth",
        "4task/phase2.1/replay/EuroSAT.pth",
        "4task/phase2.1/replay/Flowers.pth",
    ],
    "Phase 3: replay + teacher distill": [
        "4task/phase3.1/replay_teacher/DTD.pth",
        "4task/phase3.1/replay_teacher/MNIST.pth",
        "4task/phase3.1/replay_teacher/EuroSAT.pth",
        "4task/phase3.1/replay_teacher/Flowers.pth",
    ],
}


def load_visual_encoder(ckpt_path: str, device: str):
    """
    Load the visual encoder from a saved ImageClassifier checkpoint.
    Automatically detects and applies LoRA if the checkpoint contains LoRA weights,
    so the effective weights (base + LoRA delta) are correctly reproduced.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older PyTorch without weights_only argument
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]

    # Strip "image_encoder.model." prefix → raw CLIP model state dict
    prefix = "image_encoder.model."
    clip_state = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

    # Load a fresh CLIP model
    model, _, _ = clip_lib.load(CLIP_MODEL, device="cpu", jit=False)

    # Detect LoRA by presence of lora_ keys
    has_lora = any("lora_" in k for k in clip_state)
    if has_lora:
        # Infer rank from a lora_A tensor (shape: [r, in_features])
        r_key = next(
            k for k in clip_state
            if k.endswith(".lora_A") or k.endswith(".lora_q_A")
        )
        r = clip_state[r_key].shape[0]
        print(f"    LoRA detected (r={r}) — injecting before loading weights.")
        inject_lora(model, DEFAULT_LORA_TARGETS, r=r, alpha=16, dropout=0.0)

    missing, unexpected = model.load_state_dict(clip_state, strict=False)
    # Unexpected keys expected: classification_head.*, logit_scale, etc.
    lora_unexpected = [k for k in unexpected if "lora_" not in k and "transformer" not in k]
    if lora_unexpected:
        print(f"    Unexpected (non-LoRA) keys: {lora_unexpected[:5]}")

    model.eval()
    return model.to(device)


@torch.no_grad()
def get_embeddings(model, images: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    """Run images through model.encode_image in batches; return L2-normalised embeddings."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        embs = model.encode_image(batch).float()
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def cosine_drift(student_embs: torch.Tensor, frozen_embs: torch.Tensor) -> float:
    """mean(1 - cosine_similarity); both tensors must already be L2-normalised."""
    cos_sim = (student_embs * frozen_embs).sum(dim=-1)  # element-wise dot product
    return (1.0 - cos_sim).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=1000,
                        help="Number of CIFAR100 test images to use (default 1000)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device} | N images: {args.n_images} | Batch: {args.batch_size}\n")

    # ── Frozen CLIP anchor ─────────────────────────────────────────────────────
    print("Loading frozen zero-shot CLIP (anchor)...")
    frozen_clip, _, val_preprocess = clip_lib.load(CLIP_MODEL, device=device, jit=False)
    frozen_clip.eval()

    # ── Reference image set ────────────────────────────────────────────────────
    print("Loading CIFAR100 test set (OOD reference — not in 4-task training)...")
    cifar100 = torchvision.datasets.CIFAR100(
        root=os.path.join("data", "cifar100_drift_ref"),
        train=False,
        download=True,
        transform=val_preprocess,
    )
    n = min(args.n_images, len(cifar100))
    subset = torch.utils.data.Subset(cifar100, list(range(n)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False
    )
    all_images = torch.cat([imgs for imgs, _ in loader], dim=0)
    print(f"  {len(all_images)} images ready.\n")

    # ── Frozen embeddings (computed once) ──────────────────────────────────────
    print("Computing frozen CLIP embeddings...")
    frozen_embs = get_embeddings(frozen_clip, all_images, args.batch_size, device)
    sanity = cosine_drift(frozen_embs, frozen_embs)
    print(f"  Sanity check (frozen vs frozen): {sanity:.6f}  (expected ~0.0)\n")

    # ── Drift per condition per task ───────────────────────────────────────────
    results = {}
    task_labels = ["Zero-shot"] + TASK_ORDER

    for condition, ckpt_paths in CONDITIONS.items():
        print(f"{'─'*60}")
        print(f"Condition: {condition}")
        # Zero-shot = drift 0 by definition (student IS the frozen model)
        drifts = [0.0]

        for task_name, rel_path in zip(TASK_ORDER, ckpt_paths):
            full_path = os.path.join(CKPT_ROOT, rel_path)
            if not os.path.exists(full_path):
                print(f"  [{task_name}] MISSING — {full_path}")
                drifts.append(None)
                continue

            print(f"  [{task_name}] {rel_path}")
            student = load_visual_encoder(full_path, device)
            student_embs = get_embeddings(student, all_images, args.batch_size, device)
            drift = cosine_drift(student_embs, frozen_embs)
            drifts.append(drift)
            print(f"    → drift: {drift:.5f}")

            del student
            if device == "cuda":
                torch.cuda.empty_cache()

        results[condition] = drifts
        print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "task_labels": task_labels,
        "conditions": results,
        "n_images": n,
        "reference_dataset": "CIFAR100 test (OOD for the 4-task sequence)",
        "metric": "mean(1 - cosine_similarity) vs frozen zero-shot CLIP",
    }
    with open("drift_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Raw numbers → drift_analysis.json")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n=== DRIFT SUMMARY ===")
    col_w = 30
    header = f"{'Task':<14}" + "".join(f"{c[:col_w-2]:<{col_w}}" for c in results)
    print(header)
    print("─" * len(header))
    for i, task in enumerate(task_labels):
        row = f"{task:<14}"
        for drifts in results.values():
            val = drifts[i] if i < len(drifts) and drifts[i] is not None else "N/A"
            row += f"{(val if isinstance(val, str) else f'{val:.5f}'):<{col_w}}"
        print(row)

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(task_labels)))
        styles = [
            dict(color="#2196F3", marker="s", linestyle="--",  linewidth=2,   markersize=8),
            dict(color="#FF9800", marker="o", linestyle="-.",   linewidth=2,   markersize=8),
            dict(color="#4CAF50", marker="D", linestyle="-",    linewidth=2.5, markersize=9),
        ]

        for (cond, drifts), style in zip(results.items(), styles):
            valid_x = [xi for xi, d in zip(x, drifts) if d is not None]
            valid_d = [d for d in drifts if d is not None]
            ax.plot(valid_x, valid_d, label=cond, **style)

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=15, ha="right", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Task Checkpoint", fontsize=12)
        ax.set_ylabel("Mean Cosine Drift from Frozen CLIP", fontsize=12)
        ax.set_title(
            "CLIP Embedding Drift During Continual Learning\n"
            f"(CIFAR100 test reference, N={n}, out-of-distribution)",
            fontsize=12,
        )
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-0.002)
        plt.tight_layout()
        plt.savefig("drift_analysis.png", dpi=150, bbox_inches="tight")
        print("Plot → drift_analysis.png")

    except ImportError:
        print("matplotlib not found — install with: pip install matplotlib")


if __name__ == "__main__":
    main()
