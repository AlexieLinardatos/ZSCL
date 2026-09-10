"""
Representation drift measurement — the go/no-go gate for feature replay.

Feature replay stores an image embedding once, at the task boundary, and never
refreshes it.  Its known failure mode is that the encoder keeps moving, so the
stored vector describes a manifold the model no longer produces.  This script
measures how far it actually moves in our setup.

Method.  Take a saved *pixel* replay buffer (replay_buffer_memory.pt, written by
the Phase 2/3 outer loop) and re-encode every exemplar with each per-task
checkpoint from the same run.  Task t's exemplars would have been stored by the
checkpoint at the end of task t, so the number feature replay cares about is

    cos( f_t(x), f_final(x) )   for x in task t's slice of the buffer

reported per task, alongside the full trajectory cos(f_t(x), f_s(x)) for every
later s so drift accumulation is visible rather than just its endpoint.

Interpretation (replay_storage_proposal.md section 3):
    mean cos > ~0.95  ->  stored features stay valid; feature replay is licensed
    mean cos ~ 0.7    ->  drift dominates; pure feature replay will not hold

Run it on a ZSCL+RD run and on a Seq-FT run to get the contrast: the claim worth
making is that ZSCL's distillation terms hold the image encoder nearly still,
which is *why* feature replay is affordable here and not in general.

Usage (from mtil/):

    python -m src.measure_drift \
        --buffer   ckpt/11task/phase3_no_lora_v4/replay_buffer_memory.pt \
        --ckpt-dir ckpt/11task/phase3_no_lora_v4 \
        --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
        --label zscl_rd \
        --out drift_zscl_rd.csv

Only the checkpoints that exist on disk are used, so this also works on a
partially finished run.
"""

import argparse
import csv
import os
from typing import Dict, List

import torch

import clip.clip as clip
from src import utils
from src.models.lora import apply_lora_if_enabled


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer", required=True,
                   help="Path to replay_buffer_memory.pt (a pixel buffer).")
    p.add_argument("--ckpt-dir", required=True,
                   help="Directory holding per-task checkpoints named <task>.pth.")
    p.add_argument("--dataset_order", required=True,
                   help="Comma-separated task order, same as the training run. "
                        "Position in this list is the task_id used in the buffer.")
    p.add_argument("--model", default="ViT-B/16",
                   help="CLIP variant the run used.")
    p.add_argument("--label", default="run",
                   help="Name for this run in the output (e.g. zscl_rd, seqft).")
    p.add_argument("--out", default="drift.csv", help="Output CSV path.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-per-task", type=int, default=None,
                   help="Subsample each task's slice to at most this many "
                        "exemplars. Use to get a quick answer on a big buffer.")
    # LoRA passthrough, for runs trained with adapters.
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target_modules", default=None)
    p.add_argument("--lora_bias", default="none")
    return p.parse_args()


def load_buffer(path: str, max_per_task) -> Dict[int, torch.Tensor]:
    """
    Load a pixel replay buffer and return {task_id: stacked image tensor}.

    Rejects a feature buffer with a clear message: its entries are already
    embeddings, so there is nothing left to re-encode.
    """
    memory = torch.load(path, weights_only=False)

    images_per_task: Dict[int, torch.Tensor] = {}
    for task_id, entry in sorted(memory.items()):
        if isinstance(entry, dict):
            raise ValueError(
                f"{path} looks like a FeatureReplayBuffer (task {task_id} holds "
                f"keys {sorted(entry)}). Drift can only be measured from a pixel "
                f"buffer, since it requires re-encoding the original images."
            )
        imgs = [item[0] for item in entry]
        if max_per_task is not None and len(imgs) > max_per_task:
            step = len(imgs) / max_per_task
            imgs = [imgs[int(i * step)] for i in range(max_per_task)]
        images_per_task[int(task_id)] = torch.stack(imgs)

    return images_per_task


def build_model(args, ckpt_path: str):
    model, _, _ = clip.load(args.model, jit=False, pretrained=True)
    model = apply_lora_if_enabled(args, model)
    utils.torch_load(model, ckpt_path)
    model = model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def encode_all(model, images_per_task, batch_size, device) -> Dict[int, torch.Tensor]:
    """Encode every task's exemplars, returning L2-normalised fp16 features."""
    out = {}
    for task_id, images in images_per_task.items():
        feats = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)
            emb = model(batch, None)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.half().cpu())
        out[task_id] = torch.cat(feats, dim=0)
    return out


def main():
    args = parse_args()
    task_names: List[str] = [t.strip() for t in args.dataset_order.split(",") if t.strip()]

    # Which checkpoints actually exist, in task order.
    ckpts = []
    for idx, name in enumerate(task_names):
        path = os.path.join(args.ckpt_dir, f"{name}.pth")
        if os.path.exists(path):
            ckpts.append((idx, name, path))
        else:
            print(f"[drift] Missing checkpoint for task {idx} ({name}) — skipping.")
    if len(ckpts) < 2:
        raise SystemExit("[drift] Need at least two checkpoints to measure drift.")

    images_per_task = load_buffer(args.buffer, args.max_per_task)
    print(f"[drift] Buffer: {len(images_per_task)} tasks, "
          f"{sum(v.shape[0] for v in images_per_task.values())} exemplars")
    print(f"[drift] Checkpoints: {[n for _, n, _ in ckpts]}")

    # features[ckpt_idx][task_id] -> (N_t, D)
    features: Dict[int, Dict[int, torch.Tensor]] = {}
    for idx, name, path in ckpts:
        print(f"[drift] Encoding with checkpoint '{name}' (task {idx})...")
        model = build_model(args, path)
        features[idx] = encode_all(model, images_per_task, args.batch_size, args.device)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    final_idx = ckpts[-1][0]
    rows = []

    # Full trajectory: task t's exemplars as seen by its storage checkpoint t
    # versus every later checkpoint s.
    for store_idx, store_name, _ in ckpts:
        if store_idx not in images_per_task:
            continue
        stored = features[store_idx][store_idx].float()
        for eval_idx, eval_name, _ in ckpts:
            if eval_idx < store_idx:
                continue
            current = features[eval_idx][store_idx].float()
            cos = (stored * current).sum(dim=-1)
            rows.append({
                "run": args.label,
                "task_id": store_idx,
                "task_name": store_name,
                "stored_at": store_name,
                "evaluated_at": eval_name,
                "tasks_elapsed": eval_idx - store_idx,
                "n": cos.numel(),
                "mean_cos": f"{cos.mean().item():.4f}",
                "p05_cos": f"{cos.quantile(0.05).item():.4f}",
                "min_cos": f"{cos.min().item():.4f}",
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[drift] Full trajectory written to {args.out} ({len(rows)} rows)")

    # Headline table: storage checkpoint versus the final checkpoint.
    print(f"\n{'task':<16}{'stored at':<16}{'mean cos':>10}{'p05':>8}{'min':>8}")
    print("-" * 58)
    endpoint = [r for r in rows if r["evaluated_at"] == ckpts[-1][1]]
    for r in endpoint:
        print(f"{r['task_name']:<16}{r['stored_at']:<16}"
              f"{float(r['mean_cos']):>10.4f}{float(r['p05_cos']):>8.4f}"
              f"{float(r['min_cos']):>8.4f}")
    if endpoint:
        overall = sum(float(r["mean_cos"]) for r in endpoint) / len(endpoint)
        print("-" * 58)
        print(f"{'MEAN':<32}{overall:>10.4f}")
        verdict = (
            "licensed (>0.95)" if overall > 0.95
            else "marginal (0.85-0.95), adaptation needed" if overall > 0.85
            else "drift dominates (<0.85), pure feature replay will not hold"
        )
        print(f"\n[drift] Verdict for '{args.label}': {verdict}")


if __name__ == "__main__":
    main()
