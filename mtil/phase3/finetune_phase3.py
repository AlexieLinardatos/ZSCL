"""
Phase 3 outer loop: ZSCL + Replay + Replay Teacher Distillation.

Mirrors the Phase 2 outer loop (finetune_multi_task_replay in
src/models/finetune_replay.py) exactly, but calls custom_finetune_phase3
instead of custom_finetune.

Phase 2 files are NOT modified.

Resume-safe: completed task names and buffer memory are persisted to disk
after each task, so interrupted jobs can continue from where they left off.
Outputs go to args.save — point this at a phase3/ directory to keep Phase 2
and Phase 3 results separate.
"""

import copy
import csv
import glob
import os

import torch
import clip.clip as clip

from src import datasets, templates
from src.feature_replay_buffer import FeatureReplayBuffer
from src.replay_buffer import ReplayBuffer
from .trainer_phase3 import custom_finetune_phase3


# ---------------------------------------------------------------------------
# Feature-buffer encoding
# ---------------------------------------------------------------------------

def _load_encoder(args, ckpt_path):
    """
    Rebuild the image encoder from a task checkpoint, for the one-time pass
    that fills the feature buffer.

    Goes through load_base_model so LoRA layers are applied before the state
    dict lands, matching however the run was trained.
    """
    from src.models.training import load_base_model

    args_enc = copy.copy(args)
    args_enc.load = ckpt_path
    args_enc.start_iteration = None
    model, _, _, _ = load_base_model(args_enc)
    return model.cuda()


def _log_drift_stats(save_dir, task_idx, task_name, method, anchors, stats):
    """Append one row per task boundary to drift_adaptation.csv."""
    path = os.path.join(save_dir, "drift_adaptation.csv")
    fieldnames = ["task_idx", "task_name", "method", "anchors"] + list(stats.keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"task_idx": task_idx, "task_name": task_name,
               "method": method, "anchors": anchors}
        row.update(stats)
        writer.writerow(row)


def _materialize_anchor_loader(dataset, num_samples, batch_size):
    """
    Build a loader over a fixed, already-decoded slice of the current task.

    Drift is observed by encoding the *same* pixels with two encoders and
    subtracting.  Iterating a train_dataset twice would re-run its random
    augmentation and hand each encoder a different crop, so the images are
    stacked into a tensor once here and both passes read from that.
    """
    n = len(dataset)
    num_samples = min(num_samples, n)
    step = n / num_samples
    indices = [int(i * step) for i in range(num_samples)]

    images, labels = [], []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            img, label = item[0], item[1]
        else:
            img, label = item["images"], item["labels"]
        images.append(img)
        labels.append(int(label))

    tensor_ds = torch.utils.data.TensorDataset(
        torch.stack(images), torch.tensor(labels, dtype=torch.long)
    )
    return torch.utils.data.DataLoader(
        tensor_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )


# ---------------------------------------------------------------------------
# Buffer persistence helpers (identical logic to Phase 2)
# ---------------------------------------------------------------------------

def _buffer_memory_path(save_dir):
    return os.path.join(save_dir, "replay_buffer_memory.pt")


def _completed_tasks_path(save_dir):
    return os.path.join(save_dir, "replay_completed_tasks.txt")


def _save_buffer_state(save_dir, replay_buffer, completed_task_name):
    """Persist buffer memory and append completed task name to disk."""
    torch.save(replay_buffer.memory, _buffer_memory_path(save_dir))
    with open(_completed_tasks_path(save_dir), "a") as f:
        f.write(completed_task_name + "\n")
    print(f"[Phase3 Replay] Buffer state saved after task '{completed_task_name}'.")


def _load_completed_tasks(save_dir):
    """Return list of task names that fully completed in a previous run."""
    path = _completed_tasks_path(save_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_buffer_memory(save_dir, replay_buffer):
    """Reload exemplar tensors into replay_buffer.memory from disk."""
    path = _buffer_memory_path(save_dir)
    if not os.path.exists(path):
        return False
    replay_buffer.memory = torch.load(path, weights_only=False)
    print(
        f"[Phase3 Replay] Reloaded buffer: {len(replay_buffer)} exemplars "
        f"across {len(replay_buffer.memory)} tasks."
    )
    return True


def _clear_task_metrics(save_dir):
    """Clear stale metrics CSVs when restarting a task from scratch."""
    cleared = []
    for f in glob.glob(os.path.join(save_dir, "metrics_*.csv")):
        os.remove(f)
        cleared.append(os.path.basename(f))
    if cleared:
        print(f"[Phase3] Cleared stale metrics: {', '.join(cleared)}")


def _save_task_summary(save_dir, task_idx, task_name, eval_datasets):
    """Read final-iteration accuracy from each metrics CSV and append to task_summary.csv.

    This file is NOT cleared between tasks, so it accumulates the full
    accuracy matrix needed for computing Avg and Transfer metrics.
    """
    summary_path = os.path.join(save_dir, "task_summary.csv")

    # Collect final accuracy for each eval dataset
    row = {"task_idx": task_idx, "task_name": task_name}
    for ds in eval_datasets:
        csv_path = os.path.join(save_dir, f"metrics_{ds}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row[ds] = rows[-1]["top1"]
            else:
                row[ds] = ""
        else:
            row[ds] = ""

    # Compute average across all eval datasets that have values
    vals = [float(row[ds]) for ds in eval_datasets if row[ds]]
    row["avg"] = f"{sum(vals) / len(vals):.4f}" if vals else ""

    # Write header if file doesn't exist, then append row
    fieldnames = ["task_idx", "task_name"] + eval_datasets + ["avg"]
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Phase3] Task summary saved → {summary_path} (after '{task_name}')")


# ---------------------------------------------------------------------------
# Phase 3 outer loop
# ---------------------------------------------------------------------------

def finetune_multi_task_phase3(args):
    """
    Phase 3 outer loop: train on each task sequentially with a replay
    buffer and replay teacher distillation.

    Identical structure to finetune_multi_task_replay (Phase 2) except
    it delegates each task's training to custom_finetune_phase3.

    Args:
        args: Parsed CLI + Phase 3 arguments.  Key fields consumed here:
              dataset_order, replay_budget, replay_batch_size,
              replay_loss_weight, save, load, model, data_location,
              batch_size, batch_size_eval, template,
              + all Phase 3 flags set by apply_phase3_args().
    """
    task_names = args.dataset_order
    if not task_names:
        raise ValueError("--dataset_order must specify at least one dataset name.")

    os.makedirs(args.save, exist_ok=True)

    replay_storage = getattr(args, "replay_storage", "pixel")
    if replay_storage == "feature":
        replay_buffer = FeatureReplayBuffer(total_budget=args.replay_budget)
    else:
        replay_buffer = ReplayBuffer(total_budget=args.replay_budget)
    print(f"[Phase3] Replay storage mode: {replay_storage}")

    # ------------------------------------------------------------------
    # Resume detection
    # ------------------------------------------------------------------
    completed_tasks = _load_completed_tasks(args.save)
    if completed_tasks:
        print(f"[Phase3] Resuming — previously completed tasks: {completed_tasks}")
        _load_buffer_memory(args.save, replay_buffer)
    else:
        print("[Phase3] Starting fresh (no previous state found).")

    _, train_preprocess, _ = clip.load(args.model, jit=False)
    initial_load = args.load

    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*60}")
        print(f"[Phase3 outer loop] Task {task_idx + 1}/{len(task_names)}: {task_name}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # Skip already-completed tasks but rebuild buffer metadata
        # ------------------------------------------------------------------
        if task_name in completed_tasks:
            print(
                f"[Phase3] '{task_name}' already completed — skipping training, "
                f"rebuilding buffer metadata."
            )
            dataset_class = getattr(datasets, task_name)
            task_dataset_obj = dataset_class(
                train_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                batch_size_eval=args.batch_size_eval,
            )
            task_template = (
                getattr(templates, args.template)[0]
                if args.template is not None
                else task_dataset_obj.template
            )
            if task_idx in replay_buffer.memory:
                replay_buffer.task_info[task_idx] = {
                    "classnames": task_dataset_obj.classnames,
                    "template": task_template,
                }
            continue

        # ------------------------------------------------------------------
        # Build per-task args
        # ------------------------------------------------------------------
        args_task = copy.copy(args)
        args_task.train_dataset = task_name

        # Per-task iteration override
        task_iters = getattr(args, "task_iterations", {})
        if task_iters and task_name in task_iters:
            args_task.iterations = task_iters[task_name]
            print(f"[Phase3] Per-task iterations for '{task_name}': {args_task.iterations}")

        task_ckpt = os.path.join(args.save, f"{task_name}.pth")
        if os.path.exists(task_ckpt):
            saved_iter = torch.load(task_ckpt, weights_only=False)["iteration"]
            print(
                f"[Phase3] Found partial checkpoint for '{task_name}' "
                f"at iteration {saved_iter}. Resuming from there."
            )
            args_task.load = task_ckpt
            args_task.start_iteration = saved_iter
        elif task_idx == 0:
            args_task.load = initial_load
            args_task.start_iteration = 0
        else:
            prev_task = task_names[task_idx - 1]
            args_task.load = os.path.join(args.save, f"{prev_task}.pth")
            args_task.start_iteration = 0

        # ------------------------------------------------------------------
        # Train this task (Phase 3 trainer)
        # ------------------------------------------------------------------

        # Clear stale metrics from a previous crashed run of this task
        if args_task.start_iteration == 0:
            _clear_task_metrics(args.save)

        current_replay = replay_buffer if task_idx > 0 else None
        if current_replay is not None:
            print(
                f"[Phase3 outer loop] Buffer entering task {task_idx + 1}: "
                f"{len(current_replay)} exemplars"
            )

        custom_finetune_phase3(args_task, replay_buffer=current_replay)

        # ------------------------------------------------------------------
        # Update replay buffer with exemplars from the just-trained task
        # ------------------------------------------------------------------
        dataset_class = getattr(datasets, task_name)
        task_dataset_obj = dataset_class(
            train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        task_template = (
            getattr(templates, args.template)[0]
            if args.template is not None
            else task_dataset_obj.template
        )

        if replay_storage == "feature":
            # Encode this task's exemplars with the checkpoint that just
            # finished training on it.  They are never re-encoded afterwards,
            # so they go stale as the encoder keeps moving through later tasks.
            encoder = _load_encoder(args, os.path.join(args.save, f"{task_name}.pth"))

            # Before adding this task, move everything already in the buffer
            # onto the new encoder's manifold.  Runs first so text anchors are
            # built from previous tasks' class names only — this task's own
            # features are fresh and need no correction.
            feature_adapt = getattr(args, "feature_adapt", "none")
            if feature_adapt != "none" and len(replay_buffer) > 0:
                prev_ckpt = os.path.join(args.save, f"{task_names[task_idx - 1]}.pth")
                anchors = getattr(args, "feature_adapt_anchors", "both")
                print(f"[Phase3] Adapting {len(replay_buffer)} stored features: "
                      f"method={feature_adapt}  anchors={anchors}")

                old_encoder = _load_encoder(args, prev_ckpt)
                old_encoder.eval()
                encoder.eval()

                anchor_loader = None
                if anchors in ("image", "both"):
                    anchor_loader = _materialize_anchor_loader(
                        task_dataset_obj.train_dataset,
                        getattr(args, "feature_adapt_samples", 2000),
                        getattr(args, "replay_encode_batch_size", 64),
                    )

                stats = replay_buffer.adapt(
                    method=feature_adapt,
                    old_model=old_encoder,
                    new_model=encoder,
                    current_loader=anchor_loader,
                    anchors=anchors,
                    k=getattr(args, "feature_adapt_k", 20),
                    alpha=getattr(args, "feature_adapt_alpha", 0.85),
                    iters=getattr(args, "feature_adapt_iters", 30),
                    sigma=getattr(args, "feature_adapt_sigma", None),
                    steps=getattr(args, "feature_adapt_steps", 500),
                )
                print("[Phase3] Drift: " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in stats.items()
                ))
                _log_drift_stats(args.save, task_idx, task_name, feature_adapt,
                                 anchors, stats)

                del old_encoder, anchor_loader
                torch.cuda.empty_cache()

            replay_buffer.add_task(
                task_id=task_idx,
                dataset=task_dataset_obj.train_dataset,
                num_samples=args.replay_budget,
                classnames=task_dataset_obj.classnames,
                template=task_template,
                model=encoder,
                batch_size=getattr(args, "replay_encode_batch_size", 64),
            )
            del encoder
            torch.cuda.empty_cache()
        else:
            replay_buffer.add_task(
                task_id=task_idx,
                dataset=task_dataset_obj.train_dataset,
                num_samples=args.replay_budget,
                classnames=task_dataset_obj.classnames,
                template=task_template,
            )
        class_counts = {
            tid: len(replay_buffer.task_info[tid]["classnames"])
            for tid in replay_buffer.memory
            if tid in replay_buffer.task_info
        }
        if getattr(args, 'no_proportional_replay', False):
            replay_buffer.rebalance()
        else:
            replay_buffer.rebalance_proportional(class_counts)

        print(f"[Phase3 outer loop] Buffer after task {task_idx + 1}:")
        print(replay_buffer)

        _save_buffer_state(args.save, replay_buffer, task_name)

        # Save per-task accuracy snapshot before metrics CSVs get cleared
        eval_ds = args.eval_datasets if isinstance(args.eval_datasets, list) else (args.eval_datasets.split(",") if args.eval_datasets else [])
        _save_task_summary(args.save, task_idx, task_name, eval_ds)

    print(f"\n[Phase3] Finished all {len(task_names)} tasks.")
    print(f"Final checkpoint: {os.path.join(args.save, task_names[-1] + '.pth')}")
