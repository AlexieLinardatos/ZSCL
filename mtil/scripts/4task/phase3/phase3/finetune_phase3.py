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
from src.models import multi_teacher_merge as mtm
from src.replay_buffer import ReplayBuffer
from .trainer_phase3 import custom_finetune_phase3


# ---------------------------------------------------------------------------
# Multi-teacher merge helpers (NS3) — outer-loop side: delta save after task.
# ---------------------------------------------------------------------------

_THETA0_CACHE = None  # zero-shot CLIP state dict cached across tasks


def _get_theta0_state(args):
    """Lazy-load + cache zero-shot CLIP state dict (CPU fp32)."""
    global _THETA0_CACHE
    if _THETA0_CACHE is None:
        model0, _, _ = clip.load(args.model, jit=False)
        _THETA0_CACHE = {
            k: v.detach().to("cpu", torch.float32).clone()
            for k, v in model0.state_dict().items()
        }
        del model0
        print(f"[MultiTeacherMerge] Outer loop cached θ_0 "
              f"({len(_THETA0_CACHE)} tensors).")
    return _THETA0_CACHE


def _maybe_save_delta(args, task_idx, task_name, task_names):
    """After a task finishes, compute and save δ_t = θ_t − θ_{t-1} to disk.

    For task 0, θ_{t-1} = θ_0 (zero-shot CLIP).
    For task t>0, θ_{t-1} is loaded from {prev_task}.pth on disk.
    Safe to call repeatedly — skips if the δ file already exists.
    """
    if not getattr(args, "use_multi_teacher_merge", False):
        return

    curr_ckpt = os.path.join(args.save, f"{task_name}.pth")
    if not os.path.exists(curr_ckpt):
        print(f"[MultiTeacherMerge] WARN: {curr_ckpt} missing — cannot save δ_{task_idx}.")
        return

    delta_p = mtm.delta_path(args.save, task_idx, task_name)
    if os.path.exists(delta_p):
        print(f"[MultiTeacherMerge] δ_{task_idx} ({task_name}) already on disk — skipping.")
        return

    curr_state = mtm._load_state_dict_from_ckpt(curr_ckpt)
    if task_idx == 0:
        prev_state = _get_theta0_state(args)
    else:
        prev_name = task_names[task_idx - 1]
        prev_ckpt = os.path.join(args.save, f"{prev_name}.pth")
        if not os.path.exists(prev_ckpt):
            print(f"[MultiTeacherMerge] WARN: {prev_ckpt} missing — cannot save δ_{task_idx}.")
            return
        prev_state = mtm._load_state_dict_from_ckpt(prev_ckpt)

    mtm.compute_and_save_delta(
        prev_state, curr_state, args.save, task_idx, task_name,
        dtype=getattr(args, "merge_dtype", "fp16"),
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
    if getattr(args, "use_multi_teacher_merge", False):
        mtm.ensure_merge_dirs(args.save)

    replay_buffer = ReplayBuffer(total_budget=args.replay_budget)

    # ------------------------------------------------------------------
    # Resume detection
    # ------------------------------------------------------------------
    completed_tasks = _load_completed_tasks(args.save)
    if completed_tasks:
        print(f"[Phase3] Resuming — previously completed tasks: {completed_tasks}")
        _load_buffer_memory(args.save, replay_buffer)
    else:
        print("[Phase3] Starting fresh (no previous state found).")

    # Resume-safety for multi-teacher merge: backfill any missing δ files for
    # tasks that completed in a previous run before --use_multi_teacher_merge
    # was enabled. Signatures are NOT backfilled (they need θ_0 + that task's
    # dataloader, which is set up by the trainer) — they'll be regenerated on
    # next eligible task or absorbed into equal-weights fallback.
    if getattr(args, "use_multi_teacher_merge", False) and completed_tasks:
        on_disk = {n for (_, n, _) in mtm.list_available_deltas(args.save)}
        for idx, name in enumerate(completed_tasks):
            if name not in on_disk:
                print(f"[MultiTeacherMerge] Backfilling missing δ_{idx} ({name}).")
                _maybe_save_delta(args, idx, name, completed_tasks)

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
        args_task._task_idx = task_idx

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

        # Persist trajectory-aware merge artifact for this task (δ_t).
        # Signature s_t was saved by the trainer at the start of this task.
        _maybe_save_delta(args, task_idx, task_name, task_names)

        # Save per-task accuracy snapshot before metrics CSVs get cleared
        eval_ds = args.eval_datasets if isinstance(args.eval_datasets, list) else (args.eval_datasets.split(",") if args.eval_datasets else [])
        _save_task_summary(args.save, task_idx, task_name, eval_ds)

    print(f"\n[Phase3] Finished all {len(task_names)} tasks.")
    print(f"Final checkpoint: {os.path.join(args.save, task_names[-1] + '.pth')}")
