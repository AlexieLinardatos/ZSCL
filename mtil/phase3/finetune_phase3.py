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
import glob
import os

import torch
import clip.clip as clip

from src import datasets, templates
from src.replay_buffer import ReplayBuffer
from .trainer_phase3 import custom_finetune_phase3


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
        replay_buffer.rebalance()

        print(f"[Phase3 outer loop] Buffer after task {task_idx + 1}:")
        print(replay_buffer)

        _save_buffer_state(args.save, replay_buffer, task_name)

    print(f"\n[Phase3] Finished all {len(task_names)} tasks.")
    print(f"Final checkpoint: {os.path.join(args.save, task_names[-1] + '.pth')}")
