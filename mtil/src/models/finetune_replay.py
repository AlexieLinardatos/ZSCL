"""
Multi-task outer loop for ZSCL + Replay (Phase 2).

Trains on a sequence of datasets in order, maintaining a fixed-budget replay
buffer between tasks.  After each task, a random subset of that task's training
images is added to the buffer and the buffer is rebalanced.

Resume-safe: after each task completes, the buffer memory and a list of
completed task names are saved to disk.  On restart the outer loop detects
which tasks are done, reloads the buffer, and picks up from the right task.

Usage via CLI:
    python -m src.main \\
        --method ZSCL \\
        --train-mode whole \\
        --dataset_order MNIST,Flowers,EuroSAT,DTD \\
        --use_replay \\
        --replay_budget 500 \\
        --replay_batch_size 32 \\
        --replay_loss_weight 1.0 \\
        --iterations 1000 \\
        --lr 1e-5 \\
        --save ckpt/replay_run \\
        --ref-dataset ImageNet \\
        --ref-sentences conceptual_captions \\
        --image_loss --text_loss \\
        --eval-datasets MNIST,Flowers,EuroSAT,DTD \\
        --eval-interval 500
"""

import copy
import glob
import os

import torch
import clip.clip as clip

from .. import datasets, templates
from ..replay_buffer import ReplayBuffer
from .training import custom_finetune


# ---------------------------------------------------------------------------
# Helpers for persisting / reloading the replay buffer across job restarts
# ---------------------------------------------------------------------------

def _buffer_memory_path(save_dir):
    return os.path.join(save_dir, "replay_buffer_memory.pt")

def _completed_tasks_path(save_dir):
    return os.path.join(save_dir, "replay_completed_tasks.txt")


def _save_buffer_state(save_dir, replay_buffer, completed_task_name):
    """Persist buffer memory and append to the completed-tasks list."""
    # Save raw image tensors + labels (templates are re-derived from datasets on load)
    torch.save(replay_buffer.memory, _buffer_memory_path(save_dir))
    with open(_completed_tasks_path(save_dir), "a") as f:
        f.write(completed_task_name + "\n")
    print(f"[Replay] Buffer state saved to {save_dir} after task '{completed_task_name}'.")


def _load_completed_tasks(save_dir):
    """Return ordered list of task names that fully completed in a previous run."""
    path = _completed_tasks_path(save_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_buffer_memory(save_dir, replay_buffer):
    """Reload raw exemplar tensors into replay_buffer.memory from disk."""
    path = _buffer_memory_path(save_dir)
    if not os.path.exists(path):
        return False
    replay_buffer.memory = torch.load(path, weights_only=False)
    print(f"[Replay] Reloaded buffer memory: {len(replay_buffer)} exemplars "
          f"across {len(replay_buffer.memory)} tasks.")
    return True


def _clear_task_metrics(save_dir):
    """Clear stale metrics CSVs when restarting a task from scratch.

    Prevents cross-run contamination where old metrics from a crashed run
    mix with new metrics from a fresh restart.
    """
    cleared = []
    for f in glob.glob(os.path.join(save_dir, "metrics_*.csv")):
        os.remove(f)
        cleared.append(os.path.basename(f))
    if cleared:
        print(f"[Replay] Cleared stale metrics: {', '.join(cleared)}")


# ---------------------------------------------------------------------------
# Main outer loop
# ---------------------------------------------------------------------------

def finetune_multi_task_replay(args):
    """
    Outer loop: train on each task in `args.dataset_order` sequentially,
    with a growing replay buffer carrying exemplars from all previous tasks.

    Resume-safe: if the job is killed and restarted, it detects completed tasks
    from disk, reloads the replay buffer, and resumes from the interrupted task.

    After the full sequence, per-task checkpoints are saved as
    `{args.save}/{task_name}.pth`.

    Args:
        args: Parsed CLI arguments.  Key fields consumed here:
              - dataset_order     (list[str])  Task dataset names in order.
              - replay_budget     (int)        Total exemplar budget.
              - replay_batch_size (int)        Replay batch size per step.
              - replay_loss_weight (float)     Weight of replay CE loss.
              - save              (str)        Directory for checkpoints.
              - load              (str|None)   Path to initial checkpoint (task 0 only).
    """
    task_names = args.dataset_order
    if not task_names:
        raise ValueError("--dataset_order must specify at least one dataset name.")

    os.makedirs(args.save, exist_ok=True)

    replay_buffer = ReplayBuffer(total_budget=args.replay_budget)

    # -----------------------------------------------------------------------
    # Resume detection: find which tasks already finished in a previous run.
    # -----------------------------------------------------------------------
    completed_tasks = _load_completed_tasks(args.save)
    if completed_tasks:
        print(f"[Replay] Resuming — previously completed tasks: {completed_tasks}")
        _load_buffer_memory(args.save, replay_buffer)
    else:
        print("[Replay] Starting fresh (no previous state found).")

    # Load CLIP preprocess once — fixed for a given model variant.
    _, train_preprocess, _ = clip.load(args.model, jit=False)

    initial_load = args.load  # original --load path (or None)

    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*60}")
        print(f"[Replay outer loop] Task {task_idx + 1}/{len(task_names)}: {task_name}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # If this task already completed in a previous run, skip training
        # but rebuild its task_info (classnames + template) in the buffer
        # so replay works correctly for subsequent tasks.
        # ------------------------------------------------------------------
        if task_name in completed_tasks:
            print(f"[Replay] Task '{task_name}' already completed — skipping training, "
                  f"rebuilding buffer metadata.")
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
            # Re-attach template/classnames to the already-loaded memory entry.
            if task_idx in replay_buffer.memory:
                replay_buffer.task_info[task_idx] = {
                    "classnames": task_dataset_obj.classnames,
                    "template": task_template,
                }
            continue

        # ------------------------------------------------------------------
        # Build task-specific args.
        # ------------------------------------------------------------------
        args_task = copy.copy(args)
        args_task.train_dataset = task_name

        # Determine checkpoint to load.
        task_ckpt = os.path.join(args.save, f"{task_name}.pth")
        if os.path.exists(task_ckpt):
            # Partial save from a previous interrupted run of THIS task.
            saved_iter = torch.load(task_ckpt, weights_only=False)["iteration"]
            print(f"[Replay] Found partial checkpoint for '{task_name}' "
                  f"at iteration {saved_iter}. Resuming from there.")
            args_task.load = task_ckpt
            args_task.start_iteration = saved_iter
        elif task_idx == 0:
            args_task.load = initial_load  # may be None (start from pretrained CLIP)
            args_task.start_iteration = 0
        else:
            prev_task = task_names[task_idx - 1]
            args_task.load = os.path.join(args.save, f"{prev_task}.pth")
            args_task.start_iteration = 0

        # ------------------------------------------------------------------
        # Train this task (with replay from previous tasks if buffer has data).
        # ------------------------------------------------------------------

        # Clear stale metrics from a previous crashed run of this task
        if args_task.start_iteration == 0:
            _clear_task_metrics(args.save)

        current_replay = replay_buffer if task_idx > 0 else None
        if current_replay is not None:
            print(f"[Replay outer loop] Buffer entering task {task_idx + 1}: "
                  f"{len(current_replay)} exemplars")

        custom_finetune(args_task, replay_buffer=current_replay)

        # ------------------------------------------------------------------
        # Update the replay buffer with exemplars from the just-trained task.
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

        print(f"[Replay outer loop] Buffer after task {task_idx + 1}:")
        print(replay_buffer)

        # ------------------------------------------------------------------
        # Persist buffer state so the job can resume if killed.
        # ------------------------------------------------------------------
        _save_buffer_state(args.save, replay_buffer, task_name)

    print(f"\n[Replay outer loop] Finished all {len(task_names)} tasks.")
    print(f"Final checkpoint: {os.path.join(args.save, task_names[-1] + '.pth')}")
