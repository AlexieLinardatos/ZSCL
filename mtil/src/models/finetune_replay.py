"""
Multi-task outer loop for ZSCL + Replay (Phase 2).

Trains on a sequence of datasets in order, maintaining a fixed-budget replay
buffer between tasks.  After each task, a random subset of that task's training
images is added to the buffer and the buffer is rebalanced.

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
        --ref-dataset ImageNetSM \\
        --image_loss --text_loss \\
        --eval-datasets MNIST,Flowers,EuroSAT,DTD \\
        --eval-interval 500
"""

import copy
import os

import clip.clip as clip

from .. import datasets, templates
from ..replay_buffer import ReplayBuffer
from .training import custom_finetune


def finetune_multi_task_replay(args):
    """
    Outer loop: train on each task in `args.dataset_order` sequentially,
    with a growing replay buffer carrying exemplars from all previous tasks.

    After the full sequence, the final model checkpoint (for the last task) is
    the primary output.  Per-task checkpoints are saved as
    `{args.save}/{task_name}.pth`.

    Args:
        args: Parsed CLI arguments.  Key fields consumed here:
              - dataset_order  (list[str])  Task dataset names in order.
              - replay_budget  (int)        Total exemplar budget.
              - replay_batch_size (int)     Replay batch size per step.
              - replay_loss_weight (float)  Weight of replay CE loss.
              - save           (str)        Directory for checkpoints.
              - load           (str|None)   Path to initial checkpoint (task 0 only).
    """
    task_names = args.dataset_order
    if not task_names:
        raise ValueError("--dataset_order must specify at least one dataset name.")

    replay_buffer = ReplayBuffer(total_budget=args.replay_budget)

    # Load CLIP preprocess once — it is fixed for a given model variant.
    _, train_preprocess, _ = clip.load(args.model, jit=False)

    initial_load = args.load  # original --load path (or None)

    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*60}")
        print(f"[Replay outer loop] Task {task_idx + 1}/{len(task_names)}: {task_name}")
        print(f"{'='*60}")

        # Build a task-specific args copy so we don't mutate the original.
        args_task = copy.copy(args)
        args_task.train_dataset = task_name

        # Chain checkpoints: task N loads the saved model from task N-1.
        if task_idx == 0:
            args_task.load = initial_load  # may be None (start from pretrained CLIP)
        else:
            prev_task = task_names[task_idx - 1]
            args_task.load = os.path.join(args.save, f"{prev_task}.pth")

        # Train this task (with replay from previous tasks if buffer is non-empty).
        current_replay = replay_buffer if task_idx > 0 else None
        if current_replay is not None:
            print(f"[Replay outer loop] Buffer entering task {task_idx + 1}: "
                  f"{len(current_replay)} exemplars")

        custom_finetune(args_task, replay_buffer=current_replay)

        # ------------------------------------------------------------------ #
        # Update the replay buffer with exemplars from the just-trained task. #
        # ------------------------------------------------------------------ #
        dataset_class = getattr(datasets, task_name)
        task_dataset_obj = dataset_class(
            train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        # Determine template callable.
        if args.template is not None:
            task_template = getattr(templates, args.template)[0]
        else:
            task_template = task_dataset_obj.template

        # Store up to replay_budget samples; rebalance will trim them.
        replay_buffer.add_task(
            task_id=task_idx,
            dataset=task_dataset_obj.train_dataset,
            num_samples=args.replay_budget,  # generous; rebalance trims this
            classnames=task_dataset_obj.classnames,
            template=task_template,
        )
        replay_buffer.rebalance()

        print(f"[Replay outer loop] Buffer after task {task_idx + 1}:")
        print(replay_buffer)

    print(f"\n[Replay outer loop] Finished all {len(task_names)} tasks.")
    print(f"Final checkpoint: {os.path.join(args.save, task_names[-1] + '.pth')}")
