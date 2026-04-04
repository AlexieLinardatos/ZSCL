"""
Replay buffer for ZSCL + Replay (Phase 2).

Stores raw (image_tensor, label) exemplars per task with a fixed total budget.
After each new task is added, the buffer is rebalanced so all tasks get an
equal share of the budget.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class FlatReplayDataset(Dataset):
    """A flat Dataset of (image_tensor, label, task_id) tuples from the replay buffer."""

    def __init__(self, samples: List[Tuple[torch.Tensor, int, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        return self.samples[idx]


class ReplayBuffer:
    """
    Fixed-budget episodic replay buffer for MTIL (Multi-Task Incremental Learning).

    Design:
    - Equal per-task allocation: budget // num_tasks_seen images per task.
    - Random sampling (random subset, no reservoir sampling needed for static datasets).
    - Stores raw (image_tensor, label) tuples on CPU.
    - Re-balances (downsamples older tasks) when a new task is added,
      so total buffer size never exceeds `total_budget`.
    - Stores per-task classnames and template callable for loss computation at replay time.
    """

    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.memory: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
        self.task_info: Dict[int, Dict] = {}

    def add_task(
        self,
        task_id: int,
        dataset: Dataset,
        num_samples: int,
        classnames: List[str],
        template: Callable[[str], str],
    ) -> None:
        """
        Randomly sample `num_samples` examples from `dataset` and store them.

        Args:
            task_id:    Integer identifier for this task (0, 1, 2, ...).
            dataset:    A torch Dataset returning (image_tensor, label) per item.
            num_samples: Max samples to store. Clamped to dataset size.
            classnames: List of class name strings for this task.
            template:   Callable mapping classname -> text prompt string.
        """
        n = len(dataset)
        num_samples = min(num_samples, n)
        indices = random.sample(range(n), num_samples)

        samples = []
        for idx in indices:
            item = dataset[idx]
            # Handle both tuple/list and dict-style items (e.g. ImageNet)
            if isinstance(item, (tuple, list)):
                img, label = item[0], item[1]
            else:
                img, label = item["images"], item["labels"]

            if torch.is_tensor(img):
                img = img.cpu()
            samples.append((img, int(label)))

        self.memory[task_id] = samples
        self.task_info[task_id] = {
            "classnames": classnames,
            "template": template,
        }

    def rebalance(self) -> None:
        """
        Downsample all tasks equally so total stored exemplars <= total_budget.
        Call this after every `add_task`.
        """
        num_tasks = len(self.memory)
        if num_tasks == 0:
            return

        per_task_budget = max(1, self.total_budget // num_tasks)

        for task_id in self.memory:
            current = self.memory[task_id]
            if len(current) > per_task_budget:
                self.memory[task_id] = random.sample(current, per_task_budget)

    def rebalance_proportional(self, class_counts: Dict[int, int]) -> None:
        """
        Downsample tasks proportionally to their number of classes.

        Allocates more exemplars to tasks with more classes so that each class
        receives approximately the same number of exemplars across all tasks.
        Tasks absent from class_counts fall back to a budget of 1.

        Args:
            class_counts: mapping from task_id -> number of classes in that task.
        """
        if not self.memory:
            return

        total_classes = sum(class_counts.get(tid, 1) for tid in self.memory)
        for task_id in self.memory:
            n_classes = class_counts.get(task_id, 1)
            task_budget = max(1, round(self.total_budget * n_classes / total_classes))
            current = self.memory[task_id]
            if len(current) > task_budget:
                self.memory[task_id] = random.sample(current, task_budget)

    def get_combined_dataset(self) -> FlatReplayDataset:
        """Return a flat Dataset of all stored (img, label, task_id) tuples."""
        all_samples: List[Tuple[torch.Tensor, int, int]] = []
        for task_id, samples in self.memory.items():
            for img, label in samples:
                all_samples.append((img, label, task_id))
        return FlatReplayDataset(all_samples)

    def get_task_info(self, task_id: int) -> Dict:
        """Return classnames and template callable for the given task."""
        return self.task_info[task_id]

    def __len__(self) -> int:
        return sum(len(v) for v in self.memory.values())

    def __repr__(self) -> str:
        lines = [
            f"ReplayBuffer(budget={self.total_budget}, "
            f"tasks={len(self.memory)}, total_stored={len(self)})"
        ]
        for tid, samples in sorted(self.memory.items()):
            info = self.task_info.get(tid, {})
            n_classes = len(info.get("classnames", []))
            lines.append(
                f"  Task {tid}: {len(samples)} exemplars, {n_classes} classes"
            )
        return "\n".join(lines)
