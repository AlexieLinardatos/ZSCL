import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class DistillMixStats:
    public_count: int = 0
    replay_count: int = 0
    total_count: int = 0


class ReplayMemory:
    """Task-indexed replay memory for distillation sampling."""

    def __init__(self):
        self.task_to_examples: Dict[str, List[Tuple[torch.Tensor, int]]] = {}

    def task_names(self) -> List[str]:
        return sorted(self.task_to_examples.keys())

    def size_per_task(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.task_to_examples.items()}

    def total_size(self) -> int:
        return sum(len(v) for v in self.task_to_examples.values())

    def past_task_names(self, current_task: Optional[str] = None) -> List[str]:
        names = self.task_names()
        if current_task is None:
            return names
        return [n for n in names if n != current_task]

    def has_replay(self, current_task: Optional[str] = None) -> bool:
        return any(len(self.task_to_examples[t]) > 0 for t in self.past_task_names(current_task))

    @staticmethod
    def _extract_images_labels(batch):
        if isinstance(batch, dict):
            images = batch["images"]
            labels = batch["labels"]
        else:
            images, labels = batch
        return images, labels

    def add_task_examples(
        self,
        task_name: str,
        data_loader,
        memory_per_task: int,
        store_strategy: str = "fixed_per_task",
    ) -> int:
        if memory_per_task <= 0:
            self.task_to_examples[task_name] = []
            return 0

        examples: List[Tuple[torch.Tensor, int]] = []

        if store_strategy == "reservoir":
            seen = 0
            for batch in data_loader:
                images, labels = self._extract_images_labels(batch)
                bs = images.shape[0]
                for idx in range(bs):
                    seen += 1
                    sample = (images[idx].detach().cpu(), int(labels[idx]))
                    if len(examples) < memory_per_task:
                        examples.append(sample)
                    else:
                        j = int(torch.randint(0, seen, (1,)).item())
                        if j < memory_per_task:
                            examples[j] = sample
        else:
            # fixed_per_task: keep first K seen examples from shuffled task loader.
            for batch in data_loader:
                images, labels = self._extract_images_labels(batch)
                bs = images.shape[0]
                for idx in range(bs):
                    examples.append((images[idx].detach().cpu(), int(labels[idx])))
                    if len(examples) >= memory_per_task:
                        break
                if len(examples) >= memory_per_task:
                    break

        self.task_to_examples[task_name] = examples
        return len(examples)

    def sample_replay_images(
        self,
        count: int,
        current_task: Optional[str],
        strategy: str = "uniform_tasks",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, device=device)

        candidate_tasks = [t for t in self.past_task_names(current_task) if len(self.task_to_examples[t]) > 0]
        if not candidate_tasks:
            return torch.empty(0, device=device)

        chosen: List[torch.Tensor] = []
        if strategy == "proportional_examples":
            all_examples: List[Tuple[str, int]] = []
            for task in candidate_tasks:
                all_examples.extend([(task, idx) for idx in range(len(self.task_to_examples[task]))])
            if not all_examples:
                return torch.empty(0, device=device)
            for _ in range(count):
                pos = int(torch.randint(0, len(all_examples), (1,)).item())
                task, idx = all_examples[pos]
                chosen.append(self.task_to_examples[task][idx][0])
        else:
            # uniform_tasks: each draw picks a task uniformly first.
            for _ in range(count):
                task_idx = int(torch.randint(0, len(candidate_tasks), (1,)).item())
                task = candidate_tasks[task_idx]
                ex_idx = int(torch.randint(0, len(self.task_to_examples[task]), (1,)).item())
                chosen.append(self.task_to_examples[task][ex_idx][0])

        if not chosen:
            return torch.empty(0, device=device)
        batch = torch.stack(chosen, dim=0)
        if device is not None:
            batch = batch.to(device=device, non_blocking=True)
        return batch

    def save(self, path: str) -> None:
        payload = {
            "task_to_examples": self.task_to_examples,
        }
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "task_to_examples" not in payload:
            return False
        self.task_to_examples = payload["task_to_examples"]
        return True


class MixedDistillationSampler:
    """Build mixed-source distillation batches from public + replay."""

    def __init__(
        self,
        replay_memory: ReplayMemory,
        replay_mix_alpha: float,
        distill_buffer_size: int,
        replay_sampling_strategy: str = "uniform_tasks",
        public_min_ratio: float = 0.0,
    ):
        self.replay_memory = replay_memory
        self.replay_mix_alpha = replay_mix_alpha
        self.distill_buffer_size = distill_buffer_size
        self.replay_sampling_strategy = replay_sampling_strategy
        self.public_min_ratio = public_min_ratio

    @staticmethod
    def _extract_images(batch):
        if isinstance(batch, dict):
            return batch["images"]
        return batch[0]

    def _next_public_images(self, ref_iter, ref_loader, count: int) -> Tuple[torch.Tensor, any]:
        if count <= 0:
            return torch.empty(0), ref_iter

        chunks: List[torch.Tensor] = []
        collected = 0
        while collected < count:
            try:
                batch = next(ref_iter)
            except StopIteration:
                ref_iter = iter(ref_loader)
                batch = next(ref_iter)
            images = self._extract_images(batch)
            need = count - collected
            take = min(images.shape[0], need)
            chunks.append(images[:take])
            collected += take

        return torch.cat(chunks, dim=0), ref_iter

    def _counts(self, current_task: Optional[str]) -> Tuple[int, int]:
        total = max(1, self.distill_buffer_size)
        replay_count = int(math.floor(max(0.0, min(1.0, self.replay_mix_alpha)) * total))
        public_count = total - replay_count

        min_public = int(math.ceil(max(0.0, min(1.0, self.public_min_ratio)) * total))
        if min_public > public_count:
            public_count = min_public
            replay_count = total - public_count

        if not self.replay_memory.has_replay(current_task):
            replay_count = 0
            public_count = total

        return public_count, replay_count

    def sample_mixed_batch(
        self,
        ref_iter,
        ref_loader,
        current_task: Optional[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, any, DistillMixStats]:
        public_count, replay_count = self._counts(current_task)

        public_images, ref_iter = self._next_public_images(ref_iter, ref_loader, public_count)
        replay_images = self.replay_memory.sample_replay_images(
            replay_count,
            current_task=current_task,
            strategy=self.replay_sampling_strategy,
            device=None,
        )

        if replay_images.numel() == 0:
            mixed = public_images
            replay_count = 0
        elif public_images.numel() == 0:
            mixed = replay_images
            public_count = 0
        else:
            mixed = torch.cat([public_images, replay_images], dim=0)

        mixed = mixed.to(device=device, non_blocking=True)
        stats = DistillMixStats(
            public_count=public_count,
            replay_count=replay_count,
            total_count=public_count + replay_count,
        )
        return mixed, ref_iter, stats
