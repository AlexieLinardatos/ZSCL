"""
Feature replay buffer for ZSCL + Replay.

Stores the CLIP image encoder's final 512-d embedding per exemplar instead of
the preprocessed image tensor.  A 224x224x3 fp32 tensor is 602 KB; a 512-d
fp16 embedding is 1 KB, so an 11k-exemplar budget drops from ~6.6 GB to ~11 MB.

The tradeoff is the image-encoder gradient: the replay CE can no longer
backpropagate through the vision tower, only into the text embeddings it is
scored against.  See replay_storage_proposal.md section 3 — that image-side
gradient is already supplied by L_zscl and L_RD, while the text-side gradient
on previous tasks' class names is supplied by nothing else.

Features are encoded once, at the task boundary, with that task's final
checkpoint.  They are never refreshed, so they go stale as the encoder keeps
moving; correcting for that drift is what the adaptation methods layered on
top of this buffer are for.

The public API mirrors ReplayBuffer (src/replay_buffer.py) so the training
loop can hold either one, with the exception of add_task, which additionally
needs the encoder to run.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class FlatFeatureReplayDataset(Dataset):
    """A flat Dataset of (feature, label, task_id) tuples from the feature buffer."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor):
        self.features = features
        self.labels = labels
        self.task_ids = task_ids

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        return self.features[idx], int(self.labels[idx]), int(self.task_ids[idx])


class FeatureReplayBuffer:
    """
    Fixed-budget episodic replay buffer storing final image embeddings.

    Design:
    - memory[task_id] = {"features": fp16 (N, D) CPU tensor, "labels": int64 (N,)}
      Features are L2-normalised at storage time.
    - Random sampling from the task's training set, same as ReplayBuffer.
    - Re-balances (downsamples older tasks) when a new task is added, so the
      total never exceeds `total_budget`.
    - Stores per-task classnames and template callable for loss computation at
      replay time, exactly as ReplayBuffer does.
    """

    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.memory: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_info: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    @torch.no_grad()
    def add_task(
        self,
        task_id: int,
        dataset: Dataset,
        num_samples: int,
        classnames: List[str],
        template: Callable[[str], str],
        model: torch.nn.Module,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        """
        Randomly sample `num_samples` examples from `dataset`, encode them with
        `model`, and store the resulting embeddings.

        Args:
            task_id:     Integer identifier for this task (0, 1, 2, ...).
            dataset:     A torch Dataset returning (image_tensor, label) per item.
            num_samples: Max samples to store. Clamped to dataset size.
            classnames:  List of class name strings for this task.
            template:    Callable mapping classname -> text prompt string.
            model:       Encoder used to embed the exemplars.  Should be the
                         checkpoint just trained on this task.  Left in whatever
                         train/eval mode it arrived in is not safe, so this
                         switches it to eval and restores the previous mode.
            batch_size:  Encoding batch size.
            num_workers: DataLoader workers for the encoding pass.
        """
        n = len(dataset)
        num_samples = min(num_samples, n)
        indices = random.sample(range(n), num_samples)

        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        was_training = model.training
        model.eval()

        feats: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for item in loader:
            # Handle both tuple/list and dict-style items (e.g. ImageNet)
            if isinstance(item, (tuple, list)):
                images, batch_labels = item[0], item[1]
            else:
                images, batch_labels = item["images"], item["labels"]

            emb = model(images.cuda(), None)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.half().cpu())
            labels.append(batch_labels.cpu().long())

        if was_training:
            model.train()

        self.memory[task_id] = {
            "features": torch.cat(feats, dim=0),
            "labels": torch.cat(labels, dim=0),
        }
        self.task_info[task_id] = {
            "classnames": classnames,
            "template": template,
        }

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    def _subsample(self, task_id: int, task_budget: int) -> None:
        entry = self.memory[task_id]
        n = entry["features"].shape[0]
        if n <= task_budget:
            return
        keep = torch.tensor(random.sample(range(n), task_budget))
        self.memory[task_id] = {
            "features": entry["features"][keep],
            "labels": entry["labels"][keep],
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
            self._subsample(task_id, per_task_budget)

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
            self._subsample(task_id, task_budget)

    # ------------------------------------------------------------------
    # Drift adaptation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def adapt(
        self,
        method: str,
        old_model: torch.nn.Module,
        new_model: torch.nn.Module,
        current_loader=None,
        anchors: str = "both",
        device: str = "cuda",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Move every stored feature from the old encoder's manifold onto the new
        one's, using drift observed at the current task boundary.

        Call this at the end of a task, *before* add_task for that task: the new
        task's own features are encoded with the current model and need no
        correction, while everything already in the buffer does.  Keeping the
        order that way also means text anchors are built from the class names of
        previous tasks only, which is what the correction is being fitted for.

        All tasks are corrected as one matrix rather than task by task, so the
        graph-based estimator can route drift between tasks — a stored feature
        can inherit a correction through a neighbour in another task's slice.

        Args:
            method:         one of feature_adaptation.ADAPT_METHODS.
            old_model:      encoder that produced the stored features (start of
                            this task).
            new_model:      encoder just trained (end of this task).
            current_loader: deterministic (shuffle=False) loader over current-
                            task images, for image anchors. Required unless
                            anchors == "text".
            anchors:        "image", "text", or "both".
            **kwargs:       estimator hyperparameters, passed through.

        Returns:
            Dict of drift statistics for logging (see drift_report).
        """
        from .feature_adaptation import (
            apply_drift,
            collect_image_anchors,
            collect_text_anchors,
            drift_report,
            estimate_drift,
        )

        if method == "none" or not self.memory:
            return {}

        # Stack every task's features into one matrix, remembering the slices.
        task_ids = sorted(self.memory)
        slices, offset = {}, 0
        chunks = []
        for tid in task_ids:
            feats = self.memory[tid]["features"]
            chunks.append(feats.float())
            slices[tid] = (offset, offset + feats.shape[0])
            offset += feats.shape[0]
        stored = torch.cat(chunks, dim=0).to(device)

        # Collect anchors.
        src_parts, delta_parts = [], []
        if anchors in ("image", "both"):
            if current_loader is None:
                raise ValueError(
                    f"anchors='{anchors}' needs current_loader, got None."
                )
            s, d = collect_image_anchors(old_model, new_model, current_loader, device)
            src_parts.append(s)
            delta_parts.append(d)
        if anchors in ("text", "both"):
            s, d = collect_text_anchors(old_model, new_model, self.task_info, device)
            src_parts.append(s)
            delta_parts.append(d)
        if not src_parts:
            raise ValueError(f"Unknown anchor source '{anchors}'.")

        anchor_src = torch.cat(src_parts, dim=0).to(device)
        anchor_delta = torch.cat(delta_parts, dim=0).to(device)

        delta = estimate_drift(method, stored, anchor_src, anchor_delta, **kwargs)
        adapted = apply_drift(stored, delta)
        stats = drift_report(stored, adapted, anchor_src, anchor_delta)

        adapted = adapted.half().cpu()
        for tid in task_ids:
            lo, hi = slices[tid]
            self.memory[tid]["features"] = adapted[lo:hi]

        return stats

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def get_combined_dataset(self) -> FlatFeatureReplayDataset:
        """Return a flat Dataset of all stored (feature, label, task_id) tuples."""
        all_feats, all_labels, all_tids = [], [], []
        for task_id, entry in self.memory.items():
            n = entry["features"].shape[0]
            all_feats.append(entry["features"])
            all_labels.append(entry["labels"])
            all_tids.append(torch.full((n,), task_id, dtype=torch.long))

        if not all_feats:
            dim = 0
            return FlatFeatureReplayDataset(
                torch.empty(0, dim), torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        return FlatFeatureReplayDataset(
            torch.cat(all_feats, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_tids, dim=0),
        )

    def get_task_info(self, task_id: int) -> Dict:
        """Return classnames and template callable for the given task."""
        return self.task_info[task_id]

    def feature_dim(self) -> Optional[int]:
        """Embedding dimension of the stored features, or None if empty."""
        for entry in self.memory.values():
            return entry["features"].shape[1]
        return None

    def __len__(self) -> int:
        return sum(v["features"].shape[0] for v in self.memory.values())

    def nbytes(self) -> int:
        """Total bytes of stored feature data (labels excluded, they are noise)."""
        return sum(
            v["features"].numel() * v["features"].element_size()
            for v in self.memory.values()
        )

    def __repr__(self) -> str:
        dim = self.feature_dim()
        lines = [
            f"FeatureReplayBuffer(budget={self.total_budget}, "
            f"tasks={len(self.memory)}, total_stored={len(self)}, "
            f"dim={dim}, size={self.nbytes() / 1e6:.1f} MB)"
        ]
        for tid, entry in sorted(self.memory.items()):
            info = self.task_info.get(tid, {})
            n_classes = len(info.get("classnames", []))
            lines.append(
                f"  Task {tid}: {entry['features'].shape[0]} exemplars, "
                f"{n_classes} classes"
            )
        return "\n".join(lines)
