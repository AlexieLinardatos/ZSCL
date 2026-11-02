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

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Bits per stored element.  Integer formats also carry one fp16 scale per vector.
PRECISION_BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}
SELECTION_METHODS = ("random", "herding", "coverage", "el2n")


def quantize_dequantize(x: torch.Tensor, precision: str) -> torch.Tensor:
    """
    Round-trip `x` (N, D) through the storage format and return what a reader
    of the buffer would decode.  Integer formats use symmetric per-vector
    absmax scaling.  The result is held in fp32 (fp32 storage) or fp16
    (everything else); nbytes() reports the nominal packed size, so int4 is
    accounted at 4 bits even though it is not bit-packed in host memory.
    """
    if precision == "fp32":
        return x.float()
    if precision == "fp16":
        return x.half()
    qmax = 2 ** (PRECISION_BITS[precision] - 1) - 1
    x = x.float()
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return (q * scale.half().float()).half()


def _rank_within_classes(
    feats: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    num_keep: int,
    text_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Rank every candidate within its class, best first, and return a global
    order whose every prefix is close to class-frequency proportional — the
    same expected class mix as random sampling, so only *which* samples are
    kept changes.  Downsampling at rebalance keeps a prefix of this order.

    herding:  iCaRL greedy mean matching on the stored embeddings.
    coverage: farthest-point (k-center) traversal, seeded at the class mean.
    el2n:     surprisal, ||softmax(100 f T^T) - onehot(y)||_2, highest first,
              scored against the class-name embeddings of the encoding model.

    The greedy methods only rank as far as each class's share of `num_keep`
    (plus one); the rest of the class follows in arbitrary order and is never
    kept, since budgets only shrink after a task is added.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = F.normalize(feats.float(), dim=-1).to(device)
    labels = labels.to(device)
    total = len(labels)
    keys = torch.empty(total)
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        x = feats[idx]
        n = len(idx)
        k_rank = min(n, math.ceil(num_keep * n / total) + 1)
        if method == "el2n":
            p = (100.0 * x @ text_emb.to(device).T).softmax(dim=-1)
            p[:, c] -= 1.0
            order = p.norm(dim=-1).argsort(descending=True).cpu()
        elif method == "herding":
            mu = x.mean(dim=0)
            running = torch.zeros_like(mu)
            avail = torch.ones(n, dtype=torch.bool, device=device)
            order = []
            for k in range(1, k_rank + 1):
                gap = ((mu * k - running).unsqueeze(0) - x).norm(dim=-1)
                gap[~avail] = float("inf")
                i = int(gap.argmin())
                order.append(i)
                avail[i] = False
                running += x[i]
            order = _complete(order, n)
        elif method == "coverage":
            mu = F.normalize(x.mean(dim=0), dim=0)
            first = int((x @ mu).argmax())
            dist = (1 - x @ x[first]).clamp_min(0)
            dist[first] = -1
            order = [first]
            for _ in range(k_rank - 1):
                i = int(dist.argmax())
                order.append(i)
                dist = torch.minimum(dist, (1 - x @ x[i]).clamp_min(0))
                dist[order] = -1
            order = _complete(order, n)
        else:
            raise ValueError(f"Unknown selection method '{method}'.")
        # Rank r of n maps to key (r + 0.5) / n, so sorting all keys interleaves
        # classes in proportion to their size.
        keys[idx.cpu()[order]] = (torch.arange(n, dtype=torch.float) + 0.5) / n
    return keys.argsort()


def _complete(ranked: List[int], n: int) -> torch.Tensor:
    """Append the unranked indices of 0..n-1 after a partial ranking."""
    seen = set(ranked)
    return torch.tensor(ranked + [i for i in range(n) if i not in seen])


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

    def __init__(self, total_budget: int, precision: str = "fp16", selection: str = "random"):
        if precision not in PRECISION_BITS:
            raise ValueError(f"Unknown precision '{precision}'.")
        if selection not in SELECTION_METHODS:
            raise ValueError(f"Unknown selection '{selection}'.")
        self.total_budget = total_budget
        self.precision = precision
        self.selection = selection
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
        # Selection needs every candidate encoded; random only needs the keep set.
        if self.selection == "random":
            indices = random.sample(range(n), num_samples)
        else:
            indices = list(range(n))

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
            feats.append(emb.float().cpu())
            labels.append(batch_labels.cpu().long())

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        if self.selection != "random":
            text_emb = None
            if self.selection == "el2n":
                import clip.clip as clip
                tokens = clip.tokenize([template(c) for c in classnames]).cuda()
                text_emb = F.normalize(model(None, tokens).float(), dim=-1).cpu()
            # Rank on full-precision features; quantization only affects storage.
            order = _rank_within_classes(
                feats, labels, self.selection, num_samples, text_emb
            )
            keep = order[:num_samples]
            feats, labels = feats[keep], labels[keep]

        if was_training:
            model.train()

        self.memory[task_id] = {
            "features": quantize_dequantize(feats, self.precision),
            "labels": labels,
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
        if self.selection == "random":
            keep = torch.tensor(random.sample(range(n), task_budget))
        else:
            # Stored best-first in class-proportional order (_rank_within_classes),
            # so the prefix is the selection at the smaller budget.
            keep = torch.arange(task_budget)
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

        adapted = quantize_dequantize(adapted.cpu(), self.precision)
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
        """
        Nominal bytes of stored feature data at self.precision (labels excluded,
        they are noise).  Integer formats add one fp16 scale per vector.
        """
        bits = PRECISION_BITS[self.precision]
        total = 0
        for v in self.memory.values():
            n, d = v["features"].shape
            total += n * d * bits // 8
            if self.precision.startswith("int"):
                total += n * 2
        return total

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
