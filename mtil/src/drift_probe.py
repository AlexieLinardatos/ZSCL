"""
Per-task drift tracking for feature replay.

The gate script (src/measure_drift.py) answers "did the features go stale?"
*after* a run, and only for runs that kept a pixel buffer.  A feature-replay run
throws its images away, so by construction it cannot re-encode anything and the
question becomes unanswerable exactly where it matters most.

This module fixes that by keeping a deliberately tiny **probe set**: a fixed
handful of preprocessed images per task, held purely for measurement and never
replayed into any loss.  At every task boundary the probe is re-encoded with the
current model and compared against the features it had when its task finished.
That yields, for every stored task, a curve of how stale its embeddings have
become as training moved on — measured live, during the run, instead of
reconstructed afterwards.

    cos( f_t(probe_t), f_s(probe_t) )   for every task t and every later task s

Storage.  The probe is diagnostic apparatus, not replay data, and must be
excluded from any storage claim about the buffer.  At the default 64 images per
task it costs ~19 MB/task in fp16 (~210 MB over 11 tasks) — trivial as a
debugging cost, and dwarfed by the pixel buffer it is helping to eliminate.
Set --drift_probe_size 0 to turn it off entirely for a clean storage run.

Why a probe rather than the current task's images: drift observed on the current
task tells you how far the encoder moved *where it is training*, which is the
easy case. What breaks feature replay is drift at the oldest task's manifold,
somewhere the model has not looked in ten tasks. Only a probe from that task can
measure it.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DriftProbe:
    """
    Fixed per-task image sets plus the features they had at storage time.

    probes[task_id] = {
        "images":   fp16 (n, 3, H, W)  preprocessed, exactly as the encoder sees
        "features": fp16 (n, D)        L2-normalised, at that task's boundary
    }
    """

    def __init__(self, per_task: int = 64):
        self.per_task = per_task
        self.probes: Dict[int, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    @torch.no_grad()
    def add_task(
        self,
        task_id: int,
        dataset,
        model: torch.nn.Module,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        """
        Freeze a probe set for `task_id` and record its features under `model`.

        Images are stored as the already-preprocessed tensor rather than raw
        pixels, so re-encoding later is bit-identical apart from the encoder
        itself — no augmentation, no resize, nothing that could masquerade as
        drift.
        """
        if self.per_task <= 0:
            return

        n = min(self.per_task, len(dataset))
        indices = random.sample(range(len(dataset)), n)

        images = []
        for idx in indices:
            item = dataset[idx]
            img = item[0] if isinstance(item, (tuple, list)) else item["images"]
            images.append(img)
        images = torch.stack(images)

        feats = self._encode(model, images, batch_size, device)

        self.probes[task_id] = {
            "images": images.half().cpu(),
            "features": feats.half().cpu(),
        }

    @staticmethod
    @torch.no_grad()
    def _encode(model, images, batch_size, device) -> torch.Tensor:
        was_training = model.training
        model.eval()
        out = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)
            # Stored probes are fp16; match whatever the live model runs in.
            param = next(model.parameters())
            emb = model(batch.to(param.dtype), None)
            out.append(F.normalize(emb.float(), dim=-1).cpu())
        if was_training:
            model.train()
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def measure(
        self,
        model: torch.nn.Module,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> Dict[int, Dict[str, float]]:
        """
        Re-encode every task's probe with `model` and compare to storage time.

        Returns:
            {task_id: {"mean_cos", "p05_cos", "min_cos", "n"}}

        A task's mean_cos is the answer to "how far have this task's stored
        features drifted from what the model would produce for the same images
        right now".  1.0 is no drift.
        """
        results = {}
        for task_id, entry in sorted(self.probes.items()):
            current = self._encode(model, entry["images"].float(),
                                   batch_size, device)
            stored = entry["features"].float()
            cos = (stored * current).sum(dim=-1)
            results[task_id] = {
                "mean_cos": cos.mean().item(),
                "p05_cos": cos.quantile(0.05).item(),
                "min_cos": cos.min().item(),
                "n": int(cos.numel()),
            }
        return results

    @torch.no_grad()
    def refresh(
        self,
        model: torch.nn.Module,
        task_ids: Optional[List[int]] = None,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        """
        Re-baseline stored probe features to `model`.

        Only for runs that adapt stored features: after an adaptation step the
        buffer has been moved onto the new encoder, so leaving the probe at its
        original baseline would report the drift the adaptation just corrected.
        Refreshing makes the measurement 'drift since the last correction'.
        """
        for task_id in (task_ids if task_ids is not None else list(self.probes)):
            entry = self.probes.get(task_id)
            if entry is None:
                continue
            entry["features"] = self._encode(
                model, entry["images"].float(), batch_size, device
            ).half().cpu()

    # ------------------------------------------------------------------
    # Persistence / reporting
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {"per_task": self.per_task, "probes": self.probes}

    def load_state_dict(self, state: Dict) -> None:
        self.per_task = state.get("per_task", self.per_task)
        self.probes = state.get("probes", {})

    def nbytes(self) -> int:
        return sum(
            e["images"].numel() * e["images"].element_size()
            for e in self.probes.values()
        )

    def __len__(self) -> int:
        return sum(e["images"].shape[0] for e in self.probes.values())

    def __repr__(self) -> str:
        return (
            f"DriftProbe(per_task={self.per_task}, tasks={len(self.probes)}, "
            f"images={len(self)}, size={self.nbytes() / 1e6:.1f} MB)"
        )
