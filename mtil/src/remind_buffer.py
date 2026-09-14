"""
REMIND-style mid-network replay for CLIP.

Hayes et al., "REMIND Your Neural Network to Prevent Catastrophic Forgetting"
(ECCV 2020), replays compressed *intermediate* activations rather than images or
final embeddings. Three ideas carry over to CLIP, and the third is what makes it
a genuinely different point on the curve from feature replay:

  1. Store activations from the middle of the network, not the end. The replayed
     sample still flows through the upper layers, so it keeps a real gradient
     path into the image tower — the thing pure feature replay gives up.

  2. Compress them with product quantization. A raw 197x768 token grid is 300 KB,
     worse than the image; at m=32 it is 197x32 = 6.3 KB, still ~95x cheaper than
     a 602 KB pixel exemplar.

  3. Freeze everything below the storage layer after the first task. This is the
     part usually skipped in summaries, and it is the load-bearing one: stored
     activations cannot go stale, because the layers that produced them never
     move again. Drift is eliminated by construction rather than corrected after
     the fact, which is the opposite bet from the feature-adaptation ladder.

That third point is also REMIND's cost. Freezing the bottom six blocks of a
CLIP ViT is a real constraint on plasticity, and on MTIL — where tasks range
from MNIST to SUN397 — it may bite harder than it does on a single ImageNet
stream. Whether the traded plasticity is worth the eliminated drift is exactly
the question this arm answers.

Storage per exemplar, ViT-B/16 at layer 6, m=32:  197 tokens x 32 B = 6.3 KB
    vs. pixel replay   602 KB   (96x smaller)
    vs. feature replay   1 KB   (6x larger, but with gradient and no drift)
"""

from typing import Callable, Dict, List, Optional, Tuple

import random

import torch
from torch.utils.data import DataLoader, Dataset

from .product_quantizer import ProductQuantizer


# ---------------------------------------------------------------------------
# Split forward pass through CLIP's visual transformer
# ---------------------------------------------------------------------------

def encode_to_layer(visual, x: torch.Tensor, layer: int) -> torch.Tensor:
    """
    Run the patch embedding and the first `layer` residual blocks.

    Mirrors VisualTransformer.forward up to the split point. Returns tokens in
    NLD order (batch, tokens, width) — the natural layout for storage, converted
    to the LND the blocks want only inside this function.
    """
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    for block in visual.transformer.resblocks[:layer]:
        x = block(x)
    return x.permute(1, 0, 2)  # LND -> NLD


def encode_from_layer(
    visual,
    tokens: torch.Tensor,
    layer: int,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """
    Run the remaining blocks, the final norm and the projection.

    `tokens` is NLD, as returned by encode_to_layer. The output matches what
    visual.forward would have produced for the same image, up to the
    quantization error introduced in between.
    """
    x = tokens.permute(1, 0, 2)  # NLD -> LND
    for block in visual.transformer.resblocks[layer:]:
        if use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    x = x.permute(1, 0, 2)

    x = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        x = x @ visual.proj
    return x


def freeze_below_layer(model, layer: int) -> int:
    """
    Freeze the patch embedding and every residual block below `layer`.

    REMIND's guarantee that stored activations stay valid holds only if nothing
    underneath them trains, so this must be applied on every task after the one
    that fitted the codebook. Returns the number of frozen parameter tensors.
    """
    visual = model.module.visual if hasattr(model, "module") else model.visual

    frozen = 0
    for module in (visual.conv1, visual.ln_pre):
        for p in module.parameters():
            p.requires_grad_(False)
            frozen += 1
    for name in ("class_embedding", "positional_embedding"):
        param = getattr(visual, name, None)
        if param is not None:
            param.requires_grad_(False)
            frozen += 1
    for block in visual.transformer.resblocks[:layer]:
        for p in block.parameters():
            p.requires_grad_(False)
            frozen += 1
    return frozen


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class FlatRemindDataset(Dataset):
    """Flat Dataset of (codes, label, task_id); codes are uint8 (tokens, m)."""

    def __init__(self, codes: torch.Tensor, labels: torch.Tensor,
                 task_ids: torch.Tensor):
        self.codes = codes
        self.labels = labels
        self.task_ids = task_ids

    def __len__(self) -> int:
        return self.codes.shape[0]

    def __getitem__(self, idx):
        return self.codes[idx], int(self.labels[idx]), int(self.task_ids[idx])


class RemindReplayBuffer:
    """
    Fixed-budget buffer of PQ-compressed mid-network token grids.

    memory[task_id] = {"codes": uint8 (N, L, m), "labels": int64 (N,)}

    Mirrors ReplayBuffer / FeatureReplayBuffer's public surface so the trainer
    can hold any of the three.
    """

    def __init__(self, total_budget: int, layer: int = 6, m: int = 32):
        self.total_budget = total_budget
        self.layer = layer
        self.pq = ProductQuantizer(m=m)
        self.memory: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_info: Dict[int, Dict] = {}

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
        batch_size: int = 32,
        num_workers: int = 0,
        fit_pq: Optional[bool] = None,
    ) -> None:
        """
        Encode this task's exemplars to layer-`self.layer` tokens and store them
        as PQ codes.

        The codebook is fitted on the first task only (or when `fit_pq` forces
        it). Refitting later would change what every previously stored code
        means, so it is deliberately a one-time event — the same reason REMIND
        freezes the layers below the split.
        """
        n = len(dataset)
        num_samples = min(num_samples, n)
        indices = random.sample(range(n), num_samples)
        loader = DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

        was_training = model.training
        model.eval()
        visual = model.module.visual if hasattr(model, "module") else model.visual

        token_batches, label_batches = [], []
        for item in loader:
            if isinstance(item, (tuple, list)):
                images, labels = item[0], item[1]
            else:
                images, labels = item["images"], item["labels"]
            tokens = encode_to_layer(visual, images.cuda(), self.layer)
            token_batches.append(tokens.float().cpu())
            label_batches.append(labels.cpu().long())

        if was_training:
            model.train()

        tokens = torch.cat(token_batches, dim=0)        # (N, L, D)
        labels = torch.cat(label_batches, dim=0)
        n_stored, n_tokens, dim = tokens.shape

        should_fit = fit_pq if fit_pq is not None else (self.pq.codebooks is None)
        flat = tokens.reshape(-1, dim)
        if should_fit:
            print(f"[REMIND] Fitting PQ on {flat.shape[0]} token vectors "
                  f"(dim={dim}, m={self.pq.m})")
            self.pq.fit(flat.cuda())
            err = self.pq.reconstruction_error(flat[:20000].cuda())
            print(f"[REMIND] {self.pq}  mean relative reconstruction error: "
                  f"{err:.4f}")

        codes = self.pq.encode(flat.cuda()).cpu()
        self.memory[task_id] = {
            "codes": codes.reshape(n_stored, n_tokens, self.pq.m),
            "labels": labels,
        }
        self.task_info[task_id] = {"classnames": classnames, "template": template}

    # ------------------------------------------------------------------

    def _subsample(self, task_id: int, task_budget: int) -> None:
        entry = self.memory[task_id]
        n = entry["codes"].shape[0]
        if n <= task_budget:
            return
        keep = torch.tensor(random.sample(range(n), task_budget))
        self.memory[task_id] = {"codes": entry["codes"][keep],
                                "labels": entry["labels"][keep]}

    def rebalance(self) -> None:
        if not self.memory:
            return
        per_task = max(1, self.total_budget // len(self.memory))
        for task_id in self.memory:
            self._subsample(task_id, per_task)

    def rebalance_proportional(self, class_counts: Dict[int, int]) -> None:
        if not self.memory:
            return
        total_classes = sum(class_counts.get(t, 1) for t in self.memory)
        for task_id in self.memory:
            budget = max(1, round(
                self.total_budget * class_counts.get(task_id, 1) / total_classes
            ))
            self._subsample(task_id, budget)

    # ------------------------------------------------------------------

    def get_combined_dataset(self) -> FlatRemindDataset:
        codes, labels, tids = [], [], []
        for task_id, entry in self.memory.items():
            codes.append(entry["codes"])
            labels.append(entry["labels"])
            tids.append(torch.full((entry["codes"].shape[0],), task_id,
                                   dtype=torch.long))
        if not codes:
            return FlatRemindDataset(
                torch.empty(0, 0, 0, dtype=torch.uint8),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        return FlatRemindDataset(torch.cat(codes), torch.cat(labels),
                                 torch.cat(tids))

    def get_task_info(self, task_id: int) -> Dict:
        return self.task_info[task_id]

    @torch.no_grad()
    def decode_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, L, m) uint8 codes -> (B, L, D) reconstructed tokens."""
        b, l, m = codes.shape
        flat = self.pq.decode(codes.reshape(-1, m))
        return flat.reshape(b, l, -1)

    def state_dict(self) -> Dict:
        return {"memory": self.memory, "layer": self.layer,
                "pq": self.pq.state_dict()}

    def load_state_dict(self, state: Dict) -> None:
        self.memory = state["memory"]
        self.layer = state.get("layer", self.layer)
        self.pq.load_state_dict(state["pq"])

    def __len__(self) -> int:
        return sum(v["codes"].shape[0] for v in self.memory.values())

    def nbytes(self) -> int:
        return sum(v["codes"].numel() for v in self.memory.values())

    def __repr__(self) -> str:
        lines = [
            f"RemindReplayBuffer(budget={self.total_budget}, layer={self.layer}, "
            f"m={self.pq.m}, tasks={len(self.memory)}, total_stored={len(self)}, "
            f"size={self.nbytes() / 1e6:.1f} MB)"
        ]
        for tid, entry in sorted(self.memory.items()):
            info = self.task_info.get(tid, {})
            lines.append(
                f"  Task {tid}: {entry['codes'].shape[0]} exemplars, "
                f"{len(info.get('classnames', []))} classes"
            )
        return "\n".join(lines)
