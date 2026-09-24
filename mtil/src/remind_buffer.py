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

    def __init__(self, total_budget: int, layer: int = 6, m: int = 32,
                 per_task_codebook: bool = True, whiten: bool = True):
        """
        Args:
            per_task_codebook: fit a fresh codebook per task instead of sharing
                one fitted on task 0. REMIND shares a codebook because its whole
                stream is one domain (ImageNet); MTIL spans eleven, so a codebook
                fitted on Aircraft is being asked to quantize MNIST. Each task
                decodes with the codebook it was encoded by, so nothing a later
                fit does can change the meaning of an earlier code.
            whiten: standardise each channel over the task's token population
                before quantizing, and undo it on decode. Transformer residual
                streams carry a few outlier channels an order of magnitude
                larger than the rest; contiguous PQ splitting assumes variance
                is spread evenly, so those channels swamp the handful of
                subspaces they land in and leave the others starved. Note this
                is per-CHANNEL over the population, not per-token LayerNorm —
                LN rescales each token but leaves the relative scale between
                channels exactly as it was, which is the part that hurts.
        """
        self.total_budget = total_budget
        self.layer = layer
        self.m = m
        self.per_task_codebook = per_task_codebook
        self.whiten = whiten
        # Shared codebook, used only when per_task_codebook is False.
        self.pq = ProductQuantizer(m=m)
        self.pqs: Dict[int, ProductQuantizer] = {}
        # task_id -> (mu, sd), each (D,) fp16. Empty when whiten is False.
        self.norm: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.memory: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_info: Dict[int, Dict] = {}

    # ------------------------------------------------------------------

    def _pq_for(self, task_id: int) -> ProductQuantizer:
        if self.per_task_codebook:
            return self.pqs[task_id]
        return self.pq

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

        Under per_task_codebook (the default) every task fits its own codebook,
        which is kept beside its codes; `fit_pq` is then ignored. Under the
        shared-codebook setting it is fitted on the first task only, or when
        `fit_pq` forces it — refitting a *shared* codebook would silently change
        the meaning of every code already stored.
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

        flat = tokens.reshape(-1, dim)                    # (N*L, D), original space

        # Channel standardisation. Statistics come from this task's own tokens,
        # so each task is whitened against the distribution it actually has --
        # the point of the exercise on a benchmark whose domains do not share one.
        if self.whiten:
            # Round to the precision they are stored at before using them, so
            # the forward transform is the exact inverse of what decode applies.
            mu = flat.mean(dim=0).half().cpu()
            sd = flat.std(dim=0).clamp_min(1e-6).half().cpu()
            self.norm[task_id] = (mu, sd)
            flat_q = (flat - mu.float()) / sd.float()
        else:
            self.norm.pop(task_id, None)
            flat_q = flat

        if self.per_task_codebook:
            pq = ProductQuantizer(m=self.m)
            should_fit = True
        else:
            pq = self.pq
            should_fit = fit_pq if fit_pq is not None else (pq.codebooks is None)

        if should_fit:
            scope = f"task {task_id}" if self.per_task_codebook else "shared"
            print(f"[REMIND] Fitting PQ ({scope}) on {flat_q.shape[0]} token "
                  f"vectors (dim={dim}, m={self.m}, whiten={self.whiten})")
            pq.fit(flat_q.cuda())

            # Two errors, because they answer different questions. The quantised
            # one is how well the codebook covers what it was fitted on. The
            # original-space one is what the network actually receives back, and
            # is the number comparable to runs made before whitening existed.
            probe = flat_q[:20000].cuda()
            err_q = pq.reconstruction_error(probe)
            if self.whiten:
                recon = (pq.decode(pq.encode(probe))
                         * sd.float().cuda() + mu.float().cuda())
                ref = flat[:20000].cuda()
                err_o = ((recon - ref).norm(dim=-1)
                         / ref.norm(dim=-1).clamp_min(1e-8)).mean().item()
            else:
                err_o = err_q
            print(f"[REMIND] {pq}  mean relative reconstruction error: "
                  f"{err_o:.4f} (original space), {err_q:.4f} (quantised space)")

        if self.per_task_codebook:
            self.pqs[task_id] = pq

        codes = pq.encode(flat_q.cuda()).cpu()
        self.memory[task_id] = {
            "codes": codes.reshape(n_stored, n_tokens, self.m),
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
    def decode_tokens(self, codes: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        (B, L, m) uint8 codes -> (B, L, D) reconstructed tokens.

        `task_id` selects the codebook and whitening statistics the codes were
        written with; callers already loop per task to build that task's text
        embeddings, so it is on hand.
        """
        b, l, m = codes.shape
        flat = self._pq_for(task_id).decode(codes.reshape(-1, m))
        if task_id in self.norm:
            mu, sd = self.norm[task_id]
            flat = flat * sd.to(flat.device, flat.dtype) + mu.to(flat.device, flat.dtype)
        return flat.reshape(b, l, -1)

    def state_dict(self) -> Dict:
        return {
            "memory": self.memory,
            "layer": self.layer,
            "m": self.m,
            "per_task_codebook": self.per_task_codebook,
            "whiten": self.whiten,
            "pq": self.pq.state_dict(),
            "pqs": {t: p.state_dict() for t, p in self.pqs.items()},
            "norm": self.norm,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.memory = state["memory"]
        self.layer = state.get("layer", self.layer)
        self.m = state.get("m", self.m)
        # Buffers written before per-task codebooks existed carry only "pq",
        # and were necessarily unwhitened with one shared codebook.
        self.per_task_codebook = state.get("per_task_codebook", False)
        self.whiten = state.get("whiten", False)
        self.pq.load_state_dict(state["pq"])
        self.pqs = {}
        for t, s in state.get("pqs", {}).items():
            pq = ProductQuantizer(m=s["m"])
            pq.load_state_dict(s)
            self.pqs[int(t)] = pq
        self.norm = state.get("norm", {})

    def __len__(self) -> int:
        return sum(v["codes"].shape[0] for v in self.memory.values())

    def nbytes(self) -> int:
        """Codes plus the codebooks and statistics needed to decode them."""
        code_bytes = sum(v["codes"].numel() for v in self.memory.values())
        if self.per_task_codebook:
            book_bytes = sum(p.codebook_nbytes() for p in self.pqs.values())
        else:
            book_bytes = self.pq.codebook_nbytes()
        norm_bytes = sum(mu.numel() * 2 + sd.numel() * 2
                         for mu, sd in self.norm.values())
        return code_bytes + book_bytes + norm_bytes

    def __repr__(self) -> str:
        lines = [
            f"RemindReplayBuffer(budget={self.total_budget}, layer={self.layer}, "
            f"m={self.m}, codebook={'per-task' if self.per_task_codebook else 'shared'}, "
            f"whiten={self.whiten}, tasks={len(self.memory)}, total_stored={len(self)}, "
            f"size={self.nbytes() / 1e6:.1f} MB)"
        ]
        for tid, entry in sorted(self.memory.items()):
            info = self.task_info.get(tid, {})
            lines.append(
                f"  Task {tid}: {entry['codes'].shape[0]} exemplars, "
                f"{len(info.get('classnames', []))} classes"
            )
        return "\n".join(lines)
