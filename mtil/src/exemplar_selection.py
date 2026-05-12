"""
Exemplar selection strategies for the Phase 3 replay buffer.

Strategies:
    random   - uniform random (matches the original paper baseline)
    herding  - iCaRL-style greedy: pick samples whose running mean is closest
               to the class-mean CLIP feature (Rebuffi 2017)
    el2n     - score = || softmax(logits) - onehot(y) ||_2; pick top-k highest
               (Paul 2021)
    gcr      - gradient coreset replay: pick samples whose running mean
               last-layer gradient matches the full-set mean gradient
               (Tiwari 2022, simplified greedy variant)

All non-random strategies operate class-conditionally inside each task, taking
budget // num_classes slots per class (with remainder distribution).
"""

import random
from typing import Callable, Dict, List, Optional

import clip.clip as clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def select_exemplars(
    dataset: Dataset,
    num_samples: int,
    classnames: List[str],
    template: Callable[[str], str],
    strategy: str,
    model: Optional[torch.nn.Module] = None,
    device: str = "cuda",
    batch_size: int = 128,
) -> List[int]:
    """Return a list of dataset indices to keep as exemplars."""
    n = len(dataset)
    num_samples = min(num_samples, n)

    if strategy == "random" or model is None:
        return random.sample(range(n), num_samples)

    features, logits, labels = _extract_features_logits(
        model, dataset, classnames, template, device, batch_size
    )
    probs = F.softmax(logits, dim=1)

    budgets = _per_class_budgets(labels, num_samples)

    selected: List[int] = []
    for class_id, budget in budgets.items():
        if budget <= 0:
            continue
        class_indices = torch.nonzero(labels == class_id, as_tuple=True)[0].tolist()
        if budget >= len(class_indices):
            selected.extend(class_indices)
            continue

        class_feats = features[class_indices]
        class_probs = probs[class_indices]

        if strategy == "herding":
            picked = _select_herding(class_feats, budget)
        elif strategy == "el2n":
            picked = _select_el2n(class_probs, class_id, budget)
        elif strategy == "gcr":
            picked = _select_gcr(class_feats, class_probs, class_id, budget)
        else:
            raise ValueError(f"Unknown exemplar strategy: {strategy}")

        selected.extend(class_indices[i] for i in picked)

    return selected


# ---------------------------------------------------------------------------
# Feature / logit extraction (single forward pass, shared by herding/el2n/gcr)
# ---------------------------------------------------------------------------

def _extract_features_logits(
    model: torch.nn.Module,
    dataset: Dataset,
    classnames: List[str],
    template: Callable[[str], str],
    device: str,
    batch_size: int,
):
    model.eval()

    with torch.no_grad():
        prompts = [template(c) for c in classnames]
        tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False
    )

    feats_list, logits_list, labels_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[exemplar-select] extracting features"):
            if isinstance(batch, (tuple, list)):
                images, labs = batch[0], batch[1]
            else:
                images, labs = batch["images"], batch["labels"]
            images = images.to(device)

            img_feats = model.encode_image(images).float()
            img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
            batch_logits = img_feats_norm @ text_features.t()

            feats_list.append(img_feats.cpu())
            logits_list.append(batch_logits.cpu())
            labels_list.append(torch.as_tensor(labs).cpu())

    features = torch.cat(feats_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).long()
    return features, logits, labels


# ---------------------------------------------------------------------------
# Per-class budget allocation
# ---------------------------------------------------------------------------

def _per_class_budgets(labels: torch.Tensor, num_samples: int) -> Dict[int, int]:
    """Allocate num_samples across classes; small classes take all and yield
    leftover to other classes."""
    unique, counts = torch.unique(labels, return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(unique.tolist(), counts.tolist())}

    budgets: Dict[int, int] = {c: 0 for c in class_counts}
    leftover = num_samples

    classes_by_size = sorted(class_counts, key=lambda c: class_counts[c])
    n_remaining = len(classes_by_size)
    for c in classes_by_size:
        if n_remaining <= 0:
            break
        quota = leftover // n_remaining
        take = min(class_counts[c], quota)
        budgets[c] = take
        leftover -= take
        n_remaining -= 1

    classes_by_size_desc = sorted(class_counts, key=lambda c: class_counts[c], reverse=True)
    while leftover > 0:
        progressed = False
        for c in classes_by_size_desc:
            if budgets[c] < class_counts[c]:
                budgets[c] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break

    return budgets


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _select_herding(features: torch.Tensor, k: int) -> List[int]:
    feats = features / features.norm(dim=-1, keepdim=True)
    class_mean = feats.mean(dim=0)

    selected: List[int] = []
    cur_sum = torch.zeros_like(class_mean)
    remaining = list(range(len(feats)))

    for step in range(min(k, len(feats))):
        cand = feats[remaining]
        cand_means = (cur_sum.unsqueeze(0) + cand) / (step + 1)
        dists = torch.norm(class_mean.unsqueeze(0) - cand_means, dim=1)
        best_local = int(torch.argmin(dists).item())
        best_global = remaining[best_local]
        selected.append(best_global)
        cur_sum = cur_sum + feats[best_global]
        remaining.pop(best_local)

    return selected


def _select_el2n(probs: torch.Tensor, class_id: int, k: int) -> List[int]:
    n, c = probs.shape
    onehot = torch.zeros(c)
    onehot[class_id] = 1.0
    scores = torch.norm(probs - onehot.unsqueeze(0), dim=1)
    return torch.topk(scores, k=min(k, n), largest=True).indices.tolist()


def _select_gcr(
    features: torch.Tensor, probs: torch.Tensor, class_id: int, k: int
) -> List[int]:
    """Greedy gradient-matching selection.

    Last-layer gradient embedding for CE on a frozen-feature classifier is
        g_i = (p_i - y_i) ⊗ φ(x_i)
    (outer product, no autograd needed). We pick samples one at a time that
    minimize || mean(g_full) - mean(g_selected ∪ {i}) || over remaining i.
    """
    n, d = features.shape
    c = probs.shape[1]
    onehot = torch.zeros(c)
    onehot[class_id] = 1.0
    err = probs - onehot.unsqueeze(0)

    feats = features / features.norm(dim=-1, keepdim=True)
    grads = torch.einsum("nc,nd->ncd", err, feats).reshape(n, -1)
    mean_full = grads.mean(dim=0)

    selected: List[int] = []
    cur_sum = torch.zeros_like(mean_full)
    remaining = list(range(n))

    for step in range(min(k, n)):
        cand = grads[remaining]
        cand_means = (cur_sum.unsqueeze(0) + cand) / (step + 1)
        dists = torch.norm(mean_full.unsqueeze(0) - cand_means, dim=1)
        best_local = int(torch.argmin(dists).item())
        best_global = remaining[best_local]
        selected.append(best_global)
        cur_sum = cur_sum + grads[best_global]
        remaining.pop(best_local)

    return selected
