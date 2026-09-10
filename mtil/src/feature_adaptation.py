"""
Adapting stale stored features to a drifting encoder.

Feature replay stores an embedding once and never refreshes it, so as the
encoder keeps training the buffer describes a manifold the model no longer
produces.  This module estimates that drift and applies the correction.

The estimation problem.  At a task boundary we hold two encoders: f_old (the
checkpoint that stored the features, i.e. the state at the start of this task)
and f_new (the one that just finished training).  On the *current* task's data
the drift is directly observable — encode the same image twice and subtract:

    delta(x) = f_new(x) - f_old(x)

but the stored features we need to correct come from *earlier* tasks and sit
somewhere else in feature space entirely.  So the observed drift field has to be
extrapolated from where it was measured to where it is needed.  That gap is the
whole difficulty, and it is the same gap Zhang et al. address with augmented
anchors, transposed from source/target domains to old/new encoders.

Four estimators, in increasing ambition:

  none    Leave stored features alone. The V0 baseline.

  sdc     Gaussian kernel-weighted average of nearby observed drift vectors,
          following Yu et al., "Semantic Drift Compensation for Class-
          Incremental Learning" (CVPR 2020). One hyperparameter, no training.

  lp      Label propagation of the drift field over a k-NN graph on the union of
          {stored features} and {anchors}, following Zhang et al., "Label
          Propagation with Augmented Anchors" (ECCV 2020). Replaces SDC's single
          kernel with diffusion through the graph, so drift reaches stored
          features through chains of intermediate points rather than only from
          anchors that happen to be nearby.

  linear  A learned map from the old embedding space to the new one, fit on the
  mlp     observed pairs. This is Iscen et al., "Memory-Efficient Incremental
          Learning Through Feature Adaptation" (ECCV 2020) — the comparison that
          says whether the graph earns its complexity.

Anchor sources (the `anchors` argument of collect_anchors):

  image   Current-task images encoded twice, plus their class-wise means. The
          faithful port of A2LP's augmented anchors: the class means are denser
          and less noisy than individual observations.

  text    CLIP-specific, and the reason this is worth a paper. An old class's
          *text* embedding can be recomputed exactly at any time — it needs no
          stored image, just the class name. So g_old(c) and g_new(c) are both
          free, and their difference is an observed drift vector sitting at
          class c's own location in the joint space, which is where that class's
          stale image features live. A2LP has to synthesise anchors in the
          target domain because it cannot observe any; we can observe ours.

          Caveat worth testing rather than assuming: CLIP's modality gap means
          text and image embeddings of the same class occupy different cones, so
          a text anchor is offset from the image features it is meant to guide.
          A constant offset cancels in a difference, which is why this should
          work — but that is an empirical claim, hence the `image`/`text`/`both`
          ablation.

  both    Union of the two. The intended default once text anchors validate.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Anchor collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_images(model, loader, device: str) -> torch.Tensor:
    feats = []
    for item in loader:
        if isinstance(item, (tuple, list)):
            images = item[0]
        else:
            images = item["images"]
        emb = model(images.to(device), None)
        feats.append(F.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _encode_labels(loader) -> torch.Tensor:
    labels = []
    for item in loader:
        if isinstance(item, (tuple, list)):
            labels.append(item[1])
        else:
            labels.append(item["labels"])
    return torch.cat(labels, dim=0).long()


@torch.no_grad()
def collect_image_anchors(
    old_model,
    new_model,
    loader,
    device: str = "cuda",
    add_class_means: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Observe the drift field on current-task images.

    Encodes the same batches with both encoders, so anchor i is the pair
    (f_old(x_i), f_new(x_i) - f_old(x_i)).  When `add_class_means` is set, the
    per-class means of both sides are appended as additional anchors — A2LP's
    augmented anchors, which are lower-variance than individual observations.

    Returns:
        (src, delta), each (A, D) float32 CPU tensors.  `src` is the anchor's
        position in the *old* feature space, `delta` its observed drift.
    """
    # One pass per encoder over the same loader; the loader must therefore be
    # deterministic (shuffle=False) for the two passes to line up index-wise.
    old_feats = _encode_images(old_model, loader, device)
    new_feats = _encode_images(new_model, loader, device)
    src = old_feats
    delta = new_feats - old_feats

    if add_class_means:
        labels = _encode_labels(loader)
        mean_src, mean_delta = [], []
        for c in labels.unique().tolist():
            m = labels == c
            mean_src.append(F.normalize(old_feats[m].mean(dim=0), dim=-1))
            mean_delta.append(delta[m].mean(dim=0))
        if mean_src:
            src = torch.cat([src, torch.stack(mean_src)], dim=0)
            delta = torch.cat([delta, torch.stack(mean_delta)], dim=0)

    return src, delta


@torch.no_grad()
def collect_text_anchors(
    old_model,
    new_model,
    task_info: Dict[int, Dict],
    device: str = "cuda",
    batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Observe the drift field at old classes' own positions, via the text tower.

    For every class name in the buffer's task metadata, encode its prompt with
    both text encoders.  No images are needed and nothing is stored, so these
    anchors are available for *previous* tasks — unlike image anchors, which
    only exist for the current task.

    Returns:
        (src, delta), each (C, D) float32 CPU tensors.
    """
    import clip.clip as clip

    prompts: List[str] = []
    for tid in sorted(task_info):
        info = task_info[tid]
        template = info["template"]
        prompts.extend(template(c) for c in info["classnames"])

    if not prompts:
        dim = 512
        return torch.empty(0, dim), torch.empty(0, dim)

    old_embs, new_embs = [], []
    for i in range(0, len(prompts), batch_size):
        tokens = clip.tokenize(prompts[i:i + batch_size]).to(device)
        e_old = F.normalize(old_model(None, tokens).float(), dim=-1)
        e_new = F.normalize(new_model(None, tokens).float(), dim=-1)
        old_embs.append(e_old.cpu())
        new_embs.append(e_new.cpu())

    src = torch.cat(old_embs, dim=0)
    delta = torch.cat(new_embs, dim=0) - src
    return src, delta


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _pairwise_cosine_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    chunk: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k cosine similarities from each query row to the key rows.

    Chunked over queries: the full similarity matrix for a 12k-node graph is
    576 MB in fp32, which is wasteful when only k columns per row survive.

    Returns:
        (values, indices), each (Q, k).
    """
    q = F.normalize(query, dim=-1)
    kk = F.normalize(key, dim=-1)
    vals, idxs = [], []
    for i in range(0, q.shape[0], chunk):
        sim = q[i:i + chunk] @ kk.t()
        v, j = sim.topk(min(k, kk.shape[0]), dim=-1)
        vals.append(v)
        idxs.append(j)
    return torch.cat(vals, dim=0), torch.cat(idxs, dim=0)


def estimate_drift_sdc(
    stored: torch.Tensor,
    anchor_src: torch.Tensor,
    anchor_delta: torch.Tensor,
    sigma: Optional[float] = None,
    k: int = 32,
) -> torch.Tensor:
    """
    Semantic Drift Compensation: Gaussian kernel-weighted average of the
    nearest observed drift vectors (Yu et al., CVPR 2020).

        delta_hat_j = sum_i w_ij * delta_i / sum_i w_ij
        w_ij = exp(-||stored_j - anchor_src_i||^2 / (2 sigma^2))

    Restricted to each stored feature's k nearest anchors, which changes
    nothing numerically once sigma is small relative to the point cloud (distant
    anchors contribute ~0) and keeps the cost linear in k.

    Args:
        sigma: kernel width. None picks the mean distance to the k-th nearest
               anchor, which adapts to how spread out the space is.

    Returns:
        (M, D) estimated drift for each stored feature.
    """
    if anchor_src.numel() == 0:
        return torch.zeros_like(stored)

    sims, idxs = _pairwise_cosine_topk(stored, anchor_src, k)
    # Both sides are unit-norm, so squared euclidean = 2 - 2 * cosine.
    d2 = (2.0 - 2.0 * sims).clamp_min(0.0)

    if sigma is None:
        sigma = d2[:, -1].sqrt().mean().clamp_min(1e-6).item()

    w = torch.exp(-d2 / (2.0 * sigma ** 2))
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # (M, k, D) gather would blow up memory at large k; einsum over the gather
    # of anchor deltas keeps it to one (M, k, D) temporary per chunk instead.
    out = torch.empty_like(stored)
    chunk = 4096
    for i in range(0, stored.shape[0], chunk):
        neigh = anchor_delta[idxs[i:i + chunk]]           # (c, k, D)
        out[i:i + chunk] = (w[i:i + chunk].unsqueeze(-1) * neigh).sum(dim=1)
    return out


def estimate_drift_lp(
    stored: torch.Tensor,
    anchor_src: torch.Tensor,
    anchor_delta: torch.Tensor,
    k: int = 20,
    alpha: float = 0.85,
    iters: int = 30,
    gamma: float = 3.0,
    clamp_anchors: bool = True,
) -> torch.Tensor:
    """
    Label propagation of the drift field, with augmented anchors
    (Zhang et al., ECCV 2020), treating drift as a D-dimensional soft label.

    Builds a symmetric k-NN graph over the union of stored features and
    anchors, then diffuses the known anchor drifts through it:

        F <- alpha * S F + (1 - alpha) * Y,     S = D^-1/2 W D^-1/2

    iterated rather than solved as F* = (I - alpha S)^-1 Y, because the closed
    form needs an O(n^3) dense inverse and the iteration converges in tens of
    sparse matmuls.

    The difference from SDC is reachability: SDC only sees drift from anchors
    within one kernel width, so a stored feature in a region the current task
    never visits gets no correction at all.  Diffusion routes drift to it
    through intermediate stored features.

    Args:
        gamma:         similarity sharpening exponent, W_ij = relu(cos)^gamma.
        clamp_anchors: re-pin anchor rows to their observed drift after each
                       iteration. Hard clamping, appropriate here because the
                       anchor values are measured exactly rather than predicted.

    Returns:
        (M, D) estimated drift for each stored feature.
    """
    if anchor_src.numel() == 0:
        return torch.zeros_like(stored)

    M = stored.shape[0]
    A = anchor_src.shape[0]
    n = M + A
    device = stored.device

    nodes = torch.cat([stored, F.normalize(anchor_src, dim=-1)], dim=0)

    # k-NN graph. k + 1 because each node's nearest neighbour is itself.
    sims, idxs = _pairwise_cosine_topk(nodes, nodes, k + 1)
    rows = torch.arange(n, device=device).unsqueeze(1).expand_as(idxs)
    self_edge = idxs == rows
    weights = sims.clamp_min(0.0).pow(gamma)
    weights = weights.masked_fill(self_edge, 0.0)

    edge_idx = torch.stack([rows.reshape(-1), idxs.reshape(-1)], dim=0)
    edge_w = weights.reshape(-1)
    keep = edge_w > 0
    edge_idx, edge_w = edge_idx[:, keep], edge_w[keep]

    # Symmetrise by adding the transpose, then row/column normalise.
    edge_idx = torch.cat([edge_idx, edge_idx.flip(0)], dim=1)
    edge_w = torch.cat([edge_w, edge_w], dim=0)
    W = torch.sparse_coo_tensor(edge_idx, edge_w, (n, n)).coalesce()

    deg = torch.sparse.sum(W, dim=1).to_dense().clamp_min(1e-12)
    dinv = deg.pow(-0.5)
    vals = W.values() * dinv[W.indices()[0]] * dinv[W.indices()[1]]
    S = torch.sparse_coo_tensor(W.indices(), vals, (n, n)).coalesce()

    Y = torch.zeros(n, stored.shape[1], device=device, dtype=stored.dtype)
    Y[M:] = anchor_delta

    Fmat = Y.clone()
    for _ in range(iters):
        Fmat = alpha * torch.sparse.mm(S, Fmat) + (1.0 - alpha) * Y
        if clamp_anchors:
            Fmat[M:] = anchor_delta

    return Fmat[:M]


def estimate_drift_learned(
    stored: torch.Tensor,
    anchor_src: torch.Tensor,
    anchor_delta: torch.Tensor,
    hidden: Optional[int] = None,
    steps: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Fit a map from the old embedding space to the new one on the observed pairs
    and apply it to the stored features (Iscen et al., ECCV 2020).

    Predicts the drift rather than the absolute target, so the identity map is
    the zero function and an untrained adapter degrades to leaving features
    alone rather than to noise.

    Args:
        hidden: None fits a single linear layer; an int fits a one-hidden-layer
                MLP of that width with a residual connection.

    Returns:
        (M, D) estimated drift for each stored feature.
    """
    if anchor_src.numel() == 0:
        return torch.zeros_like(stored)

    device = stored.device
    dim = stored.shape[1]

    if hidden is None:
        net = torch.nn.Linear(dim, dim).to(device)
        torch.nn.init.zeros_(net.weight)
        torch.nn.init.zeros_(net.bias)
    else:
        net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, dim),
        ).to(device)
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)

    src = F.normalize(anchor_src, dim=-1).to(device)
    tgt = anchor_delta.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n = src.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        pred = net(src[idx])
        # MSE on the drift vector. Direction and magnitude are both wanted here
        # (the correction is added before re-normalising), so a plain L2 fit is
        # the right objective rather than a cosine one.
        loss = F.mse_loss(pred, tgt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (step + 1) % max(1, steps // 4) == 0:
            print(f"[feature-adapt] adapter step {step + 1}/{steps} "
                  f"mse={loss.item():.6f}")

    with torch.no_grad():
        out = torch.cat([
            net(stored[i:i + 4096]) for i in range(0, stored.shape[0], 4096)
        ], dim=0)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ADAPT_METHODS = ("none", "sdc", "lp", "linear", "mlp")


def estimate_drift(
    method: str,
    stored: torch.Tensor,
    anchor_src: torch.Tensor,
    anchor_delta: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Dispatch to the named estimator. See ADAPT_METHODS for valid names."""
    if method == "none":
        return torch.zeros_like(stored)
    if method == "sdc":
        return estimate_drift_sdc(
            stored, anchor_src, anchor_delta,
            sigma=kwargs.get("sigma"), k=kwargs.get("k", 32),
        )
    if method == "lp":
        return estimate_drift_lp(
            stored, anchor_src, anchor_delta,
            k=kwargs.get("k", 20),
            alpha=kwargs.get("alpha", 0.85),
            iters=kwargs.get("iters", 30),
            gamma=kwargs.get("gamma", 3.0),
            clamp_anchors=kwargs.get("clamp_anchors", True),
        )
    if method in ("linear", "mlp"):
        return estimate_drift_learned(
            stored, anchor_src, anchor_delta,
            hidden=(None if method == "linear" else kwargs.get("hidden", 1024)),
            steps=kwargs.get("steps", 500),
            lr=kwargs.get("lr", 1e-3),
            verbose=kwargs.get("verbose", True),
        )
    raise ValueError(f"Unknown feature adaptation method '{method}'. "
                     f"Expected one of {ADAPT_METHODS}.")


def apply_drift(stored: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Move stored features along the estimated drift and re-normalise.

    Stored features are unit-norm by construction, and the CE that consumes
    them assumes that, so the correction has to be projected back to the sphere.
    """
    return F.normalize(stored + delta, dim=-1)


def drift_report(
    stored_before: torch.Tensor,
    stored_after: torch.Tensor,
    anchor_src: torch.Tensor,
    anchor_delta: torch.Tensor,
) -> Dict[str, float]:
    """
    Summary statistics for one adaptation step, for logging.

    `observed_drift_cos` is the mean cosine between the anchors' old and new
    positions — how far the encoder actually moved on data we could measure.
    `correction_cos` is how far the stored features were moved.  The two being
    wildly different is the signal that the extrapolation has gone wrong.
    """
    with torch.no_grad():
        moved = (stored_before * stored_after).sum(dim=-1)
        if anchor_src.numel():
            a_new = F.normalize(F.normalize(anchor_src, dim=-1) + anchor_delta, dim=-1)
            observed = (F.normalize(anchor_src, dim=-1) * a_new).sum(dim=-1)
            observed_mean = observed.mean().item()
        else:
            observed_mean = float("nan")
        return {
            "observed_drift_cos": observed_mean,
            "correction_cos": moved.mean().item(),
            "correction_norm": (stored_after - stored_before).norm(dim=-1).mean().item(),
            "n_anchors": int(anchor_src.shape[0]) if anchor_src.numel() else 0,
        }
