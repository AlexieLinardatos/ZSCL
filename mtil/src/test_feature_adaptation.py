"""
Self-test for the drift estimators, on synthetic data with a known answer.

No CLIP, no checkpoints, no dataset — runs in seconds and needs only torch, so
it is the cheap way to confirm the estimators are wired correctly before
spending GPU-hours on a run.

The setup mimics the real problem's structure. Points live in clusters (tasks),
the encoder's drift is a smooth field that varies across the space, and the
anchors observing that drift are drawn from *one* cluster while the points
needing correction sit in the others. That last part is the whole difficulty:
an estimator that only interpolates between nearby anchors has nothing to work
with, which is exactly the gap augmented anchors are meant to close.

Run:  python -m src.test_feature_adaptation
"""

import torch
import torch.nn.functional as F

from src.feature_adaptation import (
    apply_drift,
    estimate_drift,
)


def make_problem(
    dim: int = 128,
    n_clusters: int = 6,
    per_cluster: int = 300,
    n_anchor_clusters: int = 1,
    warp_strength: float = 0.25,
    seed: int = 0,
):
    """
    Build a synthetic old/new feature space with a known drift field.

    Returns:
        stored:       (M, D) old features needing correction.
        stored_true:  (M, D) their true position under the new encoder.
        anchor_src:   (A, D) old features whose drift is observed.
        anchor_delta: (A, D) the observed drift.
    """
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(n_clusters, dim, generator=g), dim=-1)

    feats, cluster_id = [], []
    for c in range(n_clusters):
        pts = centers[c] + 0.15 * torch.randn(per_cluster, dim, generator=g)
        feats.append(F.normalize(pts, dim=-1))
        cluster_id.append(torch.full((per_cluster,), c))
    feats = torch.cat(feats)
    cluster_id = torch.cat(cluster_id)

    # Drift field: a fixed random linear map, so drift varies smoothly with
    # position rather than being a constant translation an estimator could
    # trivially average out.
    A = torch.randn(dim, dim, generator=g) / dim ** 0.5
    new_feats = F.normalize(feats + warp_strength * (feats @ A), dim=-1)

    is_anchor = cluster_id < n_anchor_clusters
    anchor_src = feats[is_anchor]
    anchor_delta = new_feats[is_anchor] - anchor_src

    return feats[~is_anchor], new_feats[~is_anchor], anchor_src, anchor_delta


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] device={device}")

    stored, stored_true, anchor_src, anchor_delta = make_problem()
    stored = stored.to(device)
    stored_true = stored_true.to(device)
    anchor_src = anchor_src.to(device)
    anchor_delta = anchor_delta.to(device)

    print(f"[test] {stored.shape[0]} stored features to correct, "
          f"{anchor_src.shape[0]} anchors, dim={stored.shape[1]}")
    print(f"[test] anchors come from a cluster the stored features are not in, "
          f"so this measures extrapolation, not interpolation.\n")

    baseline = (stored * stored_true).sum(-1).mean().item()
    print(f"{'method':<12}{'cos to truth':>14}{'vs. uncorrected':>18}")
    print("-" * 44)
    print(f"{'none':<12}{baseline:>14.4f}{'—':>18}")

    results = {}
    for method in ("sdc", "lp", "linear", "mlp"):
        kwargs = (
            {"verbose": False, "steps": 3000, "lr": 3e-3}
            if method in ("linear", "mlp") else {}
        )
        delta = estimate_drift(method, stored, anchor_src, anchor_delta, **kwargs)
        adapted = apply_drift(stored, delta)
        cos = (adapted * stored_true).sum(-1).mean().item()
        results[method] = cos
        print(f"{method:<12}{cos:>14.4f}{cos - baseline:>+18.4f}")

    print()
    failures = []
    for method, cos in results.items():
        if cos < baseline:
            failures.append(
                f"{method} made things worse ({cos:.4f} < {baseline:.4f})"
            )

    # A linear map is the ground truth here, so the learned linear adapter
    # should be close to exact. If it is not, the fit is broken rather than the
    # problem being hard.
    if results["linear"] < 0.98:
        failures.append(
            f"linear adapter only reached {results['linear']:.4f} on a problem "
            f"whose true drift is linear — expected > 0.98"
        )

    if failures:
        print("[test] FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("[test] PASSED — every estimator improves on leaving features stale.")


if __name__ == "__main__":
    main()
