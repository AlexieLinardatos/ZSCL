"""
Toy 2D illustration of the exemplar-selection strategies (thesis Background).

Each rule mirrors mtil/src/exemplar_selection.py:
  random   : uniform over the whole task (not per class)
  herding  : per class, greedy running-mean matching to the class mean
  coverage : global; one seed per class (nearest its mean), then farthest-point fill
  el2n     : per class, largest ||p - onehot(y)||_2 (highest error first)
  gcr      : per class, greedy match of mean last-layer gradient (p - y) ⊗ x
Herding/EL2N/GCR use equal per-class quotas, as _per_class_budgets does.
Differences from the real code: 2D raw features (no L2 normalisation, which
would collapse 2D onto a circle) and a softmax regression in place of CLIP's
image-text logits.

Run from repo root:  python mtil/scripts/plot_selection_toy.py
Saves figs/selection_toy.{pdf,png}.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT = "figs/selection_toy"
K = 18  # total picks (6 per class for the per-class rules)
rng = np.random.default_rng(3)

# ---- data: 3 overlapping Gaussian classes, a few far outliers -------------
CENTERS = np.array([[0.0, 0.0], [3.2, 0.4], [1.5, 2.9]])
X, y = [], []
for c, mu in enumerate(CENTERS):
    X.append(rng.normal(mu, [0.85, 0.7], size=(150, 2)))
    y += [c] * 150
X = np.vstack(X)
y = np.array(y)
outliers = np.array([[-2.6, -1.6], [5.9, -1.3], [1.2, 5.6], [-1.8, 3.1]])
X = np.vstack([X, outliers])
y = np.concatenate([y, [0, 1, 2, 0]])
C = len(CENTERS)
COLORS = ["#4C78A8", "#F58518", "#54A24B"]


# ---- softmax regression (stand-in for CLIP zero-shot probabilities) -------
def fit_softmax(X, y, iters=3000, lr=0.1):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W = np.zeros((3, C))
    Y = np.eye(C)[y]
    for _ in range(iters):
        P = softmax(Xb @ W)
        W -= lr * Xb.T @ (P - Y) / len(X)
    return W


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


W = fit_softmax(X, y)
P = softmax(np.hstack([X, np.ones((len(X), 1))]) @ W)


# ---- selection rules (ports of exemplar_selection.py) ---------------------
def per_class(fn):
    k = K // C
    out = []
    for c in range(C):
        idx = np.where(y == c)[0]
        out += [idx[i] for i in fn(idx, c, k)]
    return out


def herding(idx, c, k):
    f = X[idx]
    mu = f.mean(0)
    sel, s, rem = [], np.zeros(2), list(range(len(f)))
    for step in range(k):
        d = np.linalg.norm(mu - (s + f[rem]) / (step + 1), axis=1)
        j = rem.pop(int(np.argmin(d)))
        sel.append(j)
        s += f[j]
    return sel


def el2n(idx, c, k):
    score = np.linalg.norm(P[idx] - np.eye(C)[c], axis=1)
    return list(np.argsort(-score)[:k])


def gcr(idx, c, k):
    err = P[idx] - np.eye(C)[c]
    g = np.einsum("nc,nd->ncd", err, X[idx]).reshape(len(idx), -1)
    full = g.mean(0)
    sel, s, rem = [], np.zeros_like(full), list(range(len(idx)))
    for step in range(k):
        d = np.linalg.norm(full - (s + g[rem]) / (step + 1), axis=1)
        j = rem.pop(int(np.argmin(d)))
        sel.append(j)
        s += g[j]
    return sel


def coverage():
    sel = []
    for c in range(C):
        idx = np.where(y == c)[0]
        sel.append(idx[np.argmin(np.linalg.norm(X[idx] - X[idx].mean(0), axis=1))])
    seeds = list(sel)
    dmin = np.min(np.linalg.norm(X[:, None] - X[sel][None], axis=2), axis=1)
    while len(sel) < K:
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[j], axis=1))
    return sel, seeds


PANELS = [
    ("Random", list(rng.choice(len(X), K, replace=False)), "uniform over the task"),
    ("Herding", per_class(herding), "near each class mean"),
    ("Coverage (k-center)", None, "spread over the whole cloud"),
    ("EL2N", per_class(el2n), "highest error: boundary + outliers"),
    ("GCR", per_class(gcr), "matches the mean gradient"),
]
cov_sel, cov_seeds = coverage()


def main():
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0), sharex=True, sharey=True)
    gx, gy = np.meshgrid(np.linspace(-3.5, 7, 300), np.linspace(-2.8, 6.5, 300))
    G = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
    region = softmax(G @ W).argmax(1).reshape(gx.shape)
    for ax, (name, sel, sub) in zip(axes, PANELS):
        if sel is None:
            sel = cov_sel
        if name in ("EL2N", "GCR"):  # show the classifier these rules score with
            ax.contourf(gx, gy, region, levels=[-.5, .5, 1.5, 2.5], colors=COLORS, alpha=0.07)
            ax.contour(gx, gy, region, levels=[.5, 1.5], colors="0.6", linewidths=0.6)
        for c in range(C):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], s=5, color=COLORS[c], alpha=0.18, linewidths=0)
        sel = np.array(sel)
        ax.scatter(X[sel, 0], X[sel, 1], s=34, c=[COLORS[c] for c in y[sel]],
                   edgecolors="black", linewidths=0.7, zorder=3)
        if name == "Herding":
            for c in range(C):
                mu = X[y == c].mean(0)
                ax.scatter(*mu, marker="x", s=55, color="black", linewidths=1.6, zorder=4)
        if name.startswith("Coverage"):
            ax.scatter(X[cov_seeds, 0], X[cov_seeds, 1], marker="*", s=130,
                       c=[COLORS[c] for c in y[cov_seeds]], edgecolors="black",
                       linewidths=0.7, zorder=4)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.75")
        ax.set_aspect("equal")
    axes[0].set_xlim(-3.5, 7); axes[0].set_ylim(-2.8, 6.5)
    fig.tight_layout(w_pad=0.6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
