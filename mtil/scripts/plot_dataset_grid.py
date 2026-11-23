"""MTIL sample grid: 3 test images per dataset, in training order.

Run on Nibi from mtil/ (needs the datasets and torchvision):
    python scripts/plot_dataset_grid.py                 # 11 x 3, about one page
    python scripts/plot_dataset_grid.py --layout cols   # 3 x 11, full text width
"""
import argparse
import os
import random
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import collections as C  # noqa: E402

ORDER = [("Aircraft", C.Aircraft), ("Caltech101", C.Caltech101),
         ("CIFAR100", C.CIFAR100), ("DTD", C.DTD), ("EuroSAT", C.EuroSAT),
         ("Flowers", C.Flowers), ("Food", C.Food), ("MNIST", C.MNIST),
         ("OxfordPet", C.OxfordPet), ("Cars", C.StanfordCars), ("SUN397", C.SUN397)]
K, TILE = 3, 192


def to_tile(img):
    if not isinstance(img, Image.Image):          # tensor fallback
        from torchvision.transforms.functional import to_pil_image
        img = to_pil_image(img)
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((TILE, TILE), Image.BICUBIC)


def sample(ds, rng):
    """K test images with distinct classes."""
    data, picks, seen = ds.test_dataset, [], set()
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for i in idx:
        img, y = data[i]
        y = int(y)
        if y in seen:
            continue
        seen.add(y)
        picks.append((to_tile(img), ds.classnames[y]))
        if len(picks) == K:
            break
    return picks


def label(name, width):
    return "\n".join(textwrap.wrap(name, width)[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-location", default="/scratch/alexie/data")
    ap.add_argument("--layout", choices=["rows", "cols"], default="rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../figs/mtil_samples")
    a = ap.parse_args()

    rows = []
    for name, cls in ORDER:
        ds = cls(None, location=a.data_location, batch_size=1, num_workers=0)
        rows.append((name, len(ds.classnames), sample(ds, random.Random(a.seed))))
        print(f"{name}: {len(ds.classnames)} classes, {[c for _, c in rows[-1][2]]}")

    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
    n = len(rows)
    if a.layout == "rows":
        fig, axes = plt.subplots(n, K, figsize=(4.6, 9.6),
                                 gridspec_kw=dict(wspace=0.08, hspace=0.42))
        grid = lambda r, k: axes[r, k]
    else:
        fig, axes = plt.subplots(K, n, figsize=(6.9, 2.35),
                                 gridspec_kw=dict(wspace=0.06, hspace=0.38))
        grid = lambda r, k: axes[k, r]

    for r, (name, ncls, picks) in enumerate(rows):
        for k, (img, cname) in enumerate(picks):
            ax = grid(r, k)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            ax.set_xlabel(label(cname, 22 if a.layout == "rows" else 13),
                          fontsize=5.5 if a.layout == "rows" else 4.5, labelpad=1.5)
        head = f"{name}\n({ncls} classes)"
        if a.layout == "rows":
            grid(r, 0).set_ylabel(head, fontsize=7, rotation=0, ha="right",
                                  va="center", labelpad=6)
        else:
            grid(r, 0).set_title(head, fontsize=5.5, pad=2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    suffix = "" if a.layout == "rows" else "_wide"
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}{suffix}.{ext}", dpi=300, bbox_inches="tight")
    print("saved", f"{a.out}{suffix}.pdf/.png")


if __name__ == "__main__":
    main()
