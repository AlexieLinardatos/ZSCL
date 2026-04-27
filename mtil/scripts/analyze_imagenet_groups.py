"""
Semantic group analysis of per-class ImageNet accuracy.

Takes the two JSONs produced by perclass_imagenet_eval.py and produces:
  - A bar chart of accuracy delta (Ours - ZSCL) per semantic group
  - A CSV of group-level accuracy for both methods

Usage (locally, after scp-ing the JSONs from Nibi):

    python scripts/analyze_imagenet_groups.py \
        --ours   perclass_ours.json \
        --zscl   perclass_zscl.json \
        --output imagenet_group_delta

Outputs: imagenet_group_delta.pdf, imagenet_group_delta.png, imagenet_group_delta.csv
"""

import argparse
import json
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Semantic groups: map ImageNet class name keywords → group label.
# Each class is assigned to the FIRST group whose keywords match.
# "Other" catches anything unmatched.
# ---------------------------------------------------------------------------
GROUPS = [
    ("Dogs & Cats",       ["dog", "hound", "terrier", "retriever", "spaniel",
                            "shepherd", "poodle", "bulldog", "cat", "kitten",
                            "tabby", "persian"]),
    ("Other Animals",     ["bird", "fish", "shark", "snake", "turtle", "frog",
                            "insect", "butterfly", "beetle", "spider", "crab",
                            "lobster", "whale", "dolphin", "bear", "lion",
                            "tiger", "elephant", "monkey", "gorilla", "deer",
                            "rabbit", "squirrel", "fox", "wolf", "horse",
                            "zebra", "giraffe", "penguin", "flamingo", "eagle",
                            "owl", "parrot", "hen", "rooster", "duck", "goose"]),
    ("Vehicles",          ["car", "truck", "bus", "van", "jeep", "vehicle",
                            "aircraft", "airplane", "helicopter", "jet",
                            "ship", "boat", "submarine", "train", "locomotive",
                            "bicycle", "motorcycle", "scooter", "ambulance",
                            "fire engine", "tractor", "forklift"]),
    ("Food & Produce",    ["pizza", "burger", "hot dog", "sandwich", "taco",
                            "sushi", "noodle", "soup", "bread", "cake",
                            "ice cream", "cheese", "mushroom", "broccoli",
                            "cauliflower", "cucumber", "lemon", "orange",
                            "apple", "banana", "strawberry", "pineapple",
                            "avocado", "pretzel", "espresso", "wine",
                            "beer bottle", "coffee"]),
    ("Plants & Flowers",  ["flower", "daisy", "tulip", "rose", "sunflower",
                            "orchid", "cactus", "fern", "palm", "tree",
                            "plant", "leaf", "vine", "blossom", "petal"]),
    ("Clothing & Fashion",["jersey", "suit", "tie", "sock", "shoe", "boot",
                            "sandal", "hat", "cap", "helmet", "glove",
                            "scarf", "skirt", "dress", "bikini", "apron",
                            "mask", "uniform", "sunglasses", "watch"]),
    ("Electronics & Tools",["computer", "keyboard", "mouse", "monitor",
                              "laptop", "phone", "remote", "television",
                              "camera", "projector", "printer", "clock",
                              "radio", "microphone", "speaker", "drill",
                              "hammer", "wrench", "screwdriver", "axe",
                              "saw", "shovel", "scissors"]),
    ("Household Objects", ["chair", "table", "desk", "sofa", "couch", "bed",
                            "lamp", "shelf", "cabinet", "drawer", "mirror",
                            "vase", "cup", "mug", "bowl", "plate", "spoon",
                            "fork", "knife", "bottle", "jug", "pot", "pan",
                            "pillow", "blanket", "curtain", "basket", "bucket"]),
    ("Structures & Scenes",["bridge", "building", "church", "castle", "tower",
                              "barn", "lighthouse", "fountain", "dam", "cliff",
                              "volcano", "valley", "mountain", "beach", "lake",
                              "river", "forest", "jungle", "desert", "field"]),
]


def assign_group(classname: str) -> str:
    name_lower = classname.lower()
    for group_label, keywords in GROUPS:
        if any(kw in name_lower for kw in keywords):
            return group_label
    return "Other"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True, help="perclass_ours.json")
    parser.add_argument("--zscl", required=True, help="perclass_zscl.json")
    parser.add_argument("--output", default="imagenet_group_delta",
                        help="Output filename stem (no extension).")
    args = parser.parse_args()

    ours = load_json(args.ours)
    zscl = load_json(args.zscl)

    print(f"Ours mean top-1:  {ours['mean_top1']:.2f}%")
    print(f"ZSCL mean top-1:  {zscl['mean_top1']:.2f}%")

    # Build per-group accuracy
    group_ours = {g: [] for g, _ in GROUPS}
    group_ours["Other"] = []
    group_zscl = {g: [] for g, _ in GROUPS}
    group_zscl["Other"] = []

    for o, z in zip(ours["classes"], zscl["classes"]):
        assert o["id"] == z["id"] and o["name"] == z["name"]
        if o["n"] == 0:
            continue
        grp = assign_group(o["name"])
        group_ours[grp].append(o["acc"])
        group_zscl[grp].append(z["acc"])

    group_labels = [g for g, _ in GROUPS] + ["Other"]
    ours_means = []
    zscl_means = []
    deltas = []
    counts = []
    for grp in group_labels:
        o_vals = group_ours[grp]
        z_vals = group_zscl[grp]
        if len(o_vals) == 0:
            ours_means.append(0.0)
            zscl_means.append(0.0)
            deltas.append(0.0)
            counts.append(0)
        else:
            om = float(np.mean(o_vals)) * 100
            zm = float(np.mean(z_vals)) * 100
            ours_means.append(om)
            zscl_means.append(zm)
            deltas.append(om - zm)
            counts.append(len(o_vals))

    # --- Print table ---
    print(f"\n{'Group':<25} {'#cls':>5} {'Ours':>7} {'ZSCL':>7} {'Delta':>7}")
    print("-" * 55)
    for grp, n, om, zm, d in zip(group_labels, counts, ours_means, zscl_means, deltas):
        print(f"{grp:<25} {n:>5} {om:>7.2f} {zm:>7.2f} {d:>+7.2f}")

    # --- Save CSV ---
    csv_path = args.output + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_classes", "ours_acc", "zscl_acc", "delta"])
        for grp, n, om, zm, d in zip(group_labels, counts, ours_means, zscl_means, deltas):
            w.writerow([grp, n, f"{om:.4f}", f"{zm:.4f}", f"{d:.4f}"])
    print(f"\nSaved CSV: {csv_path}")

    # --- Plot ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Filter out groups with 0 classes
    plot_labels = [g for g, n in zip(group_labels, counts) if n > 0]
    plot_deltas = [d for d, n in zip(deltas, counts) if n > 0]
    plot_counts = [n for n in counts if n > 0]

    x = np.arange(len(plot_labels))
    colors = ["#2166AC" if d >= 0 else "#D6604D" for d in plot_deltas]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(x, plot_deltas, color=colors, edgecolor="white",
                  linewidth=0.5, zorder=3)

    for bar, d, n in zip(bars, plot_deltas, plot_counts):
        va = "bottom" if d >= 0 else "top"
        ypos = d + (0.05 if d >= 0 else -0.05)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{d:+.2f}", ha="center", va=va, fontsize=7.5,
                color=bar.get_facecolor())
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(plot_deltas) - 0.4,
                f"n={n}", ha="center", va="top", fontsize=6.5,
                color="#888888")

    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy delta: Ours − ZSCL (%)")
    ax.set_title("Per-semantic-group ImageNet Transfer gap (Ours vs ZSCL)",
                 fontsize=10, pad=10)
    ax.grid(True, alpha=0.2, linestyle=":", axis="y", zorder=0)

    pos_patch = mpatches.Patch(color="#2166AC", label="Ours > ZSCL")
    neg_patch = mpatches.Patch(color="#D6604D", label="Ours < ZSCL")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="upper right")

    overall_delta = ours["mean_top1"] - zscl["mean_top1"]
    ax.axhline(overall_delta, color="#444444", linestyle="--",
               linewidth=1.0, alpha=0.6,
               label=f"Overall delta ({overall_delta:+.2f}%)")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{args.output}.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()


if __name__ == "__main__":
    main()
