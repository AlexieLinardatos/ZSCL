#!/usr/bin/env python3
"""
Per-task analysis of what Replay Distillation adds on top of plain replay
(Reviewer zvRY, Q1: "show when RD helps beyond standard replay ... to decide
whether it helps overall or only on tasks close to CLIP pretraining").

Reads two accuracy matrices written by the phase3 outer loop
(``task_summary.csv``: one row per training stage, one column per eval set)
and reports, for every task, four quantities and the ExRD - replay-only delta
for each:

  Last        accuracy on task t in the final row (after all 11 tasks)
  Learned     accuracy on task t in the row where t was trained (diagonal)
  Forgetting  Learned - Last (positive = accuracy lost after moving on)
  Pre-exp.    mean accuracy on task t over the stages BEFORE t was trained
              (the per-task zero-shot / transfer quantity; undefined for the
              first task, which is trained at stage 0)

If a zero-shot reference row is supplied (``--zeroshot-csv``, produced by
scripts/11task/rebuttal_zeroshot_eval.sh), the script additionally splits the
tasks at the median zero-shot accuracy into "near CLIP pre-training" and "far
from CLIP pre-training" groups, reports each group's mean deltas, and gives
Pearson and Spearman correlations between zero-shot accuracy and each delta.
That is the direct answer to "does RD only help on tasks close to CLIP
pretraining".

Usage (from the repo root):

    python mtil/scripts/rebuttal/per_task_rd_analysis.py \
        --exrd-csv mtil/ckpt/11task/phase3_no_lora_v4/task_summary.csv \
        --nord-csv mtil/ckpt/11task/ablation_prop_replay/task_summary.csv \
        --zeroshot-csv mtil/ckpt/rebuttal/zeroshot/evaluate_all_results.csv \
        --out-dir mtil/ckpt/11task/rd_per_task

Writes ``per_task_rd_analysis.csv`` and ``per_task_rd_analysis.tex`` to
--out-dir in addition to the stdout report.
"""

import argparse
import csv
import math
import os
import sys

DEFAULT_TASKS = [
    "Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
    "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397", "ImageNet",
]

META_COLS = {"task_idx", "task_name", "avg"}


# --------------------------------------------------------------------------- #
# IO                                                                          #
# --------------------------------------------------------------------------- #

def read_matrix(path):
    """Return (rows, trained_order) from a task_summary.csv."""
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"error: no rows in {path}")
    return rows, [r["task_name"] for r in rows]


def read_zeroshot(path):
    """Return {dataset: top1} from an evaluate_all_results.csv, or None."""
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"warning: zero-shot CSV {path} not found — skipping the "
              f"proximity-to-pre-training analysis", file=sys.stderr)
        return None
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["dataset"]] = float(row["top1"])
    return out or None


def cell(rows, stage_idx, task):
    v = rows[stage_idx].get(task, "")
    return float(v) if v not in ("", None) else None


# --------------------------------------------------------------------------- #
# Per-task quantities                                                         #
# --------------------------------------------------------------------------- #

def per_task_stats(rows, order, task):
    """Last / Learned / Forgetting / Pre-exposure for one eval column."""
    last = cell(rows, len(rows) - 1, task)

    learned = None
    if task in order:
        learned = cell(rows, order.index(task), task)

    forgetting = None
    if last is not None and learned is not None:
        forgetting = learned - last

    # Stages strictly before the task was trained.  For a held-out column
    # (ImageNet) every stage counts as pre-exposure.
    if task in order:
        pre_stages = range(order.index(task))
    else:
        pre_stages = range(len(rows))
    pre_vals = [cell(rows, i, task) for i in pre_stages]
    pre_vals = [v for v in pre_vals if v is not None]
    pre = sum(pre_vals) / len(pre_vals) if pre_vals else None

    return {"last": last, "learned": learned, "forgetting": forgetting, "pre": pre}


def delta(a, b):
    return None if (a is None or b is None) else a - b


# --------------------------------------------------------------------------- #
# Correlation helpers (no scipy dependency)                                   #
# --------------------------------------------------------------------------- #

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(_ranks(xs), _ranks(ys))


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def fmt(v, width=7, prec=2, signed=False):
    if v is None:
        return " " * (width - 1) + "-"
    return f"{v:>+{width}.{prec}f}" if signed else f"{v:>{width}.{prec}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exrd-csv", default="mtil/ckpt/11task/phase3_no_lora_v4/task_summary.csv",
                   help="task_summary.csv for the full ExRD run (lambda=0.3)")
    p.add_argument("--nord-csv",
                   default="mtil/ckpt/11task/rebuttal_replay_only_matched/task_summary.csv",
                   help="task_summary.csv for replay-only (lambda=0, no RD). Defaults to "
                        "the iteration-matched baseline from "
                        "scripts/11task/rebuttal_replay_only_matched.sh; pointing this at "
                        "ckpt/11task/ablation_prop_replay instead confounds RD with the "
                        "per-task iteration schedule (see that script's header)")
    p.add_argument("--zeroshot-csv", default=None,
                   help="evaluate_all_results.csv with pre-trained CLIP zero-shot "
                        "accuracies (optional; enables the proximity analysis)")
    p.add_argument("--out-dir", default="mtil/ckpt/11task/rd_per_task")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--labels", nargs=2, default=("ExRD", "replay-only"),
                   metavar=("A", "B"), help="names for the two runs in the report")
    args = p.parse_args()

    a_label, b_label = args.labels
    a_rows, a_order = read_matrix(args.exrd_csv)
    b_rows, b_order = read_matrix(args.nord_csv)
    if a_order != b_order:
        print(f"warning: the two runs have different task orders\n"
              f"  {args.exrd_csv}: {a_order}\n"
              f"  {args.nord_csv}: {b_order}", file=sys.stderr)
    zs = read_zeroshot(args.zeroshot_csv)

    records = []
    for t in args.tasks:
        if t not in a_rows[0] or t not in b_rows[0]:
            print(f"warning: column {t!r} missing from one of the CSVs; skipping",
                  file=sys.stderr)
            continue
        sa = per_task_stats(a_rows, a_order, t)
        sb = per_task_stats(b_rows, b_order, t)
        records.append({
            "task": t,
            "held_out": t not in a_order,
            "zeroshot": zs.get(t) if zs else None,
            "a": sa, "b": sb,
            "d_last": delta(sa["last"], sb["last"]),
            "d_learned": delta(sa["learned"], sb["learned"]),
            "d_forget": delta(sa["forgetting"], sb["forgetting"]),
            "d_pre": delta(sa["pre"], sb["pre"]),
        })

    # ---------------------------------------------------------------- report #
    print()
    print(f"Per-task effect of Replay Distillation  ({a_label} minus {b_label})")
    print(f"  {a_label:<12} {args.exrd_csv}")
    print(f"  {b_label:<12} {args.nord_csv}")
    print()
    header = (f"{'Task':<14}{'ZS':>7}{'Last A':>8}{'Last B':>8}{'dLast':>8}"
              f"{'dLearn':>8}{'dForget':>9}{'PreExp A':>10}{'PreExp B':>10}{'dPreExp':>9}")
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['task']:<14}"
              f"{fmt(r['zeroshot'], 7)}"
              f"{fmt(r['a']['last'], 8)}{fmt(r['b']['last'], 8)}"
              f"{fmt(r['d_last'], 8, signed=True)}"
              f"{fmt(r['d_learned'], 8, signed=True)}"
              f"{fmt(r['d_forget'], 9, signed=True)}"
              f"{fmt(r['a']['pre'], 10)}{fmt(r['b']['pre'], 10)}"
              f"{fmt(r['d_pre'], 9, signed=True)}")
    print("-" * len(header))

    def mean(key, subset=None, exclude_held_out=True):
        vals = [r[key] for r in (subset if subset is not None else records)
                if r[key] is not None and not (exclude_held_out and r["held_out"])]
        return sum(vals) / len(vals) if vals else None

    print(f"{'mean (trained)':<14}{'':>7}{'':>8}{'':>8}"
          f"{fmt(mean('d_last'), 8, signed=True)}"
          f"{fmt(mean('d_learned'), 8, signed=True)}"
          f"{fmt(mean('d_forget'), 9, signed=True)}"
          f"{'':>10}{'':>10}{fmt(mean('d_pre'), 9, signed=True)}")

    helped = [r["task"] for r in records if r["d_last"] is not None and r["d_last"] > 0]
    hurt = [r["task"] for r in records if r["d_last"] is not None and r["d_last"] < 0]
    print()
    print(f"RD improves final accuracy on {len(helped)}/{len(records)} eval sets: "
          f"{', '.join(helped) if helped else 'none'}")
    print(f"RD lowers final accuracy on  {len(hurt)}/{len(records)} eval sets: "
          f"{', '.join(hurt) if hurt else 'none'}")

    helped_pre = [r["task"] for r in records if r["d_pre"] is not None and r["d_pre"] > 0]
    print(f"RD improves pre-exposure (zero-shot) accuracy on "
          f"{len(helped_pre)}/{sum(1 for r in records if r['d_pre'] is not None)} eval sets: "
          f"{', '.join(helped_pre) if helped_pre else 'none'}")

    # ------------------------------------------- proximity to pre-training  #
    if zs:
        usable = [r for r in records if r["zeroshot"] is not None]
        print()
        print("Proximity to CLIP pre-training (split at the median zero-shot accuracy)")
        zs_vals = sorted(r["zeroshot"] for r in usable)
        median = (zs_vals[len(zs_vals) // 2] if len(zs_vals) % 2
                  else 0.5 * (zs_vals[len(zs_vals) // 2 - 1] + zs_vals[len(zs_vals) // 2]))
        near = [r for r in usable if r["zeroshot"] >= median]
        far = [r for r in usable if r["zeroshot"] < median]
        print(f"  median zero-shot accuracy: {median:.2f}")
        for name, grp in (("near (high zero-shot)", near), ("far  (low zero-shot)", far)):
            print(f"  {name:<24} n={len(grp):<3} "
                  f"mean dLast={fmt(mean('d_last', grp), 7, signed=True)}   "
                  f"mean dPreExp={fmt(mean('d_pre', grp), 7, signed=True)}   "
                  f"[{', '.join(r['task'] for r in grp)}]")

        for key, label in (("d_last", "dLast"), ("d_pre", "dPreExp")):
            pairs = [(r["zeroshot"], r[key]) for r in usable if r[key] is not None]
            if len(pairs) >= 3:
                xs, ys = zip(*pairs)
                pr, sr = pearson(list(xs), list(ys)), spearman(list(xs), list(ys))
                print(f"  corr(zero-shot acc, {label:<8}) "
                      f"Pearson={pr:+.3f}  Spearman={sr:+.3f}  (n={len(pairs)})")

    # ---------------------------------------------------------------- files #
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "per_task_rd_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "held_out", "zeroshot",
                    f"last_{a_label}", f"last_{b_label}", "delta_last",
                    f"learned_{a_label}", f"learned_{b_label}", "delta_learned",
                    f"forgetting_{a_label}", f"forgetting_{b_label}", "delta_forgetting",
                    f"preexposure_{a_label}", f"preexposure_{b_label}", "delta_preexposure"])
        for r in records:
            def g(v):
                return "" if v is None else f"{v:.4f}"
            w.writerow([r["task"], int(r["held_out"]), g(r["zeroshot"]),
                        g(r["a"]["last"]), g(r["b"]["last"]), g(r["d_last"]),
                        g(r["a"]["learned"]), g(r["b"]["learned"]), g(r["d_learned"]),
                        g(r["a"]["forgetting"]), g(r["b"]["forgetting"]), g(r["d_forget"]),
                        g(r["a"]["pre"]), g(r["b"]["pre"]), g(r["d_pre"])])

    tex_path = os.path.join(args.out_dir, "per_task_rd_analysis.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by mtil/scripts/rebuttal/per_task_rd_analysis.py\n")
        f.write(f"% {a_label} vs {b_label}; Pre-exp. = mean accuracy before the task was trained.\n")
        for r in records:
            def tex(v, signed=False):
                if v is None:
                    return "--"
                if signed:
                    return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.2f}"
                return f"{v:.2f}"
            f.write(f"{r['task']:<14} & {tex(r['a']['last']):>6} & {tex(r['b']['last']):>6} "
                    f"& {tex(r['d_last'], True):>9} & {tex(r['a']['pre']):>6} "
                    f"& {tex(r['b']['pre']):>6} & {tex(r['d_pre'], True):>9} \\\\\n")

    print()
    print(f"wrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
