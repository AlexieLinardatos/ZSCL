"""
Print the 2x2 (ZSCL on/off, RD on/off) ablation cell matrix for the
no-ZSCL substitution claim. Reuses task_summary.csv files written by
finetune_phase3.py.

Usage (from mtil/):
    python scripts/no_zscl_ablation_table.py

Looks for these four runs under ckpt/11task/:
    ablation_prop_replay/         (ZSCL on,  RD off)  -- existing
    phase3_no_lora_v4/            (ZSCL on,  RD on)   -- existing "Ours full"
    ablation_no_zscl_no_rd/       (ZSCL off, RD off)  -- new
    ablation_no_zscl_with_rd/     (ZSCL off, RD on)   -- new

Cells with a missing task_summary.csv are reported as MISSING so you can
run it before the new jobs finish and just see the existing two rows.

Metrics (matching mtil/scripts/compute_metrics.py + experiments_findings.readme):
    Last     = mean of task cols in the final row (ImageNet excluded)
    Avg      = mean over ALL task-col x row cells (ZSCL paper definition)
    Transfer = mean ImageNet across all rows
"""

import csv
import os
import sys

CKPT_ROOT = "ckpt/11task"

CELLS = [
    # (zscl_on, rd_on, run_dir, label)
    (True,  False, "ablation_prop_replay",       "ZSCL on,  RD off  (+ prop replay)"),
    (True,  True,  "phase3_no_lora_v4",          "ZSCL on,  RD on   (Ours full)"),
    (False, False, "ablation_no_zscl_no_rd",     "ZSCL off, RD off  (replay only)"),
    (False, True,  "ablation_no_zscl_with_rd",   "ZSCL off, RD on   (RD substitutes)"),
]


def metrics_from_csv(path):
    """Return dict with last, avg, transfer or None if file missing/empty."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    meta = {"task_idx", "task_name", "avg"}
    task_cols = [c for c in rows[0].keys() if c not in meta and c != "ImageNet"]

    # Last = final row, mean of task cols (exclude ImageNet)
    last_row = rows[-1]
    last_vals = [float(last_row[c]) for c in task_cols if last_row.get(c, "")]
    last = sum(last_vals) / len(last_vals) if last_vals else None

    # Avg = mean over all task-col x row cells (ZSCL paper definition)
    all_vals = []
    for r in rows:
        for c in task_cols:
            v = r.get(c, "")
            if v:
                all_vals.append(float(v))
    avg = sum(all_vals) / len(all_vals) if all_vals else None

    # Transfer = mean ImageNet across all rows
    in_vals = [float(r["ImageNet"]) for r in rows if r.get("ImageNet", "")]
    transfer = sum(in_vals) / len(in_vals) if in_vals else None

    return {
        "last": last,
        "avg": avg,
        "transfer": transfer,
        "n_rows": len(rows),
        "n_task_cols": len(task_cols),
    }


def fmt(x, w=6):
    return f"{x:>{w}.2f}" if x is not None else "  --  "


def main():
    results = []
    for zscl_on, rd_on, run_dir, label in CELLS:
        path = os.path.join(CKPT_ROOT, run_dir, "task_summary.csv")
        m = metrics_from_csv(path)
        results.append((zscl_on, rd_on, label, m, path))

    print()
    print("=" * 78)
    print("Per-cell metrics")
    print("=" * 78)
    print(f"{'Config':<40}  {'Last':>6}  {'Avg':>6}  {'Transfer':>8}  {'rows':>4}")
    print("-" * 78)
    for _, _, label, m, path in results:
        if m is None:
            print(f"{label:<40}  MISSING   ({path})")
        else:
            partial = "" if m["n_rows"] >= 11 else f"  [partial, {m['n_rows']}/11]"
            print(
                f"{label:<40}  {fmt(m['last'])}  {fmt(m['avg'])}  "
                f"{fmt(m['transfer'], 8)}  {m['n_rows']:>4}{partial}"
            )

    # ---- 2x2 Transfer cell matrix ----
    grid = {(z, r): (m["transfer"] if m else None) for z, r, _, m, _ in results}
    print()
    print("=" * 78)
    print("Transfer (%) — 2x2 substitution matrix")
    print("=" * 78)
    print(f"{'':<14}  {'no RD':>10}  {'with RD':>10}  {'Δ (RD effect)':>14}")
    print("-" * 60)
    for z_on, z_label in [(True, "ZSCL on"), (False, "ZSCL off")]:
        no_rd = grid.get((z_on, False))
        with_rd = grid.get((z_on, True))
        delta = (with_rd - no_rd) if (no_rd is not None and with_rd is not None) else None
        print(
            f"{z_label:<14}  {fmt(no_rd, 10)}  {fmt(with_rd, 10)}  {fmt(delta, 14)}"
        )

    # ---- Verdict on substitution claim ----
    a = grid.get((True,  False))   # ZSCL on,  RD off
    b = grid.get((False, False))   # ZSCL off, RD off  -- baseline collapse
    c = grid.get((False, True))    # ZSCL off, RD on   -- RD recovery
    if None not in (a, b, c):
        zscl_drop = a - b           # how much ZSCL was contributing
        rd_recovery = c - b         # how much RD recovers without ZSCL
        ratio = (rd_recovery / zscl_drop) if zscl_drop > 0 else None
        print()
        print("=" * 78)
        print("Substitution claim verdict")
        print("=" * 78)
        print(f"  ZSCL contribution to Transfer:  {a:.2f} - {b:.2f} = {zscl_drop:+.2f}")
        print(f"  RD recovery without ZSCL:       {c:.2f} - {b:.2f} = {rd_recovery:+.2f}")
        if ratio is not None:
            print(f"  RD recovers {ratio*100:.0f}% of ZSCL's Transfer contribution")
            if ratio >= 0.7:
                print("  -> STRONG: RD substitutes for the bulk of ZSCL anchoring.")
            elif ratio >= 0.3:
                print("  -> MODERATE: RD provides a meaningful fraction of ZSCL's effect.")
            else:
                print("  -> WEAK: RD does not substitute; fall back to Pareto framing.")
        else:
            print("  ZSCL did not improve Transfer in this comparison "
                  "-- different finding entirely.")


if __name__ == "__main__":
    sys.exit(main())
