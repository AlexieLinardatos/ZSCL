"""
Collect every task_summary.csv under one or more roots into a single .xlsx.

    python mtil/scripts/results_to_excel.py results_sync mtil/ckpt -o results.xlsx

Output workbook:
  * "Summary"  - one row per run: Last, Avg, Transfer [ImageNet],
                 Transfer [upper-tri], Diag, #rows, path.
  * one sheet per run - the full accuracy matrix as written by training.

Metric definitions follow scripts/compute_metrics.py:
  Avg      = mean of the whole trained-task matrix (ImageNet excluded)
  Last     = mean of the final row over trained tasks
  Transfer = mean of the held-out ImageNet column (project convention);
             the canonical ZSCL upper-triangle transfer is reported too.
No third-party packages: the .xlsx is written with zipfile + minimal XML.
"""

import argparse
import csv
import os
import re
import sys
import zipfile
from statistics import mean
from xml.sax.saxutils import escape

META_COLS = {"task_idx", "task_name", "avg"}
HELD_OUT = "ImageNet"


# ---------------------------------------------------------------- grouping ---
# (regex, group label) matched top-to-bottom, first hit wins, against the run's
# relative path. Smoke tests are matched first so they never pollute a family.
# Edit freely: adding a rule is all it takes to re-organise the workbook.
GROUP_RULES = [
    (r"smoke",                      "Smoke tests"),
    (r"robustness/buf_",            "Buffer sweep"),
    (r"featrep_",                   "Feature replay"),
    (r"remind_",                    "REMIND (VQ)"),
    (r"phase3_exemplar_",           "Exemplar selection"),
    (r"phase3_ext\d",               "Extensions 1-3"),
    (r"phase3_multi_teacher",       "Multi-teacher merge"),
    (r"phase3_llm_anchor",          "LLM anchor (superseded)"),
    (r"rebuttal",                   "Rebuttal controls"),
    (r"ablation_zscl_only",         "Baselines"),
    (r"ablation_full",              "Main: pixel replay + ExRD"),
    (r"ablation_",                  "Ablations"),
    (r"phase3_no_lora_seed|orderII", "Seeds / task order"),
    (r"phase3_no_lora",             "Main: pixel replay + ExRD"),
]

# Order the groups appear in the workbook.
DISPLAY_ORDER = [
    "Baselines",
    "Main: pixel replay + ExRD",
    "Seeds / task order",
    "Ablations",
    "Feature replay",
    "REMIND (VQ)",
    "Buffer sweep",
    "Exemplar selection",
    "Extensions 1-3",
    "Multi-teacher merge",
    "LLM anchor (superseded)",
    "Rebuttal controls",
    "Smoke tests",
    "Other",
]


def group_of(name):
    for pat, label in GROUP_RULES:
        if re.search(pat, name):
            return label
    return "Other"


# ---------------------------------------------------------------- metrics ---
def read_run(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    header = list(rows[0].keys())
    task_cols = [c for c in header if c not in META_COLS and c != HELD_OUT]
    order = [r["task_name"] for r in rows]

    def num(row, col):
        v = (row.get(col) or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    matrix = [v for r in rows for c in task_cols if (v := num(r, c)) is not None]
    last = [v for c in task_cols if (v := num(rows[-1], c)) is not None]
    inet = [v for r in rows if (v := num(r, HELD_OUT)) is not None]
    diag = [v for r in rows if r["task_name"] in task_cols
            and (v := num(r, r["task_name"])) is not None]
    upper = [v for i, r in enumerate(rows) for j in range(i + 1, len(order))
             if (v := num(r, order[j])) is not None]

    return {
        "header": header,
        "rows": rows,
        "n_rows": len(rows),
        "n_tasks": len(task_cols),
        "Last": mean(last) if last else None,
        "Avg": mean(matrix) if matrix else None,
        "Transfer [ImageNet]": mean(inet) if inet else None,
        "Transfer [upper-tri]": mean(upper) if upper else None,
        "Diag": mean(diag) if diag else None,
        "complete": len(rows) >= len(task_cols),
    }


def find_runs(roots):
    found = {}
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if "task_summary.csv" in files:
                path = os.path.join(dirpath, "task_summary.csv")
                name = os.path.relpath(dirpath, root).replace(os.sep, "/")
                found.setdefault(path, name if name != "." else os.path.basename(root))
    return dict(sorted(found.items(), key=lambda kv: kv[1]))


# ------------------------------------------------------------- xlsx writer ---
def col_letter(i):
    s = ""
    while i >= 0:
        s = chr(ord("A") + i % 26) + s
        i = i // 26 - 1
    return s


def sheet_xml(rows):
    out = ['<?xml version="1.0" encoding="UTF-8"?>'
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           '<sheetData>']
    for r, row in enumerate(rows, start=1):
        out.append(f'<row r="{r}">')
        for c, val in enumerate(row):
            ref = f"{col_letter(c)}{r}"
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)):
                out.append(f'<c r="{ref}"><v>{val!r}</v></c>')
            else:
                out.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">'
                           f'{escape(str(val))}</t></is></c>')
        out.append('</row>')
    out.append('</sheetData></worksheet>')
    return "".join(out)


def safe_sheet_name(name, used):
    base = re.sub(r"[\[\]:*?/\\]", "-", name)[-31:].strip() or "sheet"
    cand, n = base, 2
    while cand.lower() in used:
        suffix = f"~{n}"
        cand = base[: 31 - len(suffix)] + suffix
        n += 1
    used.add(cand.lower())
    return cand


def write_xlsx(path, sheets):
    """sheets: list of (name, rows-as-lists)."""
    ct = ['<?xml version="1.0" encoding="UTF-8"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
          'officedocument.spreadsheetml.sheet.main+xml"/>']
    wb = ['<?xml version="1.0" encoding="UTF-8"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<sheets>']
    rels = ['<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, (name, rows) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows))
            ct.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType='
                      '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            wb.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/'
                        f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        ct.append('</Types>')
        wb.append('</sheets></workbook>')
        rels.append('</Relationships>')
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
                   '2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml", "".join(wb))
        z.writestr("xl/_rels/workbook.xml.rels", "".join(rels))


# -------------------------------------------------------------------- main ---
METRICS = ["Last", "Avg", "Transfer [ImageNet]", "Transfer [upper-tri]", "Diag"]
SUMMARY_HEADER = ["group", "run", *METRICS, "rows", "tasks", "status", "path"]


def metric_row(name, group, d, path):
    return [group, name,
            *[round(d[m], 4) if d[m] is not None else "" for m in METRICS],
            d["n_rows"], d["n_tasks"],
            "ok" if d["complete"] else "PARTIAL", path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="directories to scan for task_summary.csv")
    ap.add_argument("-o", "--out", default="results.xlsx")
    ap.add_argument("--csv", action="store_true",
                    help="also write <out>.summary.csv (the flat Summary sheet)")
    ap.add_argument("--per-run-sheets", action="store_true",
                    help="one tab per run instead of one tab per group")
    ap.add_argument("--no-matrices", action="store_true",
                    help="group tabs hold only the metric table, no per-task matrices")
    args = ap.parse_args()

    runs = find_runs(args.roots)
    if not runs:
        sys.exit("No task_summary.csv found under: " + ", ".join(args.roots))

    # Read everything, bucket by group.
    buckets = {}
    for path, name in runs.items():
        d = read_run(path)
        if d is None:
            print(f"  skip (empty): {path}")
            continue
        buckets.setdefault(group_of(name), []).append((name, path, d))

    order = [g for g in DISPLAY_ORDER if g in buckets]
    order += [g for g in sorted(buckets) if g not in DISPLAY_ORDER]

    # ---- Summary: flat table, group-ordered (sort/filter/pivot friendly) ----
    summary = [SUMMARY_HEADER]
    for g in order:
        rows = sorted(buckets[g], key=lambda t: t[0])
        for name, path, d in rows:
            summary.append(metric_row(name, g, d, path))

    # ---- Groups: one row per family, aggregates over complete runs ----------
    agg = [["group", "runs", "complete", *[f"mean {m}" for m in METRICS],
            "best Last", "best run"]]
    for g in order:
        rows = buckets[g]
        good = [(n, d) for n, _p, d in rows if d["complete"]]
        means = []
        for m in METRICS:
            vals = [d[m] for _n, d in good if d[m] is not None]
            means.append(round(mean(vals), 4) if vals else "")
        best_n, best_v = "", ""
        cand = [(n, d["Last"]) for n, d in good if d["Last"] is not None]
        if cand:
            best_n, best_v = max(cand, key=lambda t: t[1])
            best_v = round(best_v, 4)
        agg.append([g, len(rows), len(good), *means, best_v, best_n])

    sheets, used = [("Summary", summary), ("Groups", agg)], {"summary", "groups"}

    if args.per_run_sheets:
        for g in order:
            for name, _path, d in sorted(buckets[g], key=lambda t: t[0]):
                sheets.append((safe_sheet_name(name, used), matrix_block(d)))
    else:
        for g in order:
            rows = sorted(buckets[g], key=lambda t: t[0])
            block = [[f"Group: {g}"], [],
                     ["run", *METRICS, "rows", "status"]]
            for name, _path, d in rows:
                block.append([name,
                              *[round(d[m], 4) if d[m] is not None else "" for m in METRICS],
                              d["n_rows"], "ok" if d["complete"] else "PARTIAL"])
            if not args.no_matrices:
                block += [[], ["Per-run accuracy matrices"], []]
                for name, _path, d in rows:
                    block.append([name])
                    block += matrix_block(d)
                    block.append([])
            sheets.append((safe_sheet_name(g, used), block))

    write_xlsx(args.out, sheets)

    # ---- console recap, grouped ----
    for g in order:
        print(f"\n{g}")
        for name, _path, d in sorted(buckets[g], key=lambda t: t[0]):
            print(f"  {name:<44} Last={_f(d['Last'])} Avg={_f(d['Avg'])} "
                  f"Transfer={_f(d['Transfer [ImageNet]'])}"
                  f"{'' if d['complete'] else '  [PARTIAL]'}")
    total = sum(len(v) for v in buckets.values())
    print(f"\nWrote {args.out}  ({total} runs, {len(order)} groups, "
          f"{len(sheets)} sheets)")

    if args.csv:
        cpath = os.path.splitext(args.out)[0] + ".summary.csv"
        with open(cpath, "w", newline="") as f:
            csv.writer(f).writerows(summary)
        print(f"Wrote {cpath}")


def matrix_block(d):
    out = [d["header"]]
    for r in d["rows"]:
        out.append([_maybe_num(r.get(c, "")) for c in d["header"]])
    return out


def _maybe_num(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return v


def _f(v):
    return f"{v:6.2f}" if v is not None else "   n/a"


if __name__ == "__main__":
    main()
