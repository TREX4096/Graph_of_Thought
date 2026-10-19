"""
Re-grade saved results with the current scorer.
===============================================

Every run writes the model's raw answer alongside the truth, so a scorer
improvement can be applied to results already on disk. No GPU time, and -
more importantly - no new sampling, so the comparison is exact: the only
thing that changed is the grading rule.

    python scripts/regrade.py --dir results/entropy_r1_20261018_104735

Prints the old and new accuracy for every (dataset, strategy, ratio) cell
and writes <name>_regraded.csv beside the original.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from got.tasks.math_reasoning import answers_match


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-grade saved results")
    ap.add_argument("--dir", required=True, help="a results/ run directory")
    ap.add_argument("--write", action="store_true",
                    help="write *_regraded.csv next to each input")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*_instances.csv")))
    if not files:
        sys.exit(f"no *_instances.csv under {args.dir}")

    for path in files:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        name = os.path.basename(path).replace("_instances.csv", "")
        print(f"\n=== {name}  ({len(rows)} rows) ===")
        print(f"{'strategy':<9} {'ratio':>6} {'old':>8} {'new':>8} {'delta':>8}")
        print("-" * 44)

        cells = collections.defaultdict(lambda: [0, 0, 0])   # old, new, n
        for r in rows:
            key = (r["strategy"], r["ratio"])
            new = answers_match(r["answer"], r["truth"])
            cells[key][0] += int(r["correct"])
            cells[key][1] += int(new)
            cells[key][2] += 1
            r["correct_regraded"] = int(new)

        total_old = total_new = total_n = 0
        for (strat, ratio), (o, n, cnt) in sorted(cells.items()):
            print(f"{strat:<9} {ratio:>6} {o/cnt:8.1%} {n/cnt:8.1%} "
                  f"{(n-o)/cnt:+8.1%}")
            total_old += o; total_new += n; total_n += cnt
        print("-" * 44)
        print(f"{'OVERALL':<9} {'':>6} {total_old/total_n:8.1%} "
              f"{total_new/total_n:8.1%} {(total_new-total_old)/total_n:+8.1%}")

        if args.write:
            out = path.replace("_instances.csv", "_regraded.csv")
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  wrote {out}")

    have_full = any(r["strategy"] == "full" for r in rows)
    if not have_full:
        print("\nNote: this run predates per-instance baseline logging, so the")
        print("'full' and 'none' rows are absent and only the PRUNED cells are")
        print("re-graded. Margins measured against that baseline are therefore")
        print("flattering - the baseline would rise too. Rerun to fix.")


if __name__ == "__main__":
    main()
