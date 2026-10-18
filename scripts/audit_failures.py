"""
Audit graded-wrong answers for scorer strictness.
=================================================

Our MATH-500 baseline came in well below the step-entropy paper's. Before
treating that as a reasoning gap, check whether it is a *grading* gap: a
scorer that marks 11\\sqrt2 wrong against 11\\sqrt{2} understates accuracy
everywhere, and understates it unevenly between full and compressed chains
because compression changes how tersely the model formats its answer.

The method is to re-grade every wrong answer under progressively looser
rules and report how many flip. Anything that flips under a rule we would
accept as correct is a scoring artefact, not a model error.

    python scripts/audit_failures.py --csv results/<run>/math500_test_instances.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from got.tasks.math_reasoning.scoring import answers_match, normalise_answer


def strip_braces(s: str) -> str:
    """\\sqrt{2} -> \\sqrt2 : brace style should never decide correctness."""
    return re.sub(r"\{([^{}]*)\}", r"\1", s)


def strip_latex_cmds(s: str) -> str:
    """Drop command backslashes so sqrt2 == \\sqrt2."""
    return re.sub(r"\\([a-zA-Z]+)", r"\1", s)


def digits_only(s: str) -> str:
    return re.sub(r"[^0-9.\-]", "", s)


RULES = [
    ("brace style", lambda p, t: strip_braces(p) == strip_braces(t)),
    ("latex commands", lambda p, t: strip_latex_cmds(strip_braces(p))
                                    == strip_latex_cmds(strip_braces(t))),
    # NOTE: a "substring" rule was tried here and removed. It flipped
    # \dfrac{\pi}{3} against \pi, 7 against "3, 5, 7" and -2 against "1,-2" -
    # a wrong value, an incomplete answer and a missed root, none of them
    # formatting. It would have overstated the corrected baseline by about
    # five points. A looser scorer that accepts those is worse than a strict
    # one that rejects a few valid answers.
    ("digits only", lambda p, t: bool(digits_only(t))
                                 and digits_only(p) == digits_only(t)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit wrong answers for scorer strictness")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--strategy", default=None, help="limit to one strategy")
    ap.add_argument("--ratio", default=None, help="limit to one ratio")
    ap.add_argument("--show", type=int, default=12, help="examples to print")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.strategy:
        rows = [r for r in rows if r["strategy"] == args.strategy]
    if args.ratio:
        rows = [r for r in rows if r["ratio"] == args.ratio]
    if not rows:
        sys.exit("no rows after filtering")

    wrong = [r for r in rows if r["correct"] == "0"]
    print(f"rows {len(rows)}   graded wrong {len(wrong)} "
          f"({len(wrong)/len(rows):.1%})\n")

    empty = [r for r in wrong if not r["answer"].strip()]
    print(f"  empty answer (nothing extracted) : {len(empty):>4}  "
          f"{len(empty)/max(1,len(wrong)):.1%} of wrong")

    flips = collections.Counter()
    examples = collections.defaultdict(list)
    for r in wrong:
        p, t = normalise_answer(r["answer"]), normalise_answer(r["truth"])
        if not p:
            continue
        for name, rule in RULES:
            try:
                if rule(p, t):
                    flips[name] += 1
                    if len(examples[name]) < 4:
                        examples[name].append((r["answer"], r["truth"]))
                    break           # count each row under its loosest-needed rule
            except re.error:
                pass

    print("\n  would flip to CORRECT under a looser rule:")
    for name, _ in RULES:
        n = flips[name]
        print(f"    {name:<16} {n:>4}  {n/max(1,len(wrong)):>6.1%} of wrong")
    total = sum(flips.values())
    print(f"    {'TOTAL':<16} {total:>4}  {total/max(1,len(wrong)):>6.1%} of wrong")

    if total:
        print("\n  examples:")
        for name, _ in RULES:
            for got_a, truth in examples[name][:2]:
                print(f"    [{name}] got {got_a[:34]!r} vs truth {truth[:34]!r}")

    n_rows = len(rows)
    base = sum(1 for r in rows if r["correct"] == "1")
    print(f"\n  accuracy as graded        : {base/n_rows:.1%}")
    print(f"  accuracy if all flipped   : {(base+total)/n_rows:.1%}")
    print(f"  genuine errors (no flip)  : {len(wrong)-total} "
          f"({(len(wrong)-total)/n_rows:.1%} of all)")

    if total / max(1, len(wrong)) > 0.15:
        print("\n  >> Scoring is materially strict. The baseline understates")
        print("     accuracy, and the understatement may differ between full")
        print("     and compressed chains. Tighten the scorer before quoting.")
    else:
        print("\n  >> Scoring looks sound; the gap is not a formatting artefact.")


if __name__ == "__main__":
    main()
