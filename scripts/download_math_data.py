"""
Fetch GSM8K and MATH-500, the benchmarks used by the step-entropy paper.
=======================================================================

Why these two
-------------
Li et al., "Making Slow Thinking Faster: Compressing LLM Chain-of-Thought
via Step Entropy" (ICLR 2026, arXiv:2508.03346) evaluates on GSM8K,
MATH-500, AIME 2024 and AIME 2025. GSM8K and MATH-500 are the two with
enough instances to measure on (1319 and 500), so they are what we start
from.

They differ from the sorting benchmark in a way that matters: these are
natural-language problems with a single scalar answer, and the reasoning
is a *chain* rather than a graph. That makes them the right place to test
the entropy criterion in isolation, before lifting it to graph vertices.

Both ship with reference solutions, so the ground truth is exact and no
LLM judge is needed - the same property that made sorting a good choice.

Usage
-----
    python scripts/download_math_data.py --out data/math

Writes CSV with one problem per row, in the same shape the benchmark
runner already expects: an id, the question, and the final answer alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
MATH500_URL = ("https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
               "resolve/main/test.jsonl")


def fetch(url: str) -> list:
    """Download a JSONL file and parse it into a list of dicts."""
    print(f"  fetching {url.split('/')[-1]} ... ", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "btp-got/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8")
    rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    print(f"{len(rows)} rows")
    return rows


# ----------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------
def parse_gsm8k(rows: list) -> list:
    """Extract the final answer, which GSM8K marks with '#### '.

    The reference solution is kept as well: the entropy experiment needs a
    reasoning chain to segment, and having the human-written one gives a
    reference point for how many steps a correct solution actually takes.
    """
    out = []
    for i, r in enumerate(rows):
        sol = r["answer"]
        if "####" not in sol:
            continue
        reasoning, _, final = sol.rpartition("####")
        out.append({
            "id": i,
            "question": r["question"].strip(),
            # Thousands separators appear in GSM8K answers and break float().
            "answer": final.strip().replace(",", ""),
            "reference_steps": reasoning.strip(),
        })
    return out


# ----------------------------------------------------------------------
# MATH-500
# ----------------------------------------------------------------------
BOXED = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str:
    """Pull the content of the last \\boxed{...}, honouring nested braces.

    A regex alone is wrong here: MATH solutions contain \\boxed{\\frac{1}{2}},
    so the first unmatched '}' is not the end of the answer. Walk the string
    and track depth instead.
    """
    starts = [m.end() for m in BOXED.finditer(text)]
    if not starts:
        return ""
    i = starts[-1]
    depth, buf = 1, []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf).strip()


def parse_math500(rows: list) -> list:
    out = []
    for i, r in enumerate(rows):
        ans = (r.get("answer") or "").strip() or extract_boxed(r.get("solution", ""))
        if not ans:
            continue
        out.append({
            "id": i,
            "question": r["problem"].strip(),
            "answer": ans,
            "reference_steps": (r.get("solution") or "").strip(),
            "level": r.get("level", ""),
            "subject": r.get("subject", ""),
        })
    return out


def write_csv(rows: list, path: str, fields: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):>5} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download GSM8K and MATH-500")
    ap.add_argument("--out", default="data/math", help="output directory")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows per dataset (0 = all); also writes a smoke file")
    args = ap.parse_args()

    print("GSM8K")
    try:
        gsm = parse_gsm8k(fetch(GSM8K_URL))
    except Exception as exc:
        sys.exit(f"  FAILED: {exc}\n  If on the cluster, start the network proxy first.")

    print("MATH-500")
    try:
        m500 = parse_math500(fetch(MATH500_URL))
    except Exception as exc:
        sys.exit(f"  FAILED: {exc}")

    fields = ["id", "question", "answer", "reference_steps", "level", "subject"]
    if args.limit:
        gsm, m500 = gsm[: args.limit], m500[: args.limit]
    write_csv(gsm, os.path.join(args.out, "gsm8k_test.csv"), fields)
    write_csv(m500, os.path.join(args.out, "math500_test.csv"), fields)

    # Small files for pipeline checks that must not cost GPU time.
    write_csv(gsm[:20], os.path.join(args.out, "smoke", "gsm8k_20.csv"), fields)
    write_csv(m500[:20], os.path.join(args.out, "smoke", "math500_20.csv"), fields)

    print("\nSanity check - first GSM8K item:")
    print(f"  Q: {gsm[0]['question'][:88]}...")
    print(f"  A: {gsm[0]['answer']}")
    print("First MATH-500 item:")
    print(f"  Q: {m500[0]['question'][:88]}...")
    print(f"  A: {m500[0]['answer']}")


if __name__ == "__main__":
    main()
