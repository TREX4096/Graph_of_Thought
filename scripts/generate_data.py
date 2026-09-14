"""
Generate the sample datasets used to validate the implementation.
=================================================================

Where the paper's data comes from
---------------------------------
The GoT paper does not ship fixed benchmark files for sorting / set
intersection / keyword counting -- it *generates* them, because the tasks are
synthetic by construction:

  * **Sorting** (Sec 5.1): "sorting numbers 0-9 with duplicates" at lengths
    32, 64 and 128.
  * **Set intersection** (Sec 5.2): "different set sizes of 32, 64 and 128
    elements and we vary the number of elements found in both sets to be
    between 25% and 75%".
  * **Keyword counting** (Sec 5.3): passages containing country mentions.
  * **Document merging** (Sec 5.4): overlapping NDA documents.

So generating them here is faithful to the paper, not a shortcut. It also
means the datasets are reproducible: a fixed seed gives byte-identical files,
which matters for comparing runs across the laptop and the HPC.

Usage
-----
    python scripts/generate_data.py --out data --seed 42

Produces CSV files with one problem instance per row, plus the known correct
answer so the harness can compute ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import List, Tuple


# Countries used by the keyword-counting task. Kept in sync with the list
# MockLM recognises, so the mock backend can simulate this task sensibly.
COUNTRIES = [
    "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany",
    "Italy", "Spain", "Norway", "Sweden", "Japan", "China", "India",
    "Australia", "Egypt", "Kenya", "Peru", "Chile", "Poland", "Greece",
]

SENTENCE_TEMPLATES = [
    "The delegation travelled from {a} to {b} before the summit.",
    "Exports from {a} rose sharply while {b} reported a decline.",
    "Researchers in {a} collaborated with a team based in {b}.",
    "After leaving {a}, the group spent three weeks in {b}.",
    "{a} and {b} signed a joint statement on trade.",
    "The documentary contrasted life in {a} with that in {b}.",
]


def gen_sorting(n_samples: int, length: int, rng: random.Random) -> List[dict]:
    """Random lists of digits 0-9 with duplicates -- the paper's setup."""
    rows = []
    for i in range(n_samples):
        nums = [rng.randint(0, 9) for _ in range(length)]
        rows.append({
            "id": i,
            "length": length,
            "input": json.dumps(nums),
            "answer": json.dumps(sorted(nums)),
        })
    return rows


def gen_set_intersection(
    n_samples: int, size: int, rng: random.Random
) -> List[dict]:
    """Two sets of ``size`` elements with a controlled overlap fraction.

    The paper varies overlap between 25% and 75%; we sample the fraction
    uniformly in that range per instance so the dataset spans the difficulty
    band rather than sitting at one point.
    """
    rows = []
    # Draw from a universe comfortably larger than the sets so that
    # non-overlapping elements really are distinct.
    universe = list(range(size * 4))
    for i in range(n_samples):
        overlap_frac = rng.uniform(0.25, 0.75)
        n_overlap = int(size * overlap_frac)

        pool = rng.sample(universe, size * 2)
        shared = pool[:n_overlap]
        rest_a = pool[n_overlap : n_overlap + (size - n_overlap)]
        rest_b = pool[n_overlap + (size - n_overlap) : n_overlap + 2 * (size - n_overlap)]

        set_a = shared + rest_a
        set_b = shared + rest_b
        rng.shuffle(set_a)
        rng.shuffle(set_b)

        answer = sorted(set(set_a) & set(set_b))
        rows.append({
            "id": i,
            "size": size,
            "set_a": json.dumps(set_a),
            "set_b": json.dumps(set_b),
            "overlap_fraction": round(overlap_frac, 3),
            "answer": json.dumps(answer),
        })
    return rows


def gen_keyword_counting(
    n_samples: int, n_sentences: int, rng: random.Random
) -> List[dict]:
    """Passages with a known, exact country-mention count.

    Because we build the text from templates we know the true counts without
    any parsing ambiguity -- which is what makes this a clean benchmark.
    """
    rows = []
    for i in range(n_samples):
        counts: dict = {}
        sentences = []
        for _ in range(n_sentences):
            a, b = rng.sample(COUNTRIES, 2)
            template = rng.choice(SENTENCE_TEMPLATES)
            sentences.append(template.format(a=a, b=b))
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1

        rows.append({
            "id": i,
            "n_sentences": n_sentences,
            "input": " ".join(sentences),
            "answer": json.dumps(dict(sorted(counts.items()))),
        })
    return rows


# Clause bank for the document-merging task. Each NDA draws a subset, so the
# documents genuinely overlap -- the condition the paper's task requires
# ("several input ones that partially overlap in terms of their contents").
NDA_CLAUSES = [
    "The Receiving Party shall hold all Confidential Information in strict confidence.",
    "Confidential Information does not include information that is publicly available.",
    "This Agreement shall remain in effect for a period of three (3) years.",
    "The Receiving Party shall not disclose Confidential Information to third parties.",
    "Upon termination, all materials shall be returned or destroyed.",
    "This Agreement is governed by the laws of the stated jurisdiction.",
    "Nothing in this Agreement grants any licence under intellectual property rights.",
    "The Receiving Party shall limit access to employees with a need to know.",
    "Any amendment to this Agreement must be made in writing and signed by both parties.",
    "Neither party is obliged to enter into any further business relationship.",
    "The Disclosing Party makes no warranty as to the accuracy of the information.",
    "Breach of this Agreement may cause irreparable harm warranting injunctive relief.",
]


def gen_document_merging(
    n_samples: int, n_docs: int, rng: random.Random
) -> List[dict]:
    """Several partially-overlapping NDA documents per instance."""
    rows = []
    for i in range(n_samples):
        docs = []
        used = set()
        for d in range(n_docs):
            k = rng.randint(5, 8)
            clauses = rng.sample(NDA_CLAUSES, k)
            used.update(clauses)
            docs.append(f"NDA DOCUMENT {d + 1}\n" + "\n".join(clauses))

        rows.append({
            "id": i,
            "n_docs": n_docs,
            "documents": json.dumps(docs),
            # There is no single correct merge, so we record the set of
            # distinct clauses: an ideal merge retains all of them exactly once.
            "expected_clauses": json.dumps(sorted(used)),
        })
    return rows


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):4d} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GoT benchmark datasets")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    ap.add_argument(
        "--n-samples", type=int, default=100,
        help="instances per configuration (paper uses 100)",
    )
    ap.add_argument(
        "--small", action="store_true",
        help="also emit tiny 5-instance files for fast local smoke tests",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Sorting:")
    for length in (32, 64, 128):
        write_csv(
            os.path.join(args.out, "sorting", f"sorting_{length}.csv"),
            gen_sorting(args.n_samples, length, rng),
        )

    print("Set intersection:")
    for size in (32, 64, 128):
        write_csv(
            os.path.join(args.out, "set_intersection", f"set_intersection_{size}.csv"),
            gen_set_intersection(args.n_samples, size, rng),
        )

    print("Keyword counting:")
    for n_sent in (4, 8, 16):
        write_csv(
            os.path.join(args.out, "keyword_counting", f"keyword_counting_{n_sent}.csv"),
            gen_keyword_counting(args.n_samples, n_sent, rng),
        )

    print("Document merging:")
    write_csv(
        os.path.join(args.out, "document_merging", "document_merging_4.csv"),
        gen_document_merging(args.n_samples, 4, rng),
    )

    if args.small:
        print("Smoke-test subsets (5 instances each):")
        rng_small = random.Random(args.seed + 1)
        write_csv(
            os.path.join(args.out, "smoke", "sorting_32_small.csv"),
            gen_sorting(5, 32, rng_small),
        )
        write_csv(
            os.path.join(args.out, "smoke", "set_intersection_32_small.csv"),
            gen_set_intersection(5, 32, rng_small),
        )
        write_csv(
            os.path.join(args.out, "smoke", "keyword_counting_4_small.csv"),
            gen_keyword_counting(5, 4, rng_small),
        )

    print("\nDone. Datasets are deterministic for a given --seed.")


if __name__ == "__main__":
    main()
