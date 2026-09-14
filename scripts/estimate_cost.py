"""
Price an HPC job BEFORE submitting it.
======================================

Why this exists
---------------
GPU hours on a cluster are expensive, and a GoT run's cost is not obvious by
inspection: it is the product of instances, schemes, branching factors,
aggregation attempts and per-step token budgets, and those multiply fast.
Submitting a job to find out what it costs is the wrong order of operations.

This script runs the **full Graph of Operations with the free mock backend**,
which executes the exact same control flow and issues the exact same number of
prompts as a real run would -- it simply doesn't need a GPU to do it. From the
recorded call shapes it projects:

  * total prompts, GPU round trips (batches), and sequences,
  * prefill tokens (with and without prefix caching),
  * decode tokens, both worst case (every call hits its cap) and expected,
  * GPU seconds and wall-clock time at a given throughput,
  * money, at a given hourly rate.

Everything is an estimate, but the *counts* are exact -- only throughput is
modelled. That makes it reliable for the decision it is meant to support:
"is this configuration affordable, and which knob do I turn if not?"

Usage
-----
    # Price the default full run
    python scripts/estimate_cost.py --data data/sorting/sorting_64.csv

    # Compare configurations before committing
    python scripts/estimate_cost.py --data data/sorting/sorting_64.csv \\
        --schemes got --aggregation-attempts 10 5 3 --limit 100

    # With your cluster's actual rate
    python scripts/estimate_cost.py --data data/sorting/sorting_64.csv \\
        --gpu-hour-rate 250 --currency INR
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, List

from got import Controller
from got.backends import MockLM
from got.tasks.sorting import SortingParser, SortingPrompter
from got.tasks.sorting.graphs import SCHEMES


# ----------------------------------------------------------------------
# Throughput model
# ----------------------------------------------------------------------
# Output tokens/second for a *well-batched* vLLM server. These are
# order-of-magnitude figures for a single modern data-centre GPU; they vary
# with hardware, quantisation and batch size, so override with --throughput
# once you have measured your own cluster (the benchmark runner reports
# wall_seconds and total_tokens per instance, which gives you the number).
THROUGHPUT_TOK_S = {
    "1.5b": 6000,
    "7b": 2500,
    "8b": 2200,
    "13b": 1400,
    "32b": 700,
    "70b": 350,
}

# Fraction of the token budget a call actually uses on average. Our prompts
# demand a bare list and we set stop strings, so most calls finish well short
# of their cap. Used for the "expected" projection; the worst case assumes 1.0.
DEFAULT_FILL_RATIO = 0.45


def profile_instance(
    scheme: str, numbers: List[int], args
) -> Dict[str, Any]:
    """Run one instance on the mock backend and return its call profile."""
    lm = MockLM(error_rate=0.15, seed=0)
    lm.record_calls = True

    builder = SCHEMES[scheme]
    if scheme == "got":
        leaves = builder(numbers, num_chunks=args.num_chunks,
                         branching_factor=args.branching_factor,
                         aggregation_attempts=args.aggregation_attempts_current)
    elif scheme == "tot":
        leaves = builder(numbers, branching_factor=args.branching_factor,
                         depth=args.tot_depth)
    elif scheme == "cot_sc":
        leaves = builder(numbers, k=args.branching_factor)
    else:
        leaves = builder(numbers)

    ctrl = Controller(lm, SortingPrompter(), SortingParser(), leaves)
    ctrl.run(num_chunks=args.num_chunks)

    sequences = sum(c["n_prompts"] * c["num_responses"] for c in lm.call_log)
    worst_decode = sum(
        c["n_prompts"] * c["num_responses"] * c["max_tokens"] for c in lm.call_log
    )
    prefill = sum(c["prompt_tokens"] for c in lm.call_log)

    # With prefix caching, the shared instruction + few-shot prefix is
    # computed once per distinct prefix rather than per prompt. Our prompts
    # share a long prefix, so we approximate the saving as the prefill of all
    # but the first prompt in each batch.
    prefill_cached = sum(
        c["prompt_tokens"] / max(1, c["n_prompts"]) for c in lm.call_log
    )

    return {
        "prompts": sum(c["n_prompts"] for c in lm.call_log),
        "batches": len(lm.call_log),
        "sequences": sequences,
        "prefill_tokens": prefill,
        "prefill_tokens_cached": prefill_cached,
        "worst_decode_tokens": worst_decode,
        "thoughts": ctrl.graph_summary()["n_thoughts"],
    }


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h < 1:
        return f"{seconds / 60:.1f} min"
    return f"{h:.2f} h"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Estimate HPC cost of a GoT run before submitting it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--schemes", nargs="+",
                    default=["io", "cot", "cot_sc", "tot", "got"])
    ap.add_argument("--limit", type=int, default=100,
                    help="instances to run (0 = whole file)")
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--branching-factor", type=int, default=3)
    ap.add_argument("--aggregation-attempts", type=int, nargs="+", default=[10],
                    help="one or more values to compare side by side")
    ap.add_argument("--tot-depth", type=int, default=3)

    ap.add_argument("--model-size", default="8b", choices=list(THROUGHPUT_TOK_S))
    ap.add_argument("--throughput", type=float, default=0,
                    help="output tokens/sec; overrides --model-size")
    ap.add_argument("--fill-ratio", type=float, default=DEFAULT_FILL_RATIO,
                    help="fraction of the token budget a call typically uses")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-hour-rate", type=float, default=0,
                    help="cost per GPU-hour; 0 hides the money column")
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--prefix-caching", action="store_true", default=True)
    args = ap.parse_args()

    with open(args.data, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    n_instances = len(rows)

    # One representative instance is enough: the GoO shape is identical for
    # every instance of a given length, so counts scale linearly.
    sample = json.loads(rows[0]["input"])

    tok_s = args.throughput or THROUGHPUT_TOK_S[args.model_size]
    tok_s *= args.n_gpus

    print(f"Dataset      : {args.data}")
    print(f"Instances    : {n_instances}   (input length {len(sample)})")
    print(f"Model        : {args.model_size} @ ~{tok_s:,.0f} output tok/s "
          f"across {args.n_gpus} GPU(s)")
    print(f"Fill ratio   : {args.fill_ratio:.0%} of token budget used on average")
    print(f"Prefix cache : {'on' if args.prefix_caching else 'off'}")
    print()

    header = (f"{'scheme':9s} {'agg_k':>6s} {'batches':>8s} {'seqs':>8s} "
              f"{'decode tok':>12s} {'GPU time':>11s}")
    if args.gpu_hour_rate:
        header += f" {'cost':>12s}"
    print(header)
    print("-" * (len(header) + 2))

    grand_seconds = 0.0

    for scheme in args.schemes:
        agg_values = args.aggregation_attempts if scheme == "got" else [0]
        for agg in agg_values:
            args.aggregation_attempts_current = agg or 1
            prof = profile_instance(scheme, sample, args)

            prefill = (prof["prefill_tokens_cached"] if args.prefix_caching
                       else prof["prefill_tokens"])
            decode = prof["worst_decode_tokens"] * args.fill_ratio

            # Prefill is far cheaper per token than decode (it is a single
            # parallel forward pass, not a sequential loop). Charging it at
            # ~10x decode throughput is a reasonable working approximation.
            seconds_per_inst = decode / tok_s + prefill / (tok_s * 10)
            total_seconds = seconds_per_inst * n_instances
            grand_seconds += total_seconds

            line = (f"{scheme:9s} {agg if agg else '-':>6} "
                    f"{prof['batches'] * n_instances:8,d} "
                    f"{prof['sequences'] * n_instances:8,d} "
                    f"{decode * n_instances:12,.0f} "
                    f"{fmt_hours(total_seconds):>11s}")
            if args.gpu_hour_rate:
                cost = total_seconds / 3600 * args.gpu_hour_rate * args.n_gpus
                line += f" {cost:12,.0f}"
            print(line)

    print("-" * (len(header) + 2))
    total_line = f"{'TOTAL':9s} {'':>6} {'':>8} {'':>8} {'':>12} " \
                 f"{fmt_hours(grand_seconds):>11s}"
    if args.gpu_hour_rate:
        total_cost = grand_seconds / 3600 * args.gpu_hour_rate * args.n_gpus
        total_line += f" {total_cost:12,.0f}"
    print(total_line)

    if args.gpu_hour_rate:
        print(f"\n(currency: {args.currency}, at {args.gpu_hour_rate:,.0f}"
              f"/GPU-hour x {args.n_gpus} GPU(s))")

    print("""
How to bring the number down, in order of impact:
  1. --aggregation-attempts : the dominant GoT cost. Try 10 -> 5 -> 3 above
                              and compare against the quality you lose.
  2. --limit                : fewer instances. 30 is usually enough to see a
                              clear separation between schemes.
  3. --branching-factor     : linear in cost.
  4. Shorter inputs         : run sorting_32 before sorting_64/128.
  5. Drop schemes           : io/cot/cot_sc are cheap; tot and got dominate.

Counts (batches, sequences) are exact. Only throughput is modelled, so
calibrate --throughput from a short pilot run before trusting the money column.
""")


if __name__ == "__main__":
    main()
