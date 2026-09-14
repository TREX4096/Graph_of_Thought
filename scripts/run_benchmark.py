"""
Benchmark runner -- the main entry point for local and HPC experiments.
=======================================================================

Runs one or more prompting schemes over a dataset and records, per instance:

  * the ground-truth correctness of the final answer,
  * the paper's error-scope score,
  * LLM calls and token cost,
  * graph statistics (thoughts, aggregations, volume, latency).

Those are exactly the axes the GoT paper compares on (Section 7): quality vs.
cost, with the latency-volume analysis alongside.

Examples
--------
Fast local sanity check with the free mock backend::

    python scripts/run_benchmark.py --task sorting --schemes io cot_sc tot got \\
        --data data/smoke/sorting_32_small.csv --backend mock --out results/smoke

Real local run against a small quantised model::

    python scripts/run_benchmark.py --task sorting --schemes got \\
        --data data/sorting/sorting_32.csv --limit 10 \\
        --backend llamacpp --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
        --out results/local

On the HPC with vLLM::

    python scripts/run_benchmark.py --task sorting --schemes io cot cot_sc tot got \\
        --data data/sorting/sorting_64.csv \\
        --backend vllm --model-id meta-llama/Llama-3.1-8B-Instruct \\
        --out results/hpc

Output
------
Two files per run: a per-instance CSV (easy to load in pandas) and a JSON
summary. Results are appended per scheme so a crashed HPC job can be resumed
by rerunning only the missing schemes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

from tqdm import tqdm

from got import Controller, graph_metrics
from got.backends import get_backend
from got.tasks.sorting import SortingParser, SortingPrompter
from got.tasks.sorting.graphs import SCHEMES as SORTING_SCHEMES
from got.tasks.sorting.scoring import sorting_error_scope


# ----------------------------------------------------------------------
# Task registry
# ----------------------------------------------------------------------
# Each task supplies: how to read a dataset row, how to build a GoO, and how
# to grade a result. Adding a task means adding one entry here.
def _load_sorting_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {"numbers": json.loads(row["input"]), "answer": json.loads(row["answer"])}


TASKS = {
    "sorting": {
        "schemes": SORTING_SCHEMES,
        "prompter": SortingPrompter,
        "parser": SortingParser,
        "load_row": _load_sorting_row,
    },
}


def build_backend(args) -> Any:
    """Instantiate the requested backend from CLI arguments."""
    common = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.backend == "mock":
        return get_backend("mock", error_rate=args.mock_error_rate,
                           seed=args.seed, **common)
    if args.backend == "llamacpp":
        if not args.model_path:
            sys.exit("--model-path is required for --backend llamacpp")
        return get_backend("llamacpp", model_path=args.model_path,
                           n_ctx=args.n_ctx, **common)
    if args.backend in ("hf", "vllm"):
        if not args.model_id:
            sys.exit(f"--model-id is required for --backend {args.backend}")
        kwargs = {"model_id": args.model_id, **common}
        if args.backend == "vllm":
            kwargs["tensor_parallel_size"] = args.tensor_parallel_size
            kwargs["max_model_len"] = args.n_ctx
        else:
            kwargs["load_in_4bit"] = args.load_in_4bit
        return get_backend(args.backend, **kwargs)
    sys.exit(f"unknown backend {args.backend}")


def run_one(
    scheme: str, task_cfg: Dict[str, Any], instance: Dict[str, Any], lm, args
) -> Dict[str, Any]:
    """Execute a single (scheme, instance) pair and collect its metrics."""
    numbers = instance["numbers"]
    builder = task_cfg["schemes"][scheme]

    # GoT and ToT take structural hyper-parameters; the simpler baselines
    # take none. Keeping this explicit avoids silently passing a branching
    # factor to a scheme that has no branches.
    if scheme == "got":
        leaves = builder(
            numbers,
            num_chunks=args.num_chunks,
            branching_factor=args.branching_factor,
            aggregation_attempts=args.aggregation_attempts,
        )
    elif scheme == "tot":
        leaves = builder(numbers, branching_factor=args.branching_factor,
                         depth=args.tot_depth)
    elif scheme == "cot_sc":
        leaves = builder(numbers, k=args.branching_factor)
    else:
        leaves = builder(numbers)

    # Reset per instance so token cost is attributable to this instance alone.
    lm.reset_usage()
    ctrl = Controller(lm, task_cfg["prompter"](), task_cfg["parser"](), leaves)

    start = time.time()
    try:
        ctrl.run(num_chunks=args.num_chunks)
        failed = False
        error_msg = ""
    except Exception as exc:          # noqa: BLE001 - never lose a whole HPC job
        # One bad instance must not kill a multi-hour run. Record and move on.
        logging.exception("instance failed under scheme %s", scheme)
        failed, error_msg = True, str(exc)
    wall = time.time() - start

    best = ctrl.best_thought() if not failed else None
    produced = best.state.get("current", []) if best else []
    gm = graph_metrics(ctrl.all_thoughts()) if not failed else {}
    summary = ctrl.graph_summary() if not failed else {}

    return {
        "scheme": scheme,
        "failed": failed,
        "error": error_msg,
        "correct": bool(produced == instance["answer"]),
        "error_scope": sorting_error_scope(numbers, produced),
        "score": best.score if best else 0.0,
        "max_score": len(numbers),
        "n_llm_calls": lm.usage.n_calls,
        # Batches are GPU round trips and therefore what maps to cluster cost;
        # n_llm_calls is prompts served. Their ratio is the mean batch size.
        "n_batches": lm.usage.n_batches,
        "mean_batch_size": round(lm.usage.mean_batch_size, 2),
        "prompt_tokens": lm.usage.prompt_tokens,
        "completion_tokens": lm.usage.completion_tokens,
        "total_tokens": lm.usage.total_tokens,
        "n_thoughts": summary.get("n_thoughts", 0),
        "n_aggregations": summary.get("n_aggregations", 0),
        "max_volume": gm.get("max_volume", 0),
        "max_latency": gm.get("max_latency", 0),
        "wall_seconds": round(wall, 3),
        "output": json.dumps(produced),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run GoT benchmarks", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--task", default="sorting", choices=list(TASKS))
    ap.add_argument("--data", required=True, help="CSV produced by generate_data.py")
    ap.add_argument("--schemes", nargs="+", default=["io", "cot", "cot_sc", "tot", "got"])
    ap.add_argument("--limit", type=int, default=0, help="cap instances (0 = all)")
    ap.add_argument("--out", default="results", help="output directory")

    # Backend selection
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "llamacpp", "hf", "vllm"])
    ap.add_argument("--model-path", help="GGUF file (llamacpp)")
    ap.add_argument("--model-id", help="HF model id (hf / vllm)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--mock-error-rate", type=float, default=0.15)

    # GoT / ToT structure (defaults are the paper's Figure 4 values)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--branching-factor", type=int, default=3)
    ap.add_argument("--aggregation-attempts", type=int, default=10)
    ap.add_argument("--tot-depth", type=int, default=3)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    task_cfg = TASKS[args.task]

    # --- Load dataset ------------------------------------------------
    with open(args.data, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    instances = [task_cfg["load_row"](r) for r in rows]

    print(f"Task      : {args.task}")
    print(f"Dataset   : {args.data}  ({len(instances)} instances)")
    print(f"Backend   : {args.backend}")
    print(f"Schemes   : {', '.join(args.schemes)}")
    print()

    lm = build_backend(args)

    os.makedirs(args.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.data))[0]
    csv_path = os.path.join(args.out, f"{args.task}_{tag}_{args.backend}.csv")
    json_path = os.path.join(args.out, f"{args.task}_{tag}_{args.backend}_summary.json")

    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}

    for scheme in args.schemes:
        if scheme not in task_cfg["schemes"]:
            print(f"  skipping unknown scheme {scheme!r}")
            continue

        results = []
        for i, inst in enumerate(
            tqdm(instances, desc=f"{scheme:7s}", unit="inst")
        ):
            rec = run_one(scheme, task_cfg, inst, lm, args)
            rec["instance_id"] = i
            results.append(rec)
            all_rows.append(rec)

        ok = [r for r in results if not r["failed"]]
        n = max(1, len(ok))
        summaries[scheme] = {
            "n_instances": len(results),
            "n_failed": sum(r["failed"] for r in results),
            "accuracy": sum(r["correct"] for r in ok) / n,
            "mean_error_scope": sum(r["error_scope"] for r in ok) / n,
            "mean_total_tokens": sum(r["total_tokens"] for r in ok) / n,
            "mean_llm_calls": sum(r["n_llm_calls"] for r in ok) / n,
            "mean_batches": sum(r["n_batches"] for r in ok) / n,
            "mean_max_volume": sum(r["max_volume"] for r in ok) / n,
            "mean_max_latency": sum(r["max_latency"] for r in ok) / n,
            "mean_wall_seconds": sum(r["wall_seconds"] for r in ok) / n,
        }

    # --- Persist -----------------------------------------------------
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"config": vars(args), "summaries": summaries}, fh, indent=2, default=str
        )

    # --- Report ------------------------------------------------------
    print(f"\n{'scheme':8s} {'acc':>7s} {'err':>8s} {'tokens':>10s} "
          f"{'calls':>7s} {'batch':>7s} {'vol':>6s} {'lat':>6s}")
    print("-" * 66)
    for scheme, s in summaries.items():
        print(f"{scheme:8s} {s['accuracy']:7.2%} {s['mean_error_scope']:8.2f} "
              f"{s['mean_total_tokens']:10.0f} {s['mean_llm_calls']:7.1f} "
              f"{s['mean_batches']:7.1f} "
              f"{s['mean_max_volume']:6.1f} {s['mean_max_latency']:6.1f}")

    print(f"\nPer-instance CSV : {csv_path}")
    print(f"Summary JSON     : {json_path}")


if __name__ == "__main__":
    main()
