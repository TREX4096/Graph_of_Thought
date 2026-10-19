"""
Step-entropy pruning on GSM8K and MATH-500.
===========================================

Reproduces the static-pruning protocol of Li et al., "Making Slow Thinking
Faster: Compressing LLM Chain-of-Thought via Step Entropy" (ICLR 2026,
arXiv:2508.03346), on our own stack.

Protocol, per problem
---------------------
  1. Generate a full chain of thought, capturing per-token entropy.
  2. Segment on blank lines; compute length-normalised entropy per step.
  3. Select a fraction of steps to remove, by one of three strategies.
  4. Replace them with [SKIP] and ask the model for the final answer only.
  5. Score against ground truth.

Why all three strategies, always
--------------------------------
Running only ``low`` would be a mistake. "Removing 80% of steps barely
hurts accuracy" is not evidence that entropy identified the redundant
steps - it may simply mean the chain was decorative and *any* 80% could
go. The claim only has content if ``high`` and ``random`` degrade. So the
controls are not optional extras; they are what makes the result a result.

This is also our Axis-1 validation: if the criterion works on a chain, the
next question is whether it transfers to the vertices of a reasoning
graph, which is where our contribution lies.

Usage
-----
    # free pipeline check, no GPU
    python scripts/run_entropy_pruning.py --data data/math/smoke/gsm8k_20.csv \\
        --backend mock --limit 5

    # the real run
    python scripts/run_entropy_pruning.py --data data/math/gsm8k_test.csv \\
        --backend vllm --model-id Qwen/Qwen2.5-7B-Instruct \\
        --limit 200 --ratios 0.5 0.8 --out results/entropy_gsm8k
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from got.backends import get_backend
from got.entropy import (
    apply_pruning,
    extract_thinking,
    measure_steps,
    pruning_report,
    select_to_prune,
)
from got.tasks.math_reasoning import (
    ANSWER_ONLY_PROMPT,
    COT_PROMPT,
    answers_match,
    build_compressed_prefill,
    extract_answer,
)

STRATEGIES = ("low", "high", "random")


def build_backend(args):
    common = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.backend == "mock":
        return get_backend("mock", seed=args.seed, **common)
    if not args.model_id:
        sys.exit(f"--model-id is required for --backend {args.backend}")
    return get_backend(
        args.backend, model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.n_ctx,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **common,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step-entropy CoT pruning on GSM8K / MATH-500",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV from download_math_data.py")
    ap.add_argument("--limit", type=int, default=50, help="problems (0 = all)")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.8],
                    help="prune fractions to evaluate")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                    choices=STRATEGIES)
    ap.add_argument("--out", default="results/entropy")
    ap.add_argument("--backend", default="mock", choices=["mock", "vllm"])
    ap.add_argument("--model-id")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--logprobs", type=int, default=20,
                    help="top-k for the entropy estimate; higher is tighter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    # Reasoning models (DeepSeek-R1) emit <think>...</think>; instruct models
    # (Qwen2.5) never do, and wrapping their prefill in tags they have never
    # seen is its own distribution shift. Default on, since the paper's models
    # are reasoning models.
    ap.add_argument("--think-tags", dest="think_tags", action="store_true",
                    default=True, help="wrap the pruned chain in <think> tags")
    ap.add_argument("--no-think-tags", dest="think_tags", action="store_false",
                    help="prefill the chain without think tags (instruct models)")
    args = ap.parse_args()

    with open(args.data, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    tag = os.path.splitext(os.path.basename(args.data))[0]
    os.makedirs(args.out, exist_ok=True)
    print(f"Dataset   : {args.data}  ({len(rows)} problems)")
    print(f"Backend   : {args.backend}  {args.model_id or ''}")
    print(f"Ratios    : {args.ratios}   Strategies: {args.strategies}\n")

    lm = build_backend(args)

    # ---------------- Stage 1: full chains, with entropy ----------------
    print("Stage 1/3  generating full chains with token entropy")
    chains: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in tqdm(range(0, len(rows), args.batch), unit="batch"):
        chunk = rows[i: i + args.batch]
        prompts = [COT_PROMPT.format(question=r["question"]) for r in chunk]
        outs = lm.generate_with_entropy(
            prompts, max_tokens=args.max_tokens, logprobs=args.logprobs)
        for r, o in zip(chunk, outs):
            body = extract_thinking(o["text"])
            steps = measure_steps(body, o["token_entropies"], o.get("token_texts"))
            chains.append({
                "id": r.get("id"), "question": r["question"], "truth": r["answer"],
                "text": o["text"], "steps": steps, "n_tokens": o.get("n_tokens", 0),
                "full_answer": extract_answer(o["text"]),
            })
    gen_s = time.time() - t0

    # Per-instance rows for EVERY stage, not just the pruned ones. The
    # baseline has to be re-gradable too: a scorer fix that lifts the pruned
    # cells but leaves the full chain at its old grading inflates every
    # margin measured against it.
    per_instance: List[Dict[str, Any]] = []
    for c in chains:
        hit = answers_match(c["full_answer"], c["truth"])
        per_instance.append({
            "id": c["id"], "ratio": "-", "strategy": "full",
            "correct": int(hit), "answer": c["full_answer"], "truth": c["truth"],
            "n_steps": len(c["steps"]), "n_pruned": 0,
            "tokens_before": c["n_tokens"], "tokens_after": c["n_tokens"],
            "mean_entropy_kept": round(
                sum(s.entropy for s in c["steps"]) / len(c["steps"]), 4
            ) if c["steps"] else 0.0,
            "mean_entropy_pruned": 0.0,
        })

    base_correct = sum(r["correct"] for r in per_instance if r["strategy"] == "full")
    base_tokens = sum(c["n_tokens"] for c in chains) / max(1, len(chains))
    mean_steps = sum(len(c["steps"]) for c in chains) / max(1, len(chains))
    print(f"\n  full chain: acc {base_correct/len(chains):.1%}  "
          f"mean tokens {base_tokens:.0f}  mean steps {mean_steps:.1f}  "
          f"({gen_s:.0f}s)")

    if mean_steps < 2:
        print("\n  !! WARNING: chains average fewer than 2 steps, so there is")
        print("     nothing to prune. The model is not emitting blank-line")
        print("     separated steps - check the prompt before trusting results.")

    # ---------------- Stage 2: no-reasoning control ----------------
    print("\nStage 2/3  answer-only control (the floor to beat)")
    ctrl_correct = 0
    for i in tqdm(range(0, len(rows), args.batch), unit="batch"):
        chunk = rows[i: i + args.batch]
        outs = lm.generate_with_entropy(
            [ANSWER_ONLY_PROMPT.format(question=r["question"]) for r in chunk],
            max_tokens=96, logprobs=1)
        for r, o in zip(chunk, outs):
            ans = extract_answer(o["text"])
            hit = answers_match(ans, r["answer"])
            ctrl_correct += hit
            per_instance.append({
                "id": r.get("id"), "ratio": "-", "strategy": "none",
                "correct": int(hit), "answer": ans, "truth": r["answer"],
                "n_steps": 0, "n_pruned": 0,
                "tokens_before": 0, "tokens_after": 0,
                "mean_entropy_kept": 0.0, "mean_entropy_pruned": 0.0,
            })
    print(f"\n  answer only: acc {ctrl_correct/len(rows):.1%}")

    # ---------------- Stage 3: prune and re-answer ----------------
    print("\nStage 3/3  pruning and re-answering")
    results: List[Dict[str, Any]] = []
    for ratio in args.ratios:
        for strat in args.strategies:
            correct, tok_before, tok_after = 0, 0, 0
            for i in tqdm(range(0, len(chains), args.batch),
                          desc=f"  {strat:<6} r={ratio}", unit="batch"):
                chunk = chains[i: i + args.batch]
                prompts, prefills, reports = [], [], []
                for c in chunk:
                    idx = select_to_prune(c["steps"], ratio, strat, args.seed)
                    pruned = apply_pruning(c["steps"], idx)
                    reports.append(pruning_report(c["steps"], idx))
                    # The question goes in the user turn; the pruned chain is
                    # pre-filled into the ASSISTANT turn, so the model resumes
                    # its own reasoning rather than reading someone else's.
                    prompts.append(COT_PROMPT.format(question=c["question"]))
                    prefills.append(build_compressed_prefill(
                        pruned, think_tags=args.think_tags))
                outs = lm.generate_with_entropy(
                    prompts, max_tokens=256, logprobs=1, prefills=prefills)
                for c, o, rep in zip(chunk, outs, reports):
                    ans = extract_answer(o["text"])
                    hit = answers_match(ans, c["truth"])
                    correct += hit
                    tok_before += rep["tokens_before"]
                    tok_after += rep["tokens_after"]
                    per_instance.append({
                        "id": c["id"], "ratio": ratio, "strategy": strat,
                        "correct": int(hit), "answer": ans, "truth": c["truth"],
                        "n_steps": rep["n_steps"], "n_pruned": rep["n_pruned"],
                        "tokens_before": rep["tokens_before"],
                        "tokens_after": rep["tokens_after"],
                        "mean_entropy_kept": round(rep["mean_entropy_kept"], 4),
                        "mean_entropy_pruned": round(rep["mean_entropy_pruned"], 4),
                    })
            results.append({
                "ratio": ratio, "strategy": strat,
                "accuracy": correct / len(chains),
                "token_reduction": 1 - tok_after / max(1, tok_before),
            })

    # ---------------- Report ----------------
    print(f"\n{'strategy':<9} {'ratio':>6} {'accuracy':>9} {'vs full':>9} {'tok saved':>10}")
    print("-" * 48)
    base_acc = base_correct / len(chains)
    print(f"{'full':<9} {'-':>6} {base_acc:9.1%} {'-':>9} {'-':>10}")
    print(f"{'none':<9} {'-':>6} {ctrl_correct/len(rows):9.1%} "
          f"{ctrl_correct/len(rows)-base_acc:+9.1%} {'100%':>10}")
    for r in results:
        print(f"{r['strategy']:<9} {r['ratio']:6.2f} {r['accuracy']:9.1%} "
              f"{r['accuracy']-base_acc:+9.1%} {r['token_reduction']:10.1%}")

    summary = {
        "config": vars(args),
        "baseline": {"accuracy": base_acc, "mean_tokens": base_tokens,
                     "mean_steps": mean_steps},
        "answer_only": {"accuracy": ctrl_correct / len(rows)},
        "pruned": results,
    }
    jp = os.path.join(args.out, f"{tag}_summary.json")
    cp = os.path.join(args.out, f"{tag}_instances.csv")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if per_instance:
        with open(cp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_instance[0].keys()))
            w.writeheader(); w.writerows(per_instance)

    # The claim only holds if the controls degrade. Say so explicitly.
    low = [r for r in results if r["strategy"] == "low"]
    others = [r for r in results if r["strategy"] in ("high", "random")]
    if low and others:
        best_low = max(r["accuracy"] for r in low)
        best_other = max(r["accuracy"] for r in others)
        print()
        if best_low > best_other + 0.02:
            print(f"Entropy criterion HOLDS: low-entropy pruning beats the controls "
                  f"by {best_low - best_other:+.1%}.")
        else:
            print(f"Entropy criterion NOT supported here: low-entropy pruning is "
                  f"{best_low - best_other:+.1%} against the controls. Either the "
                  f"chains carry little redundancy, or segmentation is wrong.")

    print(f"\nSummary  : {jp}\nPer-item : {cp}")


if __name__ == "__main__":
    main()
