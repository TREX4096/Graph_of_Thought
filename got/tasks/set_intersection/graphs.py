"""
Graphs of Operations for set intersection (GoT Section 5.2).

Structure, for two sets A and B with B split into m subsets::

    Input(A, B)
      |
      +-- split B (local) --> B1 .. Bm
            |
            +-- B1 --> Generate(intersect A, k) --> Score --> KeepBest(1) --+
            +-- B2 --> Generate(intersect A, k) --> Score --> KeepBest(1) --+--> Aggregate (union)
            ...                                                                  --> Score --> KeepBest(1)
                                                                                       |
                                                                                 GroundTruth

Same shape as sorting, different semantics at the merge (union rather than
ordered merge) -- which is exactly the point: the framework is task-agnostic
and only the Prompter/Parser/scoring change.
"""

from __future__ import annotations

from typing import List

from ...operations import (
    Generate,
    GroundTruth,
    Improve,
    InputOp,
    KeepBest,
    KeepBestPerGroup,
    Operation,
    PairwiseAggregate,
    Score,
)
from .scoring import intersection_score, is_correct_intersection


def token_budget(n_elements: int) -> int:
    """Generation cap for a list of ``n_elements`` integers.

    Set elements here can be multi-digit (the generator draws from a universe
    four times the set size), so we budget more per element than the sorting
    task does. Still far below a 1024-token default, which is the point.

    As with sorting, bias high: generation stops at the stop string, so an
    unused ceiling costs nothing, whereas truncating the list breaks the
    parse and drops the thought. See ``sorting.graphs.token_budget``.
    """
    return max(96, 5 * n_elements + 64)


STOP = ("\n\n", "\nInput", "Example:")


def got_intersection_goo(
    set_a: List[int],
    set_b: List[int],
    num_chunks: int = 4,
    branching_factor: int = 3,
    aggregation_attempts: int = 5,
    refine_attempts: int = 10,
) -> List[Operation]:
    """Build the GoT Graph of Operations for set intersection.

    Like the sorting graph, sibling work is folded into single operations
    holding many thoughts so that each level costs one batched model call
    rather than one call per branch. See ``got/tasks/sorting/graphs.py`` for
    the full rationale.
    """
    if num_chunks & (num_chunks - 1) != 0:
        raise ValueError(f"num_chunks must be a power of two, got {num_chunks}")

    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )

    split = Generate(prompt_name="split", name="SplitB")
    split.add_predecessor(root)

    # Intersect every subset of B against the whole of A -- one batched call.
    # The result of each is at most |A| elements, but in practice far fewer.
    gen = Generate(
        prompt_name="intersect",
        branching_factor=branching_factor,
        name=f"IntersectChunks(k={branching_factor})",
        max_tokens=token_budget(len(set_a)),
        stop=STOP,
    )
    gen.add_predecessor(split)

    # Local scoring against the *global* truth: a chunk result containing no
    # spurious elements scores well even though it is incomplete, which is
    # the right signal for choosing between candidate partial intersections.
    sc = Score(scoring_fn=intersection_score, name="ScoreChunks")
    sc.add_predecessor(gen)

    level: Operation = KeepBestPerGroup(group_key="chunk_index", n=1,
                                        name="KeepBestPerChunk")
    level.add_predecessor(sc)

    # Binary union tree, one batched operation per level.
    remaining = num_chunks
    depth = 0
    while remaining > 1:
        depth += 1
        agg = PairwiseAggregate(
            prompt_name="aggregate",
            num_merges=aggregation_attempts,
            name=f"Union_L{depth}(k={aggregation_attempts})",
            max_tokens=token_budget(len(set_a)),
            stop=STOP,
        )
        agg.add_predecessor(level)

        sc_u = Score(scoring_fn=intersection_score, name=f"ScoreUnion_L{depth}")
        sc_u.add_predecessor(agg)

        remaining //= 2
        keep: Operation = (
            KeepBestPerGroup(group_key="_group", n=1, name=f"KeepBestUnion_L{depth}")
            if remaining > 1
            else KeepBest(n=1, name=f"KeepBestUnion_L{depth}")
        )
        keep.add_predecessor(sc_u)
        level = keep

    # Final corrective pass, mirroring the sorting graph. The incumbent is
    # scored alongside the candidates so the pass cannot make things worse --
    # see got.tasks.sorting.graphs for the full rationale.
    if refine_attempts > 0:
        imp = Improve(
            prompt_name="improve",
            rounds=1,
            attempts=refine_attempts,
            name=f"Refine(k={refine_attempts})",
            max_tokens=token_budget(len(set_a)),
            stop=STOP,
        )
        imp.add_predecessor(level)

        sc_r = Score(scoring_fn=intersection_score, name="ScoreRefine")
        sc_r.add_predecessor(imp)
        sc_r.add_predecessor(level)

        keep_r = KeepBest(n=1, name="KeepBestRefine")
        keep_r.add_predecessor(sc_r)
        level = keep_r

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(level)
    return [gt]


def io_intersection_goo(set_a: List[int], set_b: List[int]) -> List[Operation]:
    """IO baseline: ask for the whole intersection in one call."""
    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )
    gen = Generate(prompt_name="intersect", branching_factor=1, name="Intersect(IO)",
                   max_tokens=token_budget(len(set_a)), stop=STOP)
    gen.add_predecessor(root)

    sc = Score(scoring_fn=intersection_score, name="Score")
    sc.add_predecessor(gen)

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(sc)
    return [gt]


def cot_intersection_goo(
    set_a: List[int], set_b: List[int], refine_rounds: int = 1
) -> List[Operation]:
    """CoT baseline: one chain, intersect then refine once.

    Mirrors ``sorting.graphs.cot_goo`` exactly -- more than one LLM call, but
    no branching and no aggregation. Keeping the two tasks' baselines
    structurally identical is what lets us claim any difference between tasks
    comes from the task, not from an accidentally stronger baseline.
    """
    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )
    budget = token_budget(len(set_a))

    gen = Generate(prompt_name="intersect", branching_factor=1,
                   name="Intersect(CoT)", max_tokens=budget, stop=STOP)
    gen.add_predecessor(root)

    imp = Improve(prompt_name="improve", rounds=refine_rounds, name="Refine",
                  max_tokens=budget, stop=STOP)
    imp.add_predecessor(gen)

    sc = Score(scoring_fn=intersection_score, name="Score")
    sc.add_predecessor(imp)

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(sc)
    return [gt]


def cot_sc_intersection_goo(
    set_a: List[int], set_b: List[int], k: int = 5
) -> List[Operation]:
    """CoT-SC baseline: k independent attempts, keep the best-scoring one.

    Selection is by score rather than majority vote. For a 32-element set the
    answer space is far too large for two samples to ever coincide, so a modal
    vote would be meaningless -- see explanation.md Sec 3.5.
    """
    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )
    gen = Generate(prompt_name="intersect", branching_factor=k,
                   name=f"Intersect(CoT-SC,k={k})",
                   max_tokens=token_budget(len(set_a)), stop=STOP)
    gen.add_predecessor(root)

    sc = Score(scoring_fn=intersection_score, name="Score")
    sc.add_predecessor(gen)

    keep = KeepBest(n=1, name="KeepBest")
    keep.add_predecessor(sc)

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(keep)
    return [gt]


def tot_intersection_goo(
    set_a: List[int],
    set_b: List[int],
    branching_factor: int = 3,
    depth: int = 3,
    beam_width: int = 1,
) -> List[Operation]:
    """ToT baseline: branch, score, prune, refine -- and never merge.

    Every vertex here has exactly one parent, so a good partial answer on a
    discarded branch is lost permanently. That is the restriction
    ``got_intersection_goo`` lifts with its union tree.
    """
    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )
    budget = token_budget(len(set_a))

    gen = Generate(prompt_name="intersect", branching_factor=branching_factor,
                   name="Intersect(ToT)", max_tokens=budget, stop=STOP)
    gen.add_predecessor(root)

    sc = Score(scoring_fn=intersection_score, name="Score0")
    sc.add_predecessor(gen)

    current: Operation = KeepBest(n=beam_width, name="KeepBest0")
    current.add_predecessor(sc)

    for level in range(1, depth):
        imp = Improve(prompt_name="improve", rounds=1, name=f"Refine{level}",
                      max_tokens=budget, stop=STOP)
        imp.add_predecessor(current)

        sc_l = Score(scoring_fn=intersection_score, name=f"Score{level}")
        sc_l.add_predecessor(imp)
        sc_l.add_predecessor(current)      # incumbent competes; see sorting

        keep_l = KeepBest(n=beam_width, name=f"KeepBest{level}")
        keep_l.add_predecessor(sc_l)

        current = keep_l

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(current)
    return [gt]


SCHEMES = {
    "io": io_intersection_goo,
    "cot": cot_intersection_goo,
    "cot_sc": cot_sc_intersection_goo,
    "tot": tot_intersection_goo,
    "got": got_intersection_goo,
}
