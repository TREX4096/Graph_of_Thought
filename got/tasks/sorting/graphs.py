"""
Graphs of Operations for sorting -- GoT and all four baselines.
===============================================================

This module is where the paper's central claim becomes testable. We build
*five* execution plans for the same task, differing only in graph structure:

    IO      Input -> one LLM call -> answer.              (no intermediate thoughts)
    CoT     Input -> one call, reasoning in-context.      (a chain)
    CoT-SC  Input -> k independent chains -> pick best.   (k chains, no cross-talk)
    ToT     Input -> split -> branch k, prune, descend.   (a tree)
    GoT     Input -> split -> sort chunks -> AGGREGATE.   (a DAG)

Because the Controller, backend, prompts and scoring are shared, any measured
difference between them is attributable to graph structure alone. That is the
experimental design the replication needs.

The GoT sorting graph (paper Figure 4)
--------------------------------------
Reproducing the figure for 64 numbers with 4 chunks::

    Input [64 numbers]
      |
      +-- split (local, no LLM) --> 4 chunks of 16
            |
            +-- chunk 0 --> Generate(sort, k=3) --> Score --> KeepBest(1) --+
            +-- chunk 1 --> Generate(sort, k=3) --> Score --> KeepBest(1) --+--> Aggregate(k=10)
            |                                                               |     --> Score --> KeepBest(1) --+
            +-- chunk 2 --> Generate(sort, k=3) --> Score --> KeepBest(1) --+                                  |
            +-- chunk 3 --> Generate(sort, k=3) --> Score --> KeepBest(1) --+--> Aggregate(k=10)               |
                                                                                  --> Score --> KeepBest(1) --+
                                                                                                              |
                                                                          Aggregate(k=10) <-------------------+
                                                                                  |
                                                                          Score --> KeepBest(1) --> GroundTruth

Note k=3 for sorting chunks and k=10 for aggregation: these are the paper's
own values from Figure 4 ("k=3 means that, for each 16 element chunk, we
generate three different sortings"; "k=10 means that we try 10 different
aggregations of the two input 16-element subarrays"). Aggregation gets a
larger k because merging is where errors concentrate -- it is the step that
must get the global multiset right.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...operations import (
    Aggregate,
    Generate,
    GroundTruth,
    Improve,
    InputOp,
    KeepBest,
    Operation,
    Score,
    Selector,
)
from .scoring import is_correctly_sorted, sorting_score


def _pick_chunk(index: int):
    """Selector predicate: keep only the thought whose chunk_index matches.

    Needed because ``split`` emits all chunks from a single operation, but
    each chunk must then flow down its own independent sorting branch.
    """
    def _sel(thoughts):
        return [t for t in thoughts if t.state.get("chunk_index") == index]
    return _sel


# ======================================================================
# GoT -- the full graph with aggregation
# ======================================================================
def got_sorting_goo(
    numbers: List[int],
    num_chunks: int = 4,
    branching_factor: int = 3,
    aggregation_attempts: int = 10,
    use_llm_scoring: bool = False,
) -> List[Operation]:
    """Build the GoT Graph of Operations for sorting.

    Parameters
    ----------
    numbers:
        The list to sort.
    num_chunks:
        How many pieces to decompose into. Must be a power of two for the
        binary merge tree below. The paper uses 4 for 64 elements.
    branching_factor:
        ``k`` for sorting each chunk (paper: 3).
    aggregation_attempts:
        ``k`` for each merge (paper: 10).
    use_llm_scoring:
        If False (default, and what the paper does for sorting) use the exact
        local scorer. If True, query the LLM for scores instead -- useful for
        an ablation showing how much the exact scorer contributes.

    Returns
    -------
    A single-element list holding the leaf operation, ready for the Controller.
    """
    if num_chunks & (num_chunks - 1) != 0:
        raise ValueError(
            f"num_chunks must be a power of two for the merge tree, got {num_chunks}"
        )

    scorer = None if use_llm_scoring else sorting_score

    # --- Input and structural decomposition --------------------------
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    split = Generate(prompt_name="split", branching_factor=1, name="Split")
    split.add_predecessor(root)

    # --- One independent sorting branch per chunk --------------------
    # This is the "explore many partial solutions in parallel" part that ToT
    # also has. The difference from ToT comes later, at the merge.
    branch_leaves: List[Operation] = []
    for i in range(num_chunks):
        pick = Selector(_pick_chunk(i), name=f"PickChunk{i}")
        pick.add_predecessor(split)

        gen = Generate(
            prompt_name="sort",
            branching_factor=branching_factor,
            name=f"SortChunk{i}(k={branching_factor})",
        )
        gen.add_predecessor(pick)

        sc = Score(scoring_fn=scorer, name=f"ScoreChunk{i}")
        sc.add_predecessor(gen)

        keep = KeepBest(n=1, name=f"KeepBestChunk{i}")
        keep.add_predecessor(sc)

        branch_leaves.append(keep)

    # --- Binary merge tree of Aggregations ---------------------------
    # Each level halves the number of partial solutions. This is what gives
    # latency log_k N while every leaf still reaches the root -- volume N.
    level = branch_leaves
    depth = 0
    while len(level) > 1:
        depth += 1
        next_level: List[Operation] = []
        for j in range(0, len(level), 2):
            left, right = level[j], level[j + 1]

            agg = Aggregate(
                prompt_name="aggregate",
                num_merges=aggregation_attempts,
                name=f"Merge_L{depth}_{j // 2}(k={aggregation_attempts})",
            )
            # THE defining GoT edge: two distinct parents into one vertex.
            agg.add_predecessor(left)
            agg.add_predecessor(right)

            sc = Score(scoring_fn=scorer, name=f"ScoreMerge_L{depth}_{j // 2}")
            sc.add_predecessor(agg)

            keep = KeepBest(n=1, name=f"KeepBestMerge_L{depth}_{j // 2}")
            keep.add_predecessor(sc)

            next_level.append(keep)
        level = next_level

    final = level[0]

    # --- Terminal evaluation -----------------------------------------
    # Compare against the true global input, not the local merge input.
    gt = GroundTruth(
        check_fn=lambda st: list(st.get("current", [])) == sorted(numbers),
        name="GroundTruth",
    )
    gt.add_predecessor(final)

    return [gt]


# ======================================================================
# Baselines
# ======================================================================
def io_goo(numbers: List[int]) -> List[Operation]:
    """IO baseline: a single LLM call, no intermediate thoughts.

    Figure 1(a) of the paper. This is the floor everything else must beat.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    gen = Generate(prompt_name="sort", branching_factor=1, name="Sort(IO)")
    gen.add_predecessor(root)

    sc = Score(scoring_fn=sorting_score, name="Score")
    sc.add_predecessor(gen)

    gt = GroundTruth(check_fn=is_correctly_sorted, name="GroundTruth")
    gt.add_predecessor(sc)
    return [gt]


def cot_goo(numbers: List[int], refine_rounds: int = 1) -> List[Operation]:
    """CoT baseline: one chain with an intermediate refinement step.

    A chain of thoughts, i.e. Figure 1(b). We model the "intermediate
    reasoning step" as a self-refinement pass, which keeps the comparison
    fair: CoT gets more than one LLM call, just no branching.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    gen = Generate(prompt_name="sort", branching_factor=1, name="Sort(CoT)")
    gen.add_predecessor(root)

    imp = Improve(prompt_name="improve", rounds=refine_rounds, name="Refine")
    imp.add_predecessor(gen)

    sc = Score(scoring_fn=sorting_score, name="Score")
    sc.add_predecessor(imp)

    gt = GroundTruth(check_fn=is_correctly_sorted, name="GroundTruth")
    gt.add_predecessor(sc)
    return [gt]


def cot_sc_goo(numbers: List[int], k: int = 5) -> List[Operation]:
    """CoT-SC baseline: k independent chains, keep the best.

    Figure 1(c). Crucially the k chains never exchange information -- there
    is no aggregation. This is the direct structural contrast with GoT, and
    it is why CoT-SC's volume is only N/k in Table 2.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    gen = Generate(prompt_name="sort", branching_factor=k, name=f"Sort(CoT-SC,k={k})")
    gen.add_predecessor(root)

    sc = Score(scoring_fn=sorting_score, name="Score")
    sc.add_predecessor(gen)

    keep = KeepBest(n=1, name="KeepBest")
    keep.add_predecessor(sc)

    gt = GroundTruth(check_fn=is_correctly_sorted, name="GroundTruth")
    gt.add_predecessor(keep)
    return [gt]


def tot_goo(
    numbers: List[int],
    branching_factor: int = 3,
    depth: int = 3,
    beam_width: int = 1,
) -> List[Operation]:
    """ToT baseline: iterative branch-score-prune, no aggregation.

    Figure 1(d). At each level we generate ``k`` refinements of the surviving
    thought(s), score them, and keep the best ``beam_width`` -- i.e. BFS with
    a beam, which is Algorithm 1 of the ToT paper.

    The structural point: every vertex has exactly one parent. Information
    from a discarded branch is lost forever. GoT's Aggregate is precisely
    what lifts that restriction.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    gen = Generate(prompt_name="sort", branching_factor=branching_factor, name="Sort(ToT)")
    gen.add_predecessor(root)

    sc = Score(scoring_fn=sorting_score, name="Score0")
    sc.add_predecessor(gen)

    current: Operation = KeepBest(n=beam_width, name="KeepBest0")
    current.add_predecessor(sc)

    # Each additional level is a refine-and-prune round.
    for level in range(1, depth):
        imp = Improve(prompt_name="improve", rounds=1, name=f"Refine{level}")
        imp.add_predecessor(current)

        # Refinement produces one thought per input; to branch we score and
        # prune the accumulated candidates.
        sc_l = Score(scoring_fn=sorting_score, name=f"Score{level}")
        sc_l.add_predecessor(imp)

        keep_l = KeepBest(n=beam_width, name=f"KeepBest{level}")
        keep_l.add_predecessor(sc_l)

        current = keep_l

    gt = GroundTruth(check_fn=is_correctly_sorted, name="GroundTruth")
    gt.add_predecessor(current)
    return [gt]


# Convenient registry so experiment scripts can select a scheme by name.
SCHEMES = {
    "io": io_goo,
    "cot": cot_goo,
    "cot_sc": cot_sc_goo,
    "tot": tot_goo,
    "got": got_sorting_goo,
}
