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
    Aggregate,
    Generate,
    GroundTruth,
    InputOp,
    KeepBest,
    Operation,
    Score,
    Selector,
)
from .scoring import intersection_score, is_correct_intersection


def _pick_chunk(index: int):
    def _sel(thoughts):
        return [t for t in thoughts if t.state.get("chunk_index") == index]
    return _sel


def got_intersection_goo(
    set_a: List[int],
    set_b: List[int],
    num_chunks: int = 4,
    branching_factor: int = 3,
    aggregation_attempts: int = 5,
) -> List[Operation]:
    """Build the GoT Graph of Operations for set intersection."""
    if num_chunks & (num_chunks - 1) != 0:
        raise ValueError(f"num_chunks must be a power of two, got {num_chunks}")

    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )

    split = Generate(prompt_name="split", name="SplitB")
    split.add_predecessor(root)

    leaves: List[Operation] = []
    for i in range(num_chunks):
        pick = Selector(_pick_chunk(i), name=f"PickChunk{i}")
        pick.add_predecessor(split)

        gen = Generate(
            prompt_name="intersect",
            branching_factor=branching_factor,
            name=f"Intersect{i}(k={branching_factor})",
        )
        gen.add_predecessor(pick)

        sc = Score(scoring_fn=None, name=f"ScoreChunk{i}")
        # Local scoring against the partial truth is not meaningful per chunk
        # (a chunk's correct answer is only part of the final set), so we use
        # the global scorer: a chunk result that contains no spurious elements
        # scores well even though it is incomplete.
        sc.scoring_fn = intersection_score
        sc.add_predecessor(gen)

        keep = KeepBest(n=1, name=f"KeepBestChunk{i}")
        keep.add_predecessor(sc)
        leaves.append(keep)

    # Binary union tree.
    level = leaves
    depth = 0
    while len(level) > 1:
        depth += 1
        nxt: List[Operation] = []
        for j in range(0, len(level), 2):
            agg = Aggregate(
                prompt_name="aggregate",
                num_merges=aggregation_attempts,
                name=f"Union_L{depth}_{j // 2}",
            )
            agg.add_predecessor(level[j])
            agg.add_predecessor(level[j + 1])

            sc = Score(scoring_fn=intersection_score, name=f"ScoreUnion_L{depth}_{j // 2}")
            sc.add_predecessor(agg)

            keep = KeepBest(n=1, name=f"KeepBestUnion_L{depth}_{j // 2}")
            keep.add_predecessor(sc)
            nxt.append(keep)
        level = nxt

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(level[0])
    return [gt]


def io_intersection_goo(set_a: List[int], set_b: List[int]) -> List[Operation]:
    """IO baseline: ask for the whole intersection in one call."""
    root = InputOp(
        {"set_a": list(set_a), "set_b": list(set_b), "current": list(set_b)},
        name="Input",
    )
    gen = Generate(prompt_name="intersect", branching_factor=1, name="Intersect(IO)")
    gen.add_predecessor(root)

    sc = Score(scoring_fn=intersection_score, name="Score")
    sc.add_predecessor(gen)

    gt = GroundTruth(check_fn=is_correct_intersection, name="GroundTruth")
    gt.add_predecessor(sc)
    return [gt]


SCHEMES = {
    "io": io_intersection_goo,
    "got": got_intersection_goo,
}
