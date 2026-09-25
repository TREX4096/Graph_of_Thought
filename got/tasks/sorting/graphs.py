"""
Graphs of Operations for sorting -- GoT and all four baselines.
===============================================================

This module is where the paper's central claim becomes testable. We build
*five* execution plans for the same task, differing only in graph structure:

    IO      Input -> one LLM call -> answer.              (no intermediate thoughts)
    CoT     Input -> one call, reasoning in-context.      (a chain)
    CoT-SC  Input -> k independent chains -> pick best.   (k chains, no cross-talk)
    ToT     Input -> branch, score, prune, refine.        (a tree)
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
            +-- Generate(sort, k=3)   <- ALL FOUR CHUNKS IN ONE BATCHED CALL
            +-- Score (local, exact, free)
            +-- KeepBestPerGroup(chunk_index, n=1)   -> 4 survivors
            |
            +-- PairwiseAggregate(k=10)  <- both merges in one batched call
            +-- Score --> KeepBestPerGroup(_group)   -> 2 survivors
            |
            +-- PairwiseAggregate(k=10)
            +-- Score --> KeepBest(1)                -> 1 answer
                                |
                          GroundTruth

Note k=3 for sorting chunks and k=10 for aggregation: these are the paper's
own values from Figure 4 ("k=3 means that, for each 16 element chunk, we
generate three different sortings"; "k=10 means that we try 10 different
aggregations of the two input 16-element subarrays"). Aggregation gets a
larger k because merging is where errors concentrate -- it is the step that
must get the global multiset right.

Why the chunk branches are one operation, not four
---------------------------------------------------
An earlier version of this file gave each chunk its own
Selector -> Generate -> Score -> KeepBest chain. The reasoning *graph* was
identical, but the cost was not: the Controller executes operations one at a
time, so four single-input Generate operations meant four serial model calls,
each submitting a batch of one. On a GPU that leaves the device mostly idle.

Folding them into a single ``Generate`` over all four chunk thoughts, followed
by ``KeepBestPerGroup``, produces exactly the same thoughts and the same edges
while collapsing 4 serial calls into 1 batched call of 4 prompts (12 sequences
at k=3). The same argument applies to ``PairwiseAggregate`` at each merge
level. This matters a great deal on a cluster where GPU time is the budget.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

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
from .scoring import is_correctly_sorted, sorting_score


# ----------------------------------------------------------------------
# Token budgeting -- a direct multiplier on GPU cost
# ----------------------------------------------------------------------
# Decode time is roughly linear in tokens generated. The answers here are
# short, fixed-shape lists of single digits, so the required length is
# predictable and we should say so rather than letting a 1024-token default
# stand. A list of n digits renders as "[d, d, ..., d]" -- about 3 characters
# per element, and tokenizers typically split "1," / " 2" into one token each.
# We budget ~2 tokens per element plus generous slack for the brackets and any
# short preamble a chatty model might emit.
def token_budget(n_elements: int) -> int:
    """Generation cap for an answer listing ``n_elements`` digits.

    Budget ~3 tokens per element plus 64 of slack. The earlier ``2n + 32``
    was measured against the ideal encoding (" 5" + "," = 2 tokens) and left
    no room at all: a 64-element answer needs ~130 tokens of digits, against
    a 160-token cap, so any preamble at all ("Here is the sorted list:")
    truncated the answer mid-list. A truncated list has no closing bracket,
    which made the parse fail, which made the thought invalid, which made
    KeepBest discard it -- collapsing the graph.

    Over-budgeting is close to free: generation stops at the stop string or
    EOS, so the cap is a ceiling and not a target. Under-budgeting is
    catastrophic. Bias high.
    """
    return max(96, 3 * n_elements + 64)


# Stop strings. The answer is a single line, so anything that starts a new
# block means the model has finished and moved on to padding or re-explaining.
# We deliberately do NOT stop on "]" -- the parser needs the closing bracket,
# and vLLM excludes the stop string from the returned text.
SORT_STOP: Sequence[str] = ("\n\n", "\nInput", "Example:")


# ======================================================================
# GoT -- the full graph with aggregation
# ======================================================================
def got_sorting_goo(
    numbers: List[int],
    num_chunks: int = 4,
    branching_factor: int = 3,
    aggregation_attempts: int = 10,
    refine_attempts: int = 10,
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
        ``k`` for sorting each chunk (paper: 3). Cost scales linearly in k.
    aggregation_attempts:
        ``k`` for each merge (paper: 10). This is the **single largest cost
        knob** in the graph -- every merge level generates this many full-length
        candidate lists. Halving it roughly halves total decode tokens.
    refine_attempts:
        Candidate refinements drawn in the final corrective pass, matching the
        reference implementation's ``Generate(1, 10)`` after its last
        aggregation. Pass 0 to disable the pass entirely (the graph shape we
        had before this was added, useful as an ablation).

        This is where a corrective pass pays for itself: the final merge is
        the only step that must get the *global* multiset right, so it is
        where errors concentrate. Drawing ``refine_attempts`` fixes and keeping
        the best-scoring one cannot make the answer worse, because the scorer
        is exact and the incumbent is one of the things being ranked.
    use_llm_scoring:
        If False (default, and what the paper does for sorting) use the exact
        local scorer -- free, exact, and far cheaper than querying the model.
        If True, query the LLM for scores instead, which adds one generation
        per thought; useful only as an ablation.

    Returns
    -------
    A single-element list holding the leaf operation, ready for the Controller.
    """
    if num_chunks & (num_chunks - 1) != 0:
        raise ValueError(
            f"num_chunks must be a power of two for the merge tree, got {num_chunks}"
        )

    scorer = None if use_llm_scoring else sorting_score
    n = len(numbers)
    chunk_size = max(1, -(-n // num_chunks))

    # --- Input and structural decomposition --------------------------
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    split = Generate(prompt_name="split", branching_factor=1, name="Split")
    split.add_predecessor(root)

    # --- Sort every chunk in ONE batched operation -------------------
    gen = Generate(
        prompt_name="sort",
        branching_factor=branching_factor,
        name=f"SortChunks(k={branching_factor})",
        max_tokens=token_budget(chunk_size),
        stop=SORT_STOP,
    )
    gen.add_predecessor(split)

    sc = Score(scoring_fn=scorer, name="ScoreChunks")
    sc.add_predecessor(gen)

    # Rank within each chunk so every chunk keeps its own best candidate.
    level: Operation = KeepBestPerGroup(group_key="chunk_index", n=1,
                                        name="KeepBestPerChunk")
    level.add_predecessor(sc)

    # --- Binary merge tree of Aggregations ---------------------------
    # Each level halves the number of partial solutions and doubles the
    # length of each. This is what gives latency log_k N while every leaf
    # still reaches the root -- volume N.
    remaining = num_chunks
    depth = 0
    merged_size = chunk_size
    while remaining > 1:
        depth += 1
        merged_size = min(n, merged_size * 2)

        agg = PairwiseAggregate(
            prompt_name="aggregate",
            num_merges=aggregation_attempts,
            name=f"Merge_L{depth}(k={aggregation_attempts})",
            max_tokens=token_budget(merged_size),
            stop=SORT_STOP,
        )
        agg.add_predecessor(level)

        sc_m = Score(scoring_fn=scorer, name=f"ScoreMerge_L{depth}")
        sc_m.add_predecessor(agg)

        remaining //= 2
        if remaining > 1:
            keep: Operation = KeepBestPerGroup(
                group_key="_group", n=1, name=f"KeepBestMerge_L{depth}"
            )
        else:
            # Final level: one global winner.
            keep = KeepBest(n=1, name=f"KeepBestMerge_L{depth}")
        keep.add_predecessor(sc_m)
        level = keep

    # --- Final corrective pass ---------------------------------------
    # Reference implementation, examples/sorting/sorting_032.py:
    #
    #     operations_graph.append_operation(operations.Generate(1, 10))
    #     operations_graph.append_operation(operations.Score(...))
    #     operations_graph.append_operation(operations.KeepBestN(1, False))
    #
    # i.e. after the last aggregation, draw 10 repair attempts on the merged
    # result, score them, keep the best.
    #
    # One deliberate difference from the reference: we feed the *incumbent*
    # into the same Score as the candidates, so KeepBest ranks all
    # refine_attempts + 1 together. The reference ranks only the candidates,
    # which means a bad refinement round can return an answer worse than the
    # one it started from -- precisely the failure visible in our CoT and ToT
    # baselines, where a single blind rewrite roughly doubled the error
    # scope. Including the incumbent makes the pass monotone under an exact
    # scorer: the output is never worse than the input.
    #
    # Note this is itself a fan-in (Score has two predecessors), which is
    # only expressible because the GoO is a graph -- the same structural
    # freedom the paper is about.
    if refine_attempts > 0:
        imp = Improve(
            prompt_name="improve",
            rounds=1,
            attempts=refine_attempts,
            name=f"Refine(k={refine_attempts})",
            max_tokens=token_budget(n),
            stop=SORT_STOP,
        )
        imp.add_predecessor(level)

        sc_r = Score(scoring_fn=scorer, name="ScoreRefine")
        sc_r.add_predecessor(imp)
        sc_r.add_predecessor(level)      # <-- incumbent competes too

        keep_r = KeepBest(n=1, name="KeepBestRefine")
        keep_r.add_predecessor(sc_r)
        level = keep_r

    # --- Terminal evaluation -----------------------------------------
    # Compare against the true global input, not the local merge input.
    target = sorted(numbers)
    gt = GroundTruth(
        check_fn=lambda st: list(st.get("current", [])) == target,
        name="GroundTruth",
    )
    gt.add_predecessor(level)

    return [gt]


# ======================================================================
# Baselines
# ======================================================================
def io_goo(numbers: List[int]) -> List[Operation]:
    """IO baseline: a single LLM call, no intermediate thoughts.

    Figure 1(a) of the paper. This is the floor everything else must beat.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")

    gen = Generate(
        prompt_name="sort", branching_factor=1, name="Sort(IO)",
        max_tokens=token_budget(len(numbers)), stop=SORT_STOP,
    )
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

    Deliberately NOT given a KeepBest
    ---------------------------------
    ToT and GoT score the refinement against the thought it came from and keep
    the better one. CoT does not, and must not: a chain has no selection
    mechanism by definition -- that is exactly what CoT-SC was invented to add.
    Giving this baseline a ranking step would make it something other than CoT.

    The consequence is visible in the results and is worth reporting rather
    than engineering away: on a 7B model this scheme scores *worse than plain
    IO*, because a blind rewrite degrades the answer more often than it
    improves it. That is the cleanest demonstration in the whole benchmark of
    why scoring and selection -- not extra calls -- are what make the later
    schemes work. Extra computation without a way to reject bad results is not
    merely wasted, it is actively harmful.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")
    budget = token_budget(len(numbers))

    gen = Generate(prompt_name="sort", branching_factor=1, name="Sort(CoT)",
                   max_tokens=budget, stop=SORT_STOP)
    gen.add_predecessor(root)

    imp = Improve(prompt_name="improve", rounds=refine_rounds, name="Refine",
                  max_tokens=budget, stop=SORT_STOP)
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

    gen = Generate(prompt_name="sort", branching_factor=k, name=f"Sort(CoT-SC,k={k})",
                   max_tokens=token_budget(len(numbers)), stop=SORT_STOP)
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

    Figure 1(d). At each level we generate refinements of the surviving
    thought(s), score them, and keep the best ``beam_width`` -- i.e. BFS with
    a beam, which is Algorithm 1 of the ToT paper.

    The structural point: every vertex has exactly one parent. Information
    from a discarded branch is lost forever. GoT's Aggregate is precisely
    what lifts that restriction.
    """
    root = InputOp({"current": list(numbers), "original": list(numbers)}, name="Input")
    budget = token_budget(len(numbers))

    gen = Generate(prompt_name="sort", branching_factor=branching_factor,
                   name="Sort(ToT)", max_tokens=budget, stop=SORT_STOP)
    gen.add_predecessor(root)

    sc = Score(scoring_fn=sorting_score, name="Score0")
    sc.add_predecessor(gen)

    current: Operation = KeepBest(n=beam_width, name="KeepBest0")
    current.add_predecessor(sc)

    # Each additional level is a refine-and-prune round.
    #
    # The incumbent is scored alongside the refinement, so KeepBest can reject
    # a refinement that made things worse. Omitting it (an earlier version did)
    # is wrong twice over: it handicaps the baseline in precisely the
    # comparison GoT's headline claim rests on, and it is unfaithful to ToT,
    # whose defining component is a state evaluator that prunes bad states.
    # Measured effect: without this, ToT scored *worse than plain IO* --
    # a blind rewrite degrades more often than it improves.
    #
    # Note this makes the ToT graph's in-degree > 1 at Score, which looks like
    # aggregation but is not: no thought is ever *combined* with another, only
    # ranked against it. Aggregation merges content; this merely selects. The
    # reported n_aggregations stays 0, which is the honest reading.
    for level in range(1, depth):
        imp = Improve(prompt_name="improve", rounds=1, name=f"Refine{level}",
                      max_tokens=budget, stop=SORT_STOP)
        imp.add_predecessor(current)

        sc_l = Score(scoring_fn=sorting_score, name=f"Score{level}")
        sc_l.add_predecessor(imp)
        sc_l.add_predecessor(current)      # incumbent competes

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
