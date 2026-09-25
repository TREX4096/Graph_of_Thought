"""
Regression tests for silent-failure bugs found on the first real-model run.
===========================================================================

Both bugs here were invisible against the mock backend -- mock output almost
always parses -- and together they turned a 13-minute GPU run into a table of
plausible-looking zeros with no error raised anywhere. That is the worst
failure mode a benchmark can have, so they get dedicated tests.

The chain was:

    max_tokens too tight
      -> answer truncated mid-list, so no closing "]"
      -> extract_list returned None, so the thought was invalid
      -> KeepBest filtered to valid thoughts, found none, returned []
      -> every downstream operation had no inputs and silently did nothing
      -> ToT reported 1.1 LLM calls against a graph specifying 3

Each link is tested below.
"""

from __future__ import annotations

from got.operations import KeepBest, KeepBestPerGroup
from got.prompter import AbstractParser
from got.tasks.set_intersection.graphs import token_budget as si_budget
from got.tasks.sorting.graphs import token_budget as sort_budget
from got.thought import Thought


def _thought(score: float, valid: bool, group: int = 0) -> Thought:
    t = Thought(state={"current": [1, 2], "chunk_index": group}, valid=valid)
    t.score = score
    return t


# ----------------------------------------------------------------------
# Link 1: the parser must salvage a truncated list
# ----------------------------------------------------------------------
def test_extract_list_reads_a_complete_list():
    assert AbstractParser.extract_list("Output: [0, 1, 2, 3]") == [0, 1, 2, 3]


def test_extract_list_takes_the_last_list_not_the_first():
    """Chatty models restate the input before answering."""
    raw = "Input was [9, 8, 7]. Sorted: [7, 8, 9]"
    assert AbstractParser.extract_list(raw) == [7, 8, 9]


def test_extract_list_salvages_a_truncated_list():
    """A list cut off by max_tokens has no ']' -- salvage rather than discard."""
    got = AbstractParser.extract_list("Output: [0, 0, 1, 1, 2, 3, 5, 7, 8")
    # The final number is dropped: it may be a half-emitted digit.
    assert got == [0, 0, 1, 1, 2, 3, 5, 7]


def test_extract_list_still_rejects_genuine_prose():
    """Salvaging must not turn 'no answer' into a fake answer."""
    assert AbstractParser.extract_list("I cannot sort this list.") is None
    assert AbstractParser.extract_list("") is None


# ----------------------------------------------------------------------
# Link 2: ranking must never empty the graph
# ----------------------------------------------------------------------
def test_keepbest_falls_back_when_every_thought_is_invalid():
    """Returning [] here is what silently truncated whole reasoning graphs."""
    op = KeepBest(n=1)
    op.get_input_thoughts = lambda: [
        _thought(1.0, False), _thought(5.0, False), _thought(3.0, False)
    ]
    kept = op._execute(None, None, None)
    assert len(kept) == 1, "graph must keep its shape even on an all-bad batch"
    assert kept[0].score == 5.0, "the best of a bad lot is still the best"


def test_keepbest_prefers_valid_over_higher_scoring_invalid():
    op = KeepBest(n=1)
    op.get_input_thoughts = lambda: [_thought(9.0, False), _thought(2.0, True)]
    assert op._execute(None, None, None)[0].score == 2.0


def test_keepbest_returns_nothing_when_given_nothing():
    """The fallback must not invent thoughts out of an empty input."""
    op = KeepBest(n=1)
    op.get_input_thoughts = lambda: []
    assert op._execute(None, None, None) == []


def test_keepbestpergroup_never_loses_a_chunk():
    """Dropping a group would change the merge tree's shape at the next level."""
    op = KeepBestPerGroup(group_key="chunk_index", n=1)
    op.get_input_thoughts = lambda: [
        _thought(1.0, False, 0), _thought(4.0, False, 0),
        _thought(2.0, False, 1), _thought(7.0, False, 1),
    ]
    kept = op._execute(None, None, None)
    assert len(kept) == 2, "one survivor per chunk, valid or not"
    assert {t.score for t in kept} == {4.0, 7.0}


# ----------------------------------------------------------------------
# Link 3: the budget must fit the answer it is asking for
# ----------------------------------------------------------------------
def test_token_budget_fits_a_full_length_answer():
    """~2 tokens per element is the floor; the cap must clear it with slack.

    The original 2n+32 gave 160 tokens for a 64-element answer needing ~130,
    leaving no room for a preamble -- so any chatty model truncated.
    """
    for n in (16, 32, 64, 128):
        minimum = 2 * n          # digits and separators, ideal encoding
        assert sort_budget(n) > minimum * 1.25, f"sorting budget too tight at n={n}"
        assert si_budget(n) > minimum * 1.25, f"intersection budget too tight at n={n}"


def test_token_budget_grows_with_input():
    assert sort_budget(128) > sort_budget(64) > sort_budget(32)


# ----------------------------------------------------------------------
# The final corrective pass (reference repo: Generate(1,10) after the
# last aggregation). Its value depends entirely on being monotone.
# ----------------------------------------------------------------------
def test_refinement_pass_adds_the_expected_operations():
    from got.tasks.sorting.graphs import got_sorting_goo

    def names(leaves):
        seen, stack = [], list(leaves)
        while stack:
            op = stack.pop()
            if op.name in seen:
                continue
            seen.append(op.name)
            stack.extend(op.predecessors)
        return seen

    with_refine = names(got_sorting_goo(list(range(32)), refine_attempts=10))
    without = names(got_sorting_goo(list(range(32)), refine_attempts=0))

    assert any("Refine" in n for n in with_refine)
    assert not any("Refine" in n for n in without), "0 must disable the pass"
    assert len(with_refine) == len(without) + 3, "Improve + Score + KeepBest"


def test_refinement_scores_the_incumbent_alongside_candidates():
    """Monotonicity depends on this wiring, so assert it directly.

    If ScoreRefine has only the Improve operation as a predecessor, KeepBest
    ranks the candidates alone and a bad refinement round can return an answer
    *worse* than the merged result it started from.
    """
    from got.tasks.sorting.graphs import got_sorting_goo

    leaves = got_sorting_goo(list(range(32)), refine_attempts=10)
    stack, score_refine = list(leaves), None
    while stack:
        op = stack.pop()
        if op.name == "ScoreRefine":
            score_refine = op
            break
        stack.extend(op.predecessors)

    assert score_refine is not None, "ScoreRefine should be in the graph"
    assert len(score_refine.predecessors) == 2, (
        "ScoreRefine needs both the refinement candidates and the incumbent"
    )
    pred_names = {p.name for p in score_refine.predecessors}
    assert any("Refine" in n for n in pred_names)
    assert any("KeepBestMerge" in n for n in pred_names), "incumbent must compete"


def test_tot_refinement_can_reject_a_worse_rewrite():
    """ToT's state evaluator must be able to prune a bad refinement.

    Without the incumbent wired into Score, KeepBest ranks only the rewrite and
    is forced to accept it even when it is worse -- which measured *below the
    IO baseline* on a real model. It also contradicts ToT's definition, whose
    defining component is a state evaluator that prunes.
    """
    from got.tasks.sorting.graphs import tot_goo

    leaves = tot_goo(list(range(32)), branching_factor=3, depth=3)
    stack, seen, checked = list(leaves), set(), 0
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        if op.name.startswith("Score") and op.name != "Score0":
            assert len(op.predecessors) == 2, (
                f"{op.name} must rank the rewrite against the incumbent"
            )
            checked += 1
        stack.extend(op.predecessors)
    assert checked >= 1, "expected at least one refinement level to check"


def test_tot_still_reports_no_aggregation():
    """Ranking two inputs is not aggregating them.

    The fix above gives ToT's Score an in-degree of 2, which must NOT be
    mistaken for aggregation: no thought is ever combined with another, only
    ranked against it. If this ever flips, the GoT-vs-ToT structural claim
    becomes meaningless.
    """
    from got.controller import Controller
    from got.backends import get_backend
    from got.tasks.sorting import SortingParser, SortingPrompter
    from got.tasks.sorting.graphs import tot_goo

    lm = get_backend("mock", seed=0)
    ctrl = Controller(lm, SortingPrompter(), SortingParser(),
                      tot_goo(list(range(32)), branching_factor=3, depth=3))
    ctrl.run(num_chunks=4)
    assert ctrl.graph_summary()["n_aggregations"] == 0, "ToT must stay a tree"
