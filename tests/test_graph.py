"""
Tests for the graph machinery: thoughts, operations, controller, metrics.

The point of these tests is to verify the *structural* claims of the GoT
paper on graphs we actually build -- especially that aggregation produces
vertices with in-degree > 1 and that this raises volume without raising
latency, which is Table 2's whole argument.
"""

import pytest

from got import (
    Aggregate,
    Controller,
    Generate,
    InputOp,
    KeepBest,
    Score,
    Thought,
    graph_metrics,
    latency,
    theoretical_bounds,
    volume,
)
from got.backends import MockLM
from got.tasks.sorting import SortingParser, SortingPrompter
from got.tasks.sorting.graphs import SCHEMES, got_sorting_goo, tot_goo


# ----------------------------------------------------------------------
# Thought
# ----------------------------------------------------------------------
class TestThought:
    def test_ids_are_unique(self):
        a, b = Thought(), Thought()
        assert a.id != b.id

    def test_edges_are_bidirectional(self):
        parent, child = Thought(), Thought()
        child.add_predecessor(parent)
        assert parent in child.predecessors
        assert child in parent.successors

    def test_no_duplicate_edges(self):
        parent, child = Thought(), Thought()
        child.add_predecessor(parent)
        child.add_predecessor(parent)
        assert len(child.predecessors) == 1

    def test_is_aggregate_requires_two_parents(self):
        p1, p2, child = Thought(), Thought(), Thought()
        child.add_predecessor(p1)
        assert not child.is_aggregate      # a chain link, not an aggregation
        child.add_predecessor(p2)
        assert child.is_aggregate          # in-degree 2 -> genuinely a graph


# ----------------------------------------------------------------------
# Metrics (paper Section 6)
# ----------------------------------------------------------------------
class TestMetrics:
    def test_volume_of_source_is_zero(self):
        assert volume(Thought()) == 0

    def test_volume_counts_all_ancestors(self):
        # a -> b -> c  : c can be influenced by both a and b.
        a, b, c = Thought(), Thought(), Thought()
        b.add_predecessor(a)
        c.add_predecessor(b)
        assert volume(c) == 2

    def test_volume_counts_each_ancestor_once(self):
        # Diamond: a -> b, a -> c, (b, c) -> d.
        # d's volume is 3 (a, b, c) -- a must not be double counted.
        a, b, c, d = Thought(), Thought(), Thought(), Thought()
        b.add_predecessor(a)
        c.add_predecessor(a)
        d.add_predecessor(b)
        d.add_predecessor(c)
        assert volume(d) == 3
        assert d.is_aggregate

    def test_latency_is_shortest_path(self):
        # A diamond has depth 2 by either route.
        a, b, c, d = Thought(), Thought(), Thought(), Thought()
        b.add_predecessor(a)
        c.add_predecessor(a)
        d.add_predecessor(b)
        d.add_predecessor(c)
        assert latency(d) == 2

    def test_aggregation_buys_volume_without_latency(self):
        """The core Table 2 claim, on a minimal example.

        Two parallel chains of length 2 that are aggregated give volume 4 at
        latency 3. Without the aggregation, the tip of one chain sees volume
        2 at latency 2. Aggregation therefore doubles volume for one extra
        hop -- exactly the tradeoff the paper exploits.
        """
        root = Thought()
        chain_tips = []
        for _ in range(2):
            mid = Thought(); mid.add_predecessor(root)
            tip = Thought(); tip.add_predecessor(mid)
            chain_tips.append(tip)

        assert volume(chain_tips[0]) == 2
        assert latency(chain_tips[0]) == 2

        merged = Thought()
        for tip in chain_tips:
            merged.add_predecessor(tip)

        assert volume(merged) == 5      # root + 2 mids + 2 tips
        assert latency(merged) == 3

    def test_theoretical_bounds_match_table_2(self):
        b = theoretical_bounds(n=64, k=4)
        # log_4(64) = 3
        assert b["ToT"]["latency"] == pytest.approx(3.0)
        assert b["GoT"]["latency"] == pytest.approx(3.0)
        # GoT keeps full volume N where ToT collapses to log_k N.
        assert b["GoT"]["volume"] == 64
        assert b["ToT"]["volume"] == pytest.approx(3.0)
        assert b["CoT"]["latency"] == 64


# ----------------------------------------------------------------------
# Controller / GoO execution
# ----------------------------------------------------------------------
class TestController:
    def test_detects_cyclic_goo(self):
        """A cyclic plan must fail loudly, not hang."""
        a = Generate(name="A")
        b = Generate(name="B")
        a.add_predecessor(b)
        b.add_predecessor(a)

        ctrl = Controller(MockLM(), SortingPrompter(), SortingParser(), [b])
        with pytest.raises(ValueError, match="cycle"):
            ctrl.run()

    def test_topological_order_respects_dependencies(self):
        root = InputOp({"current": [3, 1, 2], "original": [3, 1, 2]})
        gen = Generate(prompt_name="sort", branching_factor=1)
        gen.add_predecessor(root)
        sc = Score(scoring_fn=lambda s: 1.0)
        sc.add_predecessor(gen)

        ctrl = Controller(MockLM(seed=1), SortingPrompter(), SortingParser(), [sc])
        ctrl.run()

        names = [op.name for op in ctrl.execution_order]
        assert names.index(root.name) < names.index(gen.name) < names.index(sc.name)

    def test_input_op_needs_no_llm_call(self):
        lm = MockLM()
        root = InputOp({"current": [1], "original": [1]})
        ctrl = Controller(lm, SortingPrompter(), SortingParser(), [root])
        ctrl.run()
        assert lm.usage.n_calls == 0

    def test_split_is_local_and_free(self):
        """Chunking must not spend an LLM call (Prompter returns None)."""
        lm = MockLM()
        root = InputOp({"current": list(range(8)), "original": list(range(8))})
        split = Generate(prompt_name="split", name="Split")
        split.add_predecessor(root)

        ctrl = Controller(lm, SortingPrompter(), SortingParser(), [split])
        ctrl.run(num_chunks=4)

        assert lm.usage.n_calls == 0
        assert len(split.thoughts) == 4
        # Chunks must partition the input exactly -- term Y depends on it.
        recombined = [v for t in split.thoughts for v in t.state["current"]]
        assert sorted(recombined) == sorted(range(8))


# ----------------------------------------------------------------------
# End-to-end structural properties of each scheme
# ----------------------------------------------------------------------
class TestSchemeStructure:
    NUMBERS = [5, 3, 8, 1, 9, 2, 7, 0, 4, 6, 1, 3, 8, 2, 5, 7]

    def _run(self, scheme):
        lm = MockLM(error_rate=0.1, seed=3)
        builder = SCHEMES[scheme]
        leaves = (
            builder(self.NUMBERS, num_chunks=4, branching_factor=2,
                    aggregation_attempts=3)
            if scheme == "got" else builder(self.NUMBERS)
        )
        ctrl = Controller(lm, SortingPrompter(), SortingParser(), leaves)
        ctrl.run(num_chunks=4)
        return ctrl, lm

    @pytest.mark.parametrize("scheme", ["io", "cot", "cot_sc", "tot", "got"])
    def test_every_scheme_runs_and_produces_a_list(self, scheme):
        ctrl, _ = self._run(scheme)
        best = ctrl.best_thought()
        assert best is not None
        assert isinstance(best.state["current"], list)
        assert ctrl.is_correct() in (True, False)

    @pytest.mark.parametrize("scheme", ["io", "cot", "cot_sc", "tot"])
    def test_baselines_contain_no_aggregation(self, scheme):
        """IO/CoT/CoT-SC/ToT must be trees -- no vertex with in-degree > 1."""
        ctrl, _ = self._run(scheme)
        assert ctrl.graph_summary()["n_aggregations"] == 0

    def test_got_contains_aggregations(self):
        """GoT must actually build a DAG, else it is just ToT."""
        ctrl, _ = self._run("got")
        assert ctrl.graph_summary()["n_aggregations"] > 0

    def test_got_achieves_higher_volume_than_tot(self):
        """The empirical counterpart of Table 2.

        GoT should see strictly more preceding thoughts at its output than
        ToT does, because aggregation connects every branch to the result.
        """
        got_ctrl, _ = self._run("got")
        tot_ctrl, _ = self._run("tot")

        got_m = graph_metrics(got_ctrl.all_thoughts())
        tot_m = graph_metrics(tot_ctrl.all_thoughts())

        assert got_m["max_volume"] > tot_m["max_volume"]

    def test_usage_is_tracked(self):
        _, lm = self._run("got")
        assert lm.usage.n_calls > 0
        assert lm.usage.total_tokens > 0
        assert lm.usage.cost == 0.0     # open-source models are free


class TestMockDeterminism:
    def test_same_seed_same_output(self):
        """Reproducibility -- the property a real LLM cannot give us."""
        outs = []
        for _ in range(2):
            lm = MockLM(seed=99, error_rate=0.3)
            leaves = SCHEMES["got"]([4, 2, 9, 1] * 4, num_chunks=2,
                                    branching_factor=2, aggregation_attempts=2)
            ctrl = Controller(lm, SortingPrompter(), SortingParser(), leaves)
            ctrl.run(num_chunks=2)
            outs.append(ctrl.best_thought().state["current"])
        assert outs[0] == outs[1]

    def test_zero_error_rate_is_a_perfect_oracle(self):
        """With no corruption the pipeline must reach a perfect answer.

        This isolates framework bugs from model errors: if this fails, the
        graph plumbing is broken, not the model.
        """
        numbers = [7, 3, 9, 1, 5, 5, 2, 8]
        lm = MockLM(seed=0, error_rate=0.0, length_sensitivity=0.0)
        leaves = got_sorting_goo(numbers, num_chunks=2, branching_factor=2,
                                 aggregation_attempts=2)
        ctrl = Controller(lm, SortingPrompter(), SortingParser(), leaves)
        ctrl.run(num_chunks=2)

        assert ctrl.best_thought().state["current"] == sorted(numbers)
        assert ctrl.is_correct() is True
