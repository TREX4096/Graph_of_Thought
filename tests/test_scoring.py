"""
Tests for the sorting score -- the exact formula from GoT Section 5.1.

These matter more than they look. The error-scope formula is the single
number every result in the replication depends on; if X or Y is wrong, every
comparison between schemes is wrong in the same direction and the bug is
invisible in aggregate plots.
"""

import pytest

from got.tasks.sorting.scoring import (
    is_correctly_sorted,
    sorting_error_scope,
    sorting_score,
)


class TestErrorScope:
    """error-scope = X + Y, where X counts descents and Y counts frequency drift."""

    def test_perfect_sort_scores_zero(self):
        original = [3, 1, 2, 1]
        assert sorting_error_scope(original, [1, 1, 2, 3]) == 0

    def test_empty_input(self):
        assert sorting_error_scope([], []) == 0

    def test_single_adjacent_descent_costs_one(self):
        # X = 1 (one descent: 3 > 2), Y = 0 (same multiset).
        original = [1, 2, 3]
        assert sorting_error_scope(original, [1, 3, 2]) == 1

    def test_fully_reversed(self):
        # Every adjacent pair is a descent -> X = n - 1, Y = 0.
        original = [1, 2, 3, 4]
        assert sorting_error_scope(original, [4, 3, 2, 1]) == 3

    def test_dropped_element_costs_one(self):
        # X = 0 (still ascending), Y = 1 (the 3 is missing).
        original = [1, 2, 3]
        assert sorting_error_scope(original, [1, 2]) == 1

    def test_duplicated_element_costs_one(self):
        # X = 0, Y = 1 (one extra 2).
        original = [1, 2, 3]
        assert sorting_error_scope(original, [1, 2, 2, 3]) == 1

    def test_both_terms_combine(self):
        # produced = [1, 3, 2] against original [1, 2, 3, 4]:
        #   X = 1  (one descent: 3 > 2)
        #   Y = 1  (the element 4 is missing; 1/2/3 all appear once each)
        original = [1, 2, 3, 4]
        assert sorting_error_scope(original, [1, 3, 2]) == 2

    def test_preserves_duplicates(self):
        # The paper's motivating failure: correct order, wrong duplicate count.
        original = [2, 2, 2, 1]
        produced = [1, 2, 2]          # one 2 lost
        assert sorting_error_scope(original, produced) == 1

    def test_hallucinated_out_of_range_value(self):
        # A value not in the input at all contributes to Y.
        original = [1, 2]
        assert sorting_error_scope(original, [1, 2, 7]) == 1

    def test_clipping(self):
        original = [1, 2, 3]
        produced = list(range(50, 0, -1))   # wildly wrong
        raw = sorting_error_scope(original, produced, clip=False)
        clipped = sorting_error_scope(original, produced, clip=True)
        assert clipped == min(raw, len(original))
        assert clipped <= 3


class TestSortingScore:
    """The higher-is-better form: max(n - error_scope, 0)."""

    def test_perfect_gets_full_marks(self):
        state = {"current": [1, 2, 3], "original": [3, 1, 2]}
        assert sorting_score(state) == 3.0

    def test_never_negative(self):
        # A catastrophically wrong answer floors at 0 rather than going negative.
        state = {"current": list(range(99, 0, -1)), "original": [1, 2]}
        assert sorting_score(state) == 0.0

    def test_missing_keys_score_zero(self):
        # An unparseable thought must never outrank a real one.
        assert sorting_score({}) == 0.0
        assert sorting_score({"current": [1]}) == 0.0

    def test_better_answer_scores_higher(self):
        original = [3, 1, 2, 5, 4]
        good = {"current": [1, 2, 3, 4, 5], "original": original}
        bad = {"current": [5, 4, 3, 2, 1], "original": original}
        assert sorting_score(good) > sorting_score(bad)


class TestGroundTruth:
    def test_exact_match(self):
        assert is_correctly_sorted({"current": [1, 2, 3], "original": [3, 2, 1]})

    def test_rejects_wrong_order(self):
        assert not is_correctly_sorted({"current": [3, 2, 1], "original": [3, 2, 1]})

    def test_rejects_missing_element(self):
        assert not is_correctly_sorted({"current": [1, 2], "original": [3, 2, 1]})

    def test_rejects_missing_keys(self):
        assert not is_correctly_sorted({})
