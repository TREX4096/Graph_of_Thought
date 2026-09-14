"""
Set-intersection scoring -- GoT Section 5.2.

    error-scope = X1 + X2 + Xd

with X1 spurious elements, X2 missing elements, Xd duplicates.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence


def intersection_error_scope(
    set_a: Sequence[int], set_b: Sequence[int], produced: Sequence[int]
) -> int:
    """Compute ``X1 + X2 + Xd`` for an intersection attempt.

    Parameters
    ----------
    set_a, set_b:
        The two input sets (as lists -- the data files store them ordered).
    produced:
        The model's output list.
    """
    truth = set(set_a) & set(set_b)
    produced = list(produced)
    produced_set = set(produced)

    # X1: things the model invented or wrongly included.
    x1 = len(produced_set - truth)
    # X2: things the model failed to find.
    x2 = len(truth - produced_set)
    # Xd: repeats -- a set expressed as a list can contain duplicates.
    counts = Counter(produced)
    xd = sum(c - 1 for c in counts.values() if c > 1)

    return x1 + x2 + xd


def intersection_score(state: Dict[str, Any]) -> float:
    """Higher-is-better score: ``max(n - error_scope, 0)``.

    ``n`` is the size of set A, matching the paper's convention of scaling by
    the input size so scores are comparable across set sizes.
    """
    produced = state.get("current")
    set_a = state.get("set_a")
    set_b = state.get("set_b")

    if produced is None or set_a is None or set_b is None:
        return 0.0

    n = len(set_a)
    return float(max(n - intersection_error_scope(set_a, set_b, produced), 0))


def is_correct_intersection(state: Dict[str, Any]) -> bool:
    """Strict ground truth: output is exactly the intersection, no duplicates."""
    produced = state.get("current")
    set_a = state.get("set_a")
    set_b = state.get("set_b")
    if produced is None or set_a is None or set_b is None:
        return False
    return sorted(set(produced)) == sorted(set(set_a) & set(set_b)) and \
        len(produced) == len(set(produced))
