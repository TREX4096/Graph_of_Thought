"""
Sorting scoring -- a faithful implementation of GoT Section 5.1.
================================================================

The paper defines the "error scope" of a produced sequence as::

    error-scope = X + Y

    X = sum_{i=1}^{m-1} sgn(max(b_i - b_{i+1}, 0))
    Y = sum_{i=0}^{9} | |{b_p : b_p = i}| - |{a_q : a_q = i}| |

where ``[a_1..a_n]`` is the input sequence and ``[b_1..b_m]`` the output.

Reading the two terms
---------------------
**X -- sortedness.** For each adjacent pair, ``sgn(max(b_i - b_{i+1}, 0))``
is 1 exactly when ``b_i > b_{i+1}`` (a descent, i.e. a local sorting error)
and 0 otherwise. So X counts *inversions between neighbours* -- not total
inversions. A sequence sorted except for one swapped pair scores X = 1.

**Y -- multiset preservation.** For every digit 0-9, compare how many times it
appears in the output versus the input. Any discrepancy -- a dropped element,
a hallucinated duplicate -- adds to Y. This is the term that catches the
failure mode the paper actually observed: "the considered LLMs are unable to
sort a sequence of such numbers correctly beyond a certain length
consistently *because duplicate counts do not match*."

Why both terms are needed: an LLM could return a perfectly ascending sequence
that has quietly lost three elements (X = 0, but wrong). Or it could return
the right multiset in the wrong order (Y = 0, but wrong). Only X + Y = 0
means genuinely correct.

Sign convention
---------------
The paper notes: "to use a 'positive score' describing 'the scope of
correctly sorted' elements, one can use the value max(n - error-scope, 0)."
Our framework ranks thoughts by *descending* score, so we use that positive
form everywhere. ``sorting_error_scope`` returns the raw error for analysis
and plotting; ``sorting_score`` returns the higher-is-better version used by
the Score operation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def sorting_error_scope(
    original: Sequence[int],
    produced: Sequence[int],
    clip: bool = False,
) -> int:
    """Compute ``error-scope = X + Y`` for a sorting attempt.

    Parameters
    ----------
    original:
        The input sequence ``[a_1 ... a_n]``.
    produced:
        The model's output sequence ``[b_1 ... b_m]``.
    clip:
        Apply the paper's plotting convenience ``min(error-scope, n)``. The
        paper clips only "to improve the clarity of plots, as some baselines
        (IO, CoT) result in large numbers of outliers with high error scope",
        so this defaults to off for raw analysis.

    Returns
    -------
    Non-negative integer; 0 means a perfect sort.
    """
    produced = list(produced)
    original = list(original)

    # --- X: count adjacent descents ---------------------------------
    # sgn(max(b_i - b_{i+1}, 0)) == 1 iff b_i > b_{i+1}
    x = sum(
        1
        for i in range(len(produced) - 1)
        if produced[i] > produced[i + 1]
    )

    # --- Y: multiset (frequency) mismatch ---------------------------
    # The paper sums over digits 0-9. We sum over the union of values
    # actually present, which is equivalent for 0-9 data and additionally
    # correct if a model hallucinates an out-of-range value.
    c_out = Counter(produced)
    c_in = Counter(original)
    y = sum(
        abs(c_out.get(v, 0) - c_in.get(v, 0))
        for v in set(c_in) | set(c_out)
    )

    error = x + y
    if clip:
        error = min(error, len(original))
    return error


def sorting_score(state: Dict[str, Any]) -> float:
    """Higher-is-better score for a sorting thought, as used by ``Score``.

    Implements ``max(n - error-scope, 0)`` from Section 5.1.

    An invalid thought (the parser could not find a list at all) scores 0 --
    the worst possible value -- so KeepBest will never select it over any
    parseable candidate.
    """
    produced: Optional[List[int]] = state.get("current")
    original: Optional[List[int]] = state.get("original")

    if produced is None or original is None:
        return 0.0

    n = len(original)
    error = sorting_error_scope(original, produced)
    return float(max(n - error, 0))


def is_correctly_sorted(state: Dict[str, Any]) -> bool:
    """Ground-truth check: is the output exactly ``sorted(original)``?

    Used by the ``GroundTruth`` operation for benchmark accuracy. This is
    strict -- the paper reports both "exactly correct" rates and error-scope
    distributions, and we keep both available.
    """
    produced = state.get("current")
    original = state.get("original")
    if produced is None or original is None:
        return False
    return list(produced) == sorted(original)
