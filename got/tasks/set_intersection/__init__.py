"""
Set intersection use case (GoT paper, Section 5.2).
===================================================

Paper: "Set intersection of two sets is implemented similarly as the sorting.
The second input set is split into subsets and the intersection of those
subsets with the first input set is determined with the help of the LLM.
Afterwards the resulting intersection sets are aggregated for the final
results."

The decomposition differs from sorting in an instructive way. In sorting we
split *the* input; here we split only set B and intersect each piece against
the whole of A. That works because intersection distributes over union:

    A INTERSECT (B1 UNION B2 UNION ... UNION Bm)
      = (A INTERSECT B1) UNION (A INTERSECT B2) UNION ... UNION (A INTERSECT Bm)

So the aggregation step is a *union*, not a merge -- and it is exact. This is
a good illustration of the paper's general point: the right graph
decomposition follows from the algebraic structure of the problem, and GoT
gives you a way to express whatever that structure turns out to be.

Scoring (Section 5.2)
---------------------
    error-scope = X1 + X2 + Xd

    X1 = |C \\ (A INTERSECT B)|   elements in the output that do not belong
    X2 = |(A INTERSECT B) \\ C|   elements missing from the output
    Xd = number of duplicates in C

Xd exists "because the LLM expresses the set as a list in natural language" --
a list can repeat an element even though a set cannot.
"""

from .scoring import intersection_error_scope, intersection_score, is_correct_intersection
from .prompts import SetIntersectionPrompter, SetIntersectionParser
from .graphs import (
    cot_intersection_goo,
    cot_sc_intersection_goo,
    got_intersection_goo,
    io_intersection_goo,
    tot_intersection_goo,
    SCHEMES,
)

__all__ = [
    "intersection_error_scope", "intersection_score", "is_correct_intersection",
    "SetIntersectionPrompter", "SetIntersectionParser",
    "cot_intersection_goo", "cot_sc_intersection_goo", "got_intersection_goo",
    "io_intersection_goo", "tot_intersection_goo", "SCHEMES",
]
