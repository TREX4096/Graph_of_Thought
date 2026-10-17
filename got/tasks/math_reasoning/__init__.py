"""
Math reasoning tasks: GSM8K and MATH-500.
=========================================

These are the benchmarks used by the step-entropy paper (Li et al., ICLR
2026, arXiv:2508.03346), and they differ from sorting in a way that
matters for this project.

Sorting has *decomposable* structure - you can split a list, sort the
pieces and merge them, which is what makes a reasoning graph natural. A
GSM8K word problem does not decompose that way: there is one question and
one scalar answer, and the reasoning is intrinsically a chain.

That makes them exactly the right place to test the entropy criterion in
isolation. On a chain, entropy is the only thing under study; no graph
structure confounds the measurement. Once the criterion is validated here,
the open question becomes whether it transfers to graph vertices - which
is the experiment that connects this paper to our contribution.

Scoring is exact: both datasets ship a reference final answer, so no LLM
judge is needed. The comparison is more delicate than sorting's, though -
MATH-500 answers are LaTeX expressions, where "1/2", "\\frac{1}{2}" and
"0.5" are the same answer written three ways.
"""

from .scoring import answers_match, extract_answer, normalise_answer
from .prompts import (
    ANSWER_ONLY_PROMPT,
    COT_PROMPT,
    build_compressed_prompt,
)

__all__ = [
    "answers_match", "extract_answer", "normalise_answer",
    "COT_PROMPT", "ANSWER_ONLY_PROMPT", "build_compressed_prompt",
]
