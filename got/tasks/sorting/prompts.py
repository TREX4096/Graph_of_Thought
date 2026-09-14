"""
Sorting Prompter and Parser.
============================

These two classes are the complete task-specific surface for sorting. The
Prompter encodes *what to ask*; the Parser encodes *how to read the answer*.

Prompt design notes
-------------------
The prompts are deliberately terse and format-strict. With GPT-3.5/4 (the
paper's models) you can afford chatty prompts; with a 1.5B open model you
cannot -- it will pad the answer with prose and break parsing. Every prompt
therefore:

  * states the output format explicitly and last (recency helps compliance),
  * includes one worked example (1-shot) to pin the format down,
  * forbids explanation.

The few-shot example is what the CoT paper (Wei et al.) showed is necessary
to elicit the behaviour at all; we keep it minimal to hold token cost down,
since cost is one of the axes the GoT paper competes on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...prompter import AbstractParser, AbstractPrompter


# ----------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------

SORT_PROMPT = """Sort the following list of numbers in ascending order.
Output only the sorted list, in the same format as the input.
Do not write any explanation, reasoning, or extra text.

Example:
Input: [3, 7, 0, 2, 8, 1, 2, 2, 2, 4, 7, 8, 5, 5, 3, 9]
Output: [0, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9]

Input: {input_list}
Output:"""


# Merging two already-sorted lists is an easier sub-problem than sorting from
# scratch, which is exactly why the merge-sort decomposition helps: each LLM
# call faces a task well within its reliable range.
AGGREGATE_PROMPT = """Merge the following two sorted lists into one single sorted list.
The merged list must contain every element from both input lists, keeping all duplicates.
Output only the merged list. Do not write any explanation.

Example:
Input list 1: [0, 1, 2, 2, 3, 5, 7, 8]
Input list 2: [0, 2, 3, 4, 4, 6, 8, 9]
Output: [0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9]

Input list 1: {list_a}
Input list 2: {list_b}
Output:"""


IMPROVE_PROMPT = """The following list was supposed to be a correctly sorted version of the input list, but it contains mistakes.
It may have elements in the wrong order, missing elements, or duplicated elements.
Fix it so that the output is the input list sorted in ascending order, containing exactly the same elements.
Output only the corrected list. Do not write any explanation.

Input list: {original}
Incorrectly sorted list: {current}
Output:"""


# Used only when running with LLM-based scoring rather than the exact local
# scorer. The local scorer is preferred for sorting (the paper says so), but
# this exists so the LLM-scoring path can be exercised and compared.
SCORE_PROMPT = """You are evaluating an attempt to sort a list of numbers.
Give a score from 0 to 10, where 10 means the output is perfectly sorted and contains exactly the same elements as the input, and 0 means it is completely wrong.
Output only the number. Do not write any explanation.

Input list: {original}
Sorted attempt: {current}
Score:"""


class SortingPrompter(AbstractPrompter):
    """Builds every prompt the sorting GoO needs."""

    def build(
        self, name: str, states: List[Dict[str, Any]], **kwargs
    ) -> Optional[str]:
        # ``split`` is structural, not reasoning: chunking a list is exact
        # Python. Returning None routes it to Parser.parse_local and saves
        # an LLM call per sample.
        if name == "split":
            return None

        if name == "sort":
            return SORT_PROMPT.format(input_list=states[0]["current"])

        if name == "aggregate":
            # Aggregate receives all predecessor states. For merge-sort we
            # merge pairwise, so we expect exactly two.
            if len(states) < 2:
                # Nothing to merge -- fall back to a plain re-sort so the
                # operation still produces a usable thought.
                return SORT_PROMPT.format(input_list=states[0]["current"])
            return AGGREGATE_PROMPT.format(
                list_a=states[0]["current"],
                list_b=states[1]["current"],
            )

        if name == "improve":
            return IMPROVE_PROMPT.format(
                original=states[0].get("original", states[0]["current"]),
                current=states[0]["current"],
            )

        if name == "score":
            return SCORE_PROMPT.format(
                original=states[0].get("original", []),
                current=states[0].get("current", []),
            )

        raise ValueError(f"SortingPrompter has no prompt named {name!r}")


class SortingParser(AbstractParser):
    """Extracts sorted lists from raw model output."""

    def parse(
        self, name: str, states: List[Dict[str, Any]], raw: str, **kwargs
    ) -> Dict[str, Any]:
        """Build the new thought state from the model's answer.

        The ``original`` key
        --------------------
        Every state carries ``original``: the multiset this thought is
        responsible for reproducing. Scoring compares ``current`` against it,
        so it must be defined *recursively down the graph*:

          * a chunk produced by ``split``  -> original = that chunk
          * a sort of a chunk              -> original = parent's original
          * a merge of thoughts A and B    -> original = A.original + B.original

        The recursion matters. If a merged thought inherited only its first
        parent's ``original``, it would be scored against half the data and
        judged catastrophically wrong even when perfectly merged. (That was a
        real bug: GoT scored 0/32 while emitting a near-perfect list.)

        At the root of the merge tree the concatenation of all chunk
        originals reconstitutes the full input multiset, so the final thought
        is scored against exactly the right target.
        """
        parent = states[0]

        if name == "aggregate" and len(states) >= 2:
            # Union of what BOTH parents were responsible for.
            original = list(states[0].get("original", [])) + list(
                states[1].get("original", [])
            )
        else:
            original = list(parent.get("original", parent.get("current", [])))

        # Grouping metadata must survive parsing. ``KeepBestPerGroup`` ranks
        # candidates within their own chunk, so a sorted chunk that lost its
        # ``chunk_index`` would fall into a shared group and all but one
        # chunk would be silently discarded -- the graph would then "sort"
        # only a fraction of the input.
        carried = {
            k: parent[k]
            for k in ("chunk_index", "_group")
            if k in parent
        }

        values = self.extract_list(raw)

        if values is None:
            # Parse failure. Mark invalid and keep the parent's data so the
            # thought is still structurally usable; KeepValid will drop it.
            return {
                **carried,
                "current": list(parent.get("current", [])),
                "original": original,
                "valid": False,
                "raw_response": raw,
            }

        return {
            **carried,
            "current": values,
            "original": original,
            "valid": True,
        }

    def parse_local(
        self, name: str, state: Dict[str, Any], **kwargs
    ) -> List[Dict[str, Any]]:
        """Handle the non-LLM ``split`` step.

        Splits the input list into ``num_chunks`` contiguous chunks. The
        paper's Figure 4 splits 64 elements into four 16-element chunks;
        ``num_chunks`` is passed through from the GoO builder.
        """
        if name != "split":
            return super().parse_local(name, state, **kwargs)

        values: List[int] = list(state["current"])
        num_chunks: int = int(kwargs.get("num_chunks", 4))
        num_chunks = max(1, min(num_chunks, len(values)))

        # Ceiling division so the final chunk absorbs any remainder; this
        # keeps every element exactly once, which term Y of the score
        # depends on.
        size = -(-len(values) // num_chunks)
        chunks = [values[i : i + size] for i in range(0, len(values), size)]

        return [
            {
                "current": chunk,
                "original": chunk,          # each chunk is its own sub-problem
                "root_original": state.get("original", values),
                "chunk_index": i,
                "valid": True,
            }
            for i, chunk in enumerate(chunks)
        ]
