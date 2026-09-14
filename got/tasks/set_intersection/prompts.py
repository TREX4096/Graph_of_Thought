"""Prompter and Parser for set intersection (GoT Section 5.2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...prompter import AbstractParser, AbstractPrompter


INTERSECT_PROMPT = """Find the intersection of the two lists below: output every number that appears in BOTH lists.
Output only the resulting list. Do not repeat any number. Do not write any explanation.

Example:
Input list 1: [11, 3, 7, 2, 9, 5]
Input list 2: [4, 9, 11, 8, 5, 1]
Output: [5, 9, 11]

Input list 1: {list_a}
Input list 2: {list_b}
Output:"""


# The union step. Kept as an LLM call rather than plain Python because the
# paper's pipeline aggregates through the model -- and because it exercises
# the same Aggregate machinery the sorting task uses.
UNION_PROMPT = """Combine the two lists below into a single list containing every number that appears in either list.
Remove any duplicates. Output only the resulting list. Do not write any explanation.

Example:
Input list 1: [5, 9]
Input list 2: [9, 11, 2]
Output: [2, 5, 9, 11]

Input list 1: {list_a}
Input list 2: {list_b}
Output:"""


class SetIntersectionPrompter(AbstractPrompter):
    """Builds prompts for the intersection pipeline."""

    def build(
        self, name: str, states: List[Dict[str, Any]], **kwargs
    ) -> Optional[str]:
        # Splitting set B into subsets is exact Python -- no LLM needed.
        if name == "split":
            return None

        if name == "intersect":
            st = states[0]
            return INTERSECT_PROMPT.format(
                list_a=st["set_a"], list_b=st["current"]
            )

        if name == "aggregate":
            if len(states) < 2:
                return None
            return UNION_PROMPT.format(
                list_a=states[0]["current"], list_b=states[1]["current"]
            )

        raise ValueError(f"SetIntersectionPrompter has no prompt named {name!r}")


class SetIntersectionParser(AbstractParser):
    """Extracts intersection results from raw model output."""

    def parse(
        self, name: str, states: List[Dict[str, Any]], raw: str, **kwargs
    ) -> Dict[str, Any]:
        parent = states[0]
        values = self.extract_list(raw)

        # ``set_a`` and ``set_b`` are the *global* inputs and must survive
        # every hop so the final thought can be scored against the true
        # intersection rather than against some sub-problem.
        base = {
            "set_a": parent.get("set_a", []),
            "set_b": parent.get("set_b", []),
        }

        if values is None:
            return {**base, "current": parent.get("current", []),
                    "valid": False, "raw_response": raw}

        return {**base, "current": values, "valid": True}

    def parse_local(
        self, name: str, state: Dict[str, Any], **kwargs
    ) -> List[Dict[str, Any]]:
        """Split set B into contiguous subsets; set A is left whole.

        Only B is split: each subset is intersected against the complete A,
        and the results are unioned. See the module docstring for why this is
        exact.
        """
        if name != "split":
            return super().parse_local(name, state, **kwargs)

        set_b: List[int] = list(state["set_b"])
        num_chunks = int(kwargs.get("num_chunks", 4))
        num_chunks = max(1, min(num_chunks, len(set_b)))

        size = -(-len(set_b) // num_chunks)
        chunks = [set_b[i : i + size] for i in range(0, len(set_b), size)]

        return [
            {
                "set_a": list(state["set_a"]),
                "set_b": list(state["set_b"]),
                "current": chunk,         # this branch's slice of B
                "chunk_index": i,
                "valid": True,
            }
            for i, chunk in enumerate(chunks)
        ]

    def aggregate_local(
        self, name: str, states: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Exact union fallback, used when only one state reaches an Aggregate."""
        merged: List[int] = []
        for st in states:
            for v in st.get("current", []):
                if v not in merged:
                    merged.append(v)
        return [{
            "set_a": states[0].get("set_a", []),
            "set_b": states[0].get("set_b", []),
            "current": merged,
            "valid": True,
        }]
