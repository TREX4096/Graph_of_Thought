"""
The Thought -- a vertex in the Graph of Thoughts.
================================================

Paper reference
---------------
GoT, Section 3.1 ("Reasoning Process"):

    "We model the reasoning process as a directed graph G = (V, E) ... A
    vertex contains a solution to a problem at hand (be it an initial,
    intermediate, or a final one). The concrete form of such a thought
    depends on the use case; it could be a paragraph (in writing tasks) or a
    sequence of numbers (in sorting). A directed edge (t1, t2) indicates that
    thought t2 has been constructed using t1 as 'direct input'."

Design decision: ``state`` is a free-form dict
----------------------------------------------
The paper's Parser "constructs the thought state, which contains this
extracted information" (Section 4.2). Because a thought's content is entirely
task-dependent -- a list of ints for sorting, a dict of counts for keyword
counting, a string for document merging -- we do not impose a typed schema.
Each task owns the keys it puts in ``state``. This mirrors the reference
implementation and keeps the core framework genuinely task-agnostic.

Validity vs. score -- why they are separate
-------------------------------------------
Section 4.3 distinguishes *validation* ("verify whether a given LLM thought
satisfies potential correctness conditions") from *scoring* ("and then we
assign it a score"). A thought can be well-formed but poor (valid, low score),
or malformed entirely (invalid -- e.g. the LLM returned prose where a list was
required). Keeping the two separate lets the Controller discard garbage
without letting it pollute the ranking.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional


class Thought:
    """A single vertex in the reasoning graph.

    Attributes
    ----------
    id:
        Unique, monotonically increasing integer. Used for graph edges,
        logging and visualisation.
    state:
        Task-specific payload produced by the Parser. For sorting this holds
        e.g. ``{"current": [1, 2, 3], "original": [...]}``.
    score:
        Numeric quality assigned by the Scoring module. Higher is better by
        convention throughout this codebase; tasks whose natural metric is an
        error convert it (the paper does exactly this:
        ``max(n - error_scope, 0)``).
    valid:
        Whether the thought passed structural validation.
    predecessors / successors:
        Graph edges. A thought with more than one predecessor is the result
        of an **Aggregation** -- the transformation unique to GoT.
    operation:
        Name of the operation that produced this thought, for provenance.
    """

    # Class-level counter guaranteeing unique ids within a process.
    _id_counter = itertools.count()

    def __init__(
        self,
        state: Optional[Dict[str, Any]] = None,
        score: float = 0.0,
        valid: bool = True,
        scored: bool = False,
        operation: str = "init",
    ) -> None:
        self.id: int = next(Thought._id_counter)
        self.state: Dict[str, Any] = dict(state or {})
        self.score: float = score
        self.valid: bool = valid
        self.scored: bool = scored
        self.operation: str = operation

        self.predecessors: List["Thought"] = []
        self.successors: List["Thought"] = []

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------
    def add_predecessor(self, other: "Thought") -> None:
        """Record that ``other`` was a direct input to this thought.

        Maintains both sides of the edge so the graph can be traversed in
        either direction (needed for the volume metric, which walks
        backwards from a thought).
        """
        if other not in self.predecessors:
            self.predecessors.append(other)
        if self not in other.successors:
            other.successors.append(self)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_aggregate(self) -> bool:
        """True if this thought merged several parents.

        This is the structural signature of GoT: CoT and ToT graphs never
        contain a vertex with in-degree > 1.
        """
        return len(self.predecessors) > 1

    def copy(self, operation: str = "copy") -> "Thought":
        """Create a new thought carrying the same state.

        Used by operations that pass data through unchanged (e.g. KeepBest
        emits fresh vertices so the graph records the selection step).
        """
        new = Thought(
            state=dict(self.state),
            score=self.score,
            valid=self.valid,
            scored=self.scored,
            operation=operation,
        )
        return new

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary, used when dumping run results."""
        return {
            "id": self.id,
            "operation": self.operation,
            "score": self.score,
            "valid": self.valid,
            "scored": self.scored,
            "is_aggregate": self.is_aggregate,
            "predecessors": [p.id for p in self.predecessors],
            "state": {
                k: v for k, v in self.state.items()
                # Keep dumps readable: skip bulky raw LLM text.
                if k not in ("raw_response",)
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        flag = "" if self.valid else " INVALID"
        return f"<Thought#{self.id} op={self.operation} score={self.score:.3g}{flag}>"
