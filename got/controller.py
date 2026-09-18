"""
The Controller -- executes a Graph of Operations.
=================================================

Paper reference
---------------
Section 4.4: "The Controller implements a specific strategy for selecting
thoughts from its GRS structure. It also selects what transformations should
be applied to which thoughts, and then passes this information to the
Prompter. It also decides whether the whole process should be finalized... In
our current design, this is dictated by the execution plan specified in the
GoO."

So the Controller is deliberately *not* clever. All task intelligence lives in
the GoO that the task author builds. The Controller's only jobs are:

  1. Determine a valid execution order for the GoO (topological sort).
  2. Run each operation, threading the LM / Prompter / Parser through.
  3. Maintain the Graph Reasoning State and make it dumpable.

Why topological order
---------------------
The GoO is a DAG of operations. An operation may only run once *all* its
predecessors have produced their thoughts -- Aggregate in particular needs
every branch finished before it can merge them. Kahn's algorithm gives us
that order and simultaneously detects a malformed (cyclic) GoO, which would
otherwise deadlock silently.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .operations import GroundTruth, Operation
from .thought import Thought


class Controller:
    """Runs a Graph of Operations to completion.

    Parameters
    ----------
    lm:
        Any :class:`AbstractLanguageModel`.
    prompter:
        Task-specific Prompter (builds prompt text).
    parser:
        Task-specific Parser (extracts thought state from raw output).
    goo_leaves:
        The terminal operation(s) of the graph. The Controller walks
        backwards from these to discover the whole plan, so the caller only
        has to hand over the end of the pipeline.
    """

    def __init__(
        self,
        lm,
        prompter,
        parser,
        goo_leaves: List[Operation],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.lm = lm
        self.prompter = prompter
        self.parser = parser
        self.goo_leaves = goo_leaves
        self.logger = logger or logging.getLogger("got.Controller")

        self.execution_order: List[Operation] = []
        self.elapsed: float = 0.0

    # ------------------------------------------------------------------
    # Plan discovery and ordering
    # ------------------------------------------------------------------
    def _collect_operations(self) -> List[Operation]:
        """Walk backwards from the leaves to find every operation in the GoO."""
        seen: Dict[int, Operation] = {}
        stack = list(self.goo_leaves)
        while stack:
            op = stack.pop()
            if id(op) in seen:
                continue
            seen[id(op)] = op
            stack.extend(op.predecessors)
        return list(seen.values())

    def _topological_order(self, ops: List[Operation]) -> List[Operation]:
        """Kahn's algorithm over the operation DAG.

        Raises
        ------
        ValueError
            If the GoO contains a cycle. A GoO must be acyclic; note this is
            a constraint on the *plan*, not on the reasoning graph's
            expressiveness -- refinement loops are handled inside ``Improve``.
        """
        indeg = {id(op): 0 for op in ops}
        for op in ops:
            for succ in op.successors:
                if id(succ) in indeg:
                    indeg[id(succ)] += 1

        queue = deque(op for op in ops if indeg[id(op)] == 0)
        order: List[Operation] = []

        while queue:
            op = queue.popleft()
            order.append(op)
            for succ in op.successors:
                if id(succ) not in indeg:
                    continue
                indeg[id(succ)] -= 1
                if indeg[id(succ)] == 0:
                    queue.append(succ)

        if len(order) != len(ops):
            raise ValueError(
                "Graph of Operations contains a cycle; cannot determine "
                "execution order. Check your add_predecessor() wiring."
            )
        return order

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, **kwargs) -> List[Thought]:
        """Execute the entire GoO and return the final thoughts.

        Extra ``kwargs`` are forwarded verbatim to every operation, and from
        there to the Prompter and Parser -- this is how task parameters such
        as ``num_chunks`` reach the prompt builders without the Controller
        needing to know about them.
        """
        ops = self._collect_operations()
        self.execution_order = self._topological_order(ops)

        self.logger.info(
            "Executing GoO with %d operations using %s",
            len(self.execution_order), self.lm.model_name,
        )

        start = time.time()
        for i, op in enumerate(self.execution_order, 1):
            self.logger.debug("[%d/%d] %s", i, len(self.execution_order), op.name)
            op.execute(self.lm, self.prompter, self.parser, **kwargs)
        self.elapsed = time.time() - start

        # The final thoughts are whatever the leaf operations produced.
        final: List[Thought] = []
        for leaf in self.goo_leaves:
            final.extend(leaf.thoughts)
        return final

    # ------------------------------------------------------------------
    # Results and reporting
    # ------------------------------------------------------------------
    def best_thought(self) -> Optional[Thought]:
        """Highest-scoring thought among the leaf outputs."""
        final: List[Thought] = []
        for leaf in self.goo_leaves:
            final.extend(leaf.thoughts)
        if not final:
            return None
        return max(final, key=lambda t: t.score)

    def is_correct(self) -> Optional[bool]:
        """Result of a ``GroundTruth`` node, if the GoO contains one."""
        for op in self.execution_order:
            if isinstance(op, GroundTruth):
                return op.correct
        return None

    def all_thoughts(self) -> List[Thought]:
        """Every thought produced during the run -- i.e. the full GRS.

        Single traversal with one shared visited set. An earlier version
        restarted the ancestor walk for each thought, which re-visited the
        same subgraphs repeatedly and cost O(V*E) on large graphs; with
        aggregation-heavy GoT runs that is a real slowdown when dumping
        results for hundreds of instances.
        """
        seen: Dict[int, Thought] = {}
        stack: List[Thought] = []

        for op in self.execution_order:
            stack.extend(op.thoughts)

        while stack:
            t = stack.pop()
            if t.id in seen:
                continue
            seen[t.id] = t
            # Include ancestors: Score passes thoughts through, so some
            # vertices are reachable only via predecessor links.
            stack.extend(t.predecessors)

        return list(seen.values())

    def graph_summary(self) -> Dict[str, Any]:
        """Structural statistics about the reasoning graph that was built.

        ``n_aggregations`` is the interesting one: if it is zero, the run was
        structurally a tree and gained nothing from GoT.
        """
        thoughts = self.all_thoughts()
        n_edges = sum(len(t.predecessors) for t in thoughts)
        n_invalid = sum(1 for t in thoughts if not t.valid)
        return {
            "n_operations": len(self.execution_order),
            "n_thoughts": len(thoughts),
            "n_edges": n_edges,
            "n_aggregations": sum(1 for t in thoughts if t.is_aggregate),
            # Share of thoughts the Parser could not read. A run can report a
            # plausible-looking accuracy while this is near 1.0 -- that is a
            # broken pipeline, not a weak model, and it is the first number to
            # check when results look bad.
            "n_invalid": n_invalid,
            "invalid_rate": round(n_invalid / len(thoughts), 3) if thoughts else 0.0,
            "elapsed_seconds": round(self.elapsed, 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-serialisable record of the run, for the results files."""
        best = self.best_thought()
        return {
            "graph": self.graph_summary(),
            "usage": self.lm.usage_report(),
            "correct": self.is_correct(),
            "best_score": best.score if best else None,
            "best_state": best.state if best else None,
            "thoughts": [t.to_dict() for t in self.all_thoughts()],
        }

    def dump(self, path: str) -> None:
        """Write :meth:`to_dict` to ``path`` as indented JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        self.logger.info("wrote run record to %s", path)
