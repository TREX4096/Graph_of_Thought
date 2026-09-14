"""
Thought transformations -- the vocabulary of the Graph of Operations (GoO).
===========================================================================

Paper reference
---------------
Section 3.2 defines three *graph-enabled thought transformations*:

  * **Generation**  (Sec 3.2): from one thought v, produce k new thoughts.
        V+ = {v1+, ..., vk+},  E+ = {(v, v1+), ..., (v, vk+)}
        This subsumes CoT's single step and ToT's branching.

  * **Aggregation** (Sec 3.2): merge k thoughts into one.
        V+ = {v+},  E+ = {(v1, v+), ..., (vk, v+)}
        *This is the transformation that CoT and ToT cannot express.* It is
        the whole reason the structure must be a graph rather than a tree.

  * **Refinement**  (Sec 3.2): improve a thought in place.
        V+ = {},  E+ = {(v, v)}
        A self-loop -- also impossible in a tree (trees are acyclic and each
        node has one parent).

Section 3.3 adds scoring E(v, G, p_theta) and ranking R(G, p_theta, h), which
we implement as the ``Score`` and ``KeepBest`` operations.

Architectural note: GoO vs. GRS
--------------------------------
Section 4.5 draws a sharp line:

  * **GoO** (Graph of Operations) is *static*. It is the execution plan, built
    once before the run. Each ``Operation`` object here is a GoO node, and
    ``predecessors``/``successors`` wire the plan together.
  * **GRS** (Graph Reasoning State) is *dynamic*. It is the thoughts actually
    produced during execution, held in ``Operation.thoughts``.

Keeping the plan separate from the state is what makes a GoO reusable across
many input samples: the plan is rebuilt per sample but its *shape* is fixed by
the task, while the thoughts differ every time.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List, Optional

from .thought import Thought


class Operation(abc.ABC):
    """Base class for every node in the Graph of Operations.

    An operation consumes the thoughts produced by its predecessor
    operations and produces its own list of thoughts.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self.predecessors: List["Operation"] = []
        self.successors: List["Operation"] = []

        # The GRS slice owned by this operation: thoughts produced when it ran.
        self.thoughts: List[Thought] = []
        self.executed: bool = False
        self.logger = logging.getLogger(f"got.op.{self.name}")

    # ------------------------------------------------------------------
    # GoO wiring
    # ------------------------------------------------------------------
    def add_predecessor(self, other: "Operation") -> "Operation":
        """Declare that ``other`` must run before this operation.

        Returns ``self`` so calls can be chained when building a GoO.
        """
        if other not in self.predecessors:
            self.predecessors.append(other)
        if self not in other.successors:
            other.successors.append(self)
        return self

    def get_input_thoughts(self) -> List[Thought]:
        """Collect every thought emitted by predecessor operations.

        With no predecessors this returns ``[]`` -- the operation is a source
        and must obtain its data some other way (see ``InputOp``).
        """
        gathered: List[Thought] = []
        for pred in self.predecessors:
            gathered.extend(pred.thoughts)
        return gathered

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        """Run this operation and cache its thoughts in the GRS."""
        if self.executed:
            return self.thoughts
        self.thoughts = self._execute(lm, prompter, parser, **kwargs)
        self.executed = True
        self.logger.debug("%s produced %d thoughts", self.name, len(self.thoughts))
        return self.thoughts

    @abc.abstractmethod
    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        """Subclass hook doing the actual work."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.name} thoughts={len(self.thoughts)}>"


# ======================================================================
# Source operation
# ======================================================================
class InputOp(Operation):
    """Injects the initial problem instance into the graph.

    Every GoO starts here. The thought it emits is the "Input" vertex drawn
    at the top of every diagram in Figure 1 of the paper.
    """

    def __init__(self, state: Dict[str, Any], name: Optional[str] = None) -> None:
        super().__init__(name)
        self.state = state

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        # No LLM call -- this is pure data injection.
        return [Thought(state=dict(self.state), operation="input")]


# ======================================================================
# Generation transformations
# ======================================================================
class Generate(Operation):
    """Generation transformation: 1 thought -> k new thoughts.

    Paper: "one can generate one or more new thoughts based on an existing
    single thought v" (Section 3.2).

    Parameters
    ----------
    prompt_name:
        Which Prompter method to invoke (e.g. ``"sort"``, ``"split"``).
        The Prompter owns all task-specific wording; this operation stays
        generic.
    branching_factor:
        ``k`` -- how many independent samples to draw per input thought.
        k=1 makes this a plain CoT step; k>1 gives ToT-style branching.
    """

    def __init__(
        self,
        prompt_name: str = "generate",
        branching_factor: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"Generate({prompt_name},k={branching_factor})")
        self.prompt_name = prompt_name
        self.branching_factor = branching_factor

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        produced: List[Thought] = []

        for parent in inputs:
            prompt = prompter.build(self.prompt_name, [parent.state], **kwargs)

            # A prompt may be "pure" -- i.e. deterministic local work that
            # needs no model at all (splitting a list into chunks is plain
            # Python). Returning None from the Prompter signals this and
            # saves a pointless LLM call. The paper's split step is likewise
            # a structural decomposition, not a reasoning step.
            if prompt is None:
                new_states = parser.parse_local(self.prompt_name, parent.state, **kwargs)
                for st in new_states:
                    t = Thought(
                        state=st,
                        valid=bool(st.get("valid", True)),
                        operation=self.prompt_name,
                    )
                    t.add_predecessor(parent)
                    produced.append(t)
                continue

            responses = lm.query(prompt, num_responses=self.branching_factor)
            for raw in responses:
                st = parser.parse(self.prompt_name, [parent.state], raw, **kwargs)
                # The Parser reports structural validity via state["valid"];
                # lift it onto the Thought so KeepValid/KeepBest can act on it.
                t = Thought(
                    state=st,
                    valid=bool(st.get("valid", True)),
                    operation=self.prompt_name,
                )
                t.add_predecessor(parent)
                produced.append(t)

        return produced


# ======================================================================
# Aggregation transformation -- the defining feature of GoT
# ======================================================================
class Aggregate(Operation):
    """Aggregation transformation: k thoughts -> 1 (or k') merged thought(s).

    Paper: "one can aggregate arbitrary thoughts into new ones, to combine
    and reinforce the advantages of these thoughts, while eliminating their
    disadvantages" (Section 3.2).

    Structurally this creates a vertex whose in-degree exceeds one. That
    single fact is what forces the reasoning structure to be a *graph*: a
    tree cannot represent it. It is also what buys GoT its volume advantage
    in Section 6 -- because every earlier thought can reach the final thought
    through an aggregation, volume rises to N while latency stays log_k N.

    Parameters
    ----------
    num_merges:
        How many *different* merge attempts to make (the paper's k=10 for
        "we try 10 different aggregations of the two input 16-element
        subarrays"). Each attempt becomes its own thought; a later KeepBest
        picks the winner.
    """

    def __init__(
        self,
        prompt_name: str = "aggregate",
        num_merges: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"Aggregate({prompt_name},k={num_merges})")
        self.prompt_name = prompt_name
        self.num_merges = num_merges

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        if not inputs:
            return []

        # All predecessor thoughts feed into every merge attempt.
        states = [t.state for t in inputs]
        prompt = prompter.build(self.prompt_name, states, **kwargs)

        if prompt is None:
            merged_states = parser.aggregate_local(self.prompt_name, states, **kwargs)
            responses_states = merged_states
        else:
            responses = lm.query(prompt, num_responses=self.num_merges)
            responses_states = [
                parser.parse(self.prompt_name, states, raw, **kwargs)
                for raw in responses
            ]

        produced: List[Thought] = []
        for st in responses_states:
            t = Thought(
                state=st,
                valid=bool(st.get("valid", True)),
                operation=self.prompt_name,
            )
            # Wire an edge from EVERY input -- this is the in-degree > 1 that
            # makes the structure a graph.
            for parent in inputs:
                t.add_predecessor(parent)
            produced.append(t)

        return produced


# ======================================================================
# Refinement transformation
# ======================================================================
class Improve(Operation):
    """Refinement transformation: improve a thought, keeping its identity.

    Paper: "the refining of a current thought v by modifying its content:
    V+ = {} and E+ = {(v, v)}. This loop in the graph indicates an iterated
    thought" (Section 3.2).

    We materialise the refined result as a new vertex whose predecessor is
    the original. Semantically this is the paper's self-loop; representing it
    as a fresh vertex keeps the graph acyclic, which makes traversal (and the
    volume computation) far simpler while preserving the full provenance
    chain. The distinction is bookkeeping, not behaviour.

    Parameters
    ----------
    rounds:
        How many successive refinement passes to apply.
    keep_best_only:
        If True, only a refinement that scores at least as well as its parent
        replaces it. This prevents the well-documented failure mode where
        self-refinement degrades an already-good answer.
    """

    def __init__(
        self,
        prompt_name: str = "improve",
        rounds: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"Improve({prompt_name},r={rounds})")
        self.prompt_name = prompt_name
        self.rounds = rounds

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        current = self.get_input_thoughts()
        produced: List[Thought] = []

        for parent in current:
            node = parent
            for _ in range(self.rounds):
                prompt = prompter.build(self.prompt_name, [node.state], **kwargs)
                if prompt is None:
                    break
                raw = lm.query(prompt, num_responses=1)[0]
                st = parser.parse(self.prompt_name, [node.state], raw, **kwargs)
                refined = Thought(
                    state=st,
                    valid=bool(st.get("valid", True)),
                    operation=self.prompt_name,
                )
                refined.add_predecessor(node)
                node = refined
            produced.append(node)

        return produced


# ======================================================================
# Scoring and ranking (Section 3.3)
# ======================================================================
class Score(Operation):
    """Assign a score to every incoming thought.

    Paper: "A score is modeled as a general function E(v, G, p_theta)"
    (Section 3.3). Two scoring modes exist, matching Section 4.3:

      * **local** -- a deterministic Python function. Sorting and set
        intersection use this ("use cases such as sorting use simple local
        scoring functions"). Free and exact.
      * **LLM-based** -- ask the model to rate the thought. Document merging
        needs this, since redundancy and information retention have no
        closed-form measure.

    Parameters
    ----------
    scoring_fn:
        Callable ``state -> float`` for local scoring. If None, the LLM is
        queried via ``prompt_name`` instead.
    n_votes:
        For LLM scoring, how many times to ask and average. The paper queries
        "3 times for each value, and take the average" for document merging,
        because single LLM judgements are noisy.
    """

    def __init__(
        self,
        scoring_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        prompt_name: str = "score",
        n_votes: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "Score")
        self.scoring_fn = scoring_fn
        self.prompt_name = prompt_name
        self.n_votes = n_votes

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()

        for t in inputs:
            if self.scoring_fn is not None:
                # Deterministic local scoring -- no model call, no cost.
                t.score = float(self.scoring_fn(t.state))
            else:
                prompt = prompter.build(self.prompt_name, [t.state], **kwargs)
                raws = lm.query(prompt, num_responses=self.n_votes)
                values = [parser.parse_score(raw) for raw in raws]
                values = [v for v in values if v is not None]
                # Average the votes; fall back to 0.0 if the model returned
                # nothing parseable rather than crashing a long HPC run.
                t.score = sum(values) / len(values) if values else 0.0
            t.scored = True

        # Score annotates thoughts in place and passes them straight through,
        # so downstream operations see the same vertices with scores attached.
        return inputs


class KeepBest(Operation):
    """Ranking: keep the ``n`` highest-scoring thoughts.

    Paper: "we most often use a simple yet effective strategy where h
    thoughts with the highest scores are returned" (Section 3.3). In the
    sorting GoO this is the ``KeepBest(N=1)`` node that collapses k=3
    candidate sortings down to the single best one.

    Emitting *copies* rather than the originals keeps the selection visible
    in the graph, so a reader of the dumped graph can see where pruning
    happened.
    """

    def __init__(self, n: int = 1, name: Optional[str] = None) -> None:
        super().__init__(name or f"KeepBest(n={n})")
        self.n = n

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = [t for t in self.get_input_thoughts() if t.valid]
        if not inputs:
            return []

        ranked = sorted(inputs, key=lambda t: t.score, reverse=True)
        kept = []
        for src in ranked[: self.n]:
            c = src.copy(operation="keepbest")
            c.add_predecessor(src)
            kept.append(c)
        return kept


class KeepValid(Operation):
    """Filter out thoughts that failed structural validation.

    Paper, Section 4.3: "we verify whether a given LLM thought satisfies
    potential correctness conditions". Malformed model output (prose instead
    of a list, truncated JSON) is common with smaller open models, so this
    guard matters far more in our open-source setting than it does with
    GPT-4.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name or "KeepValid")

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        return [t for t in self.get_input_thoughts() if t.valid]


class Selector(Operation):
    """Pass through a caller-chosen subset of incoming thoughts.

    Needed when a GoO branches: e.g. the sorting graph splits one input into
    four chunks and then must route chunk *i* to the *i*-th sorting subgraph.
    A plain edge would hand every chunk to every branch.
    """

    def __init__(
        self,
        selector: Callable[[List[Thought]], List[Thought]],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "Selector")
        self.selector = selector

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        return self.selector(self.get_input_thoughts())


class GroundTruth(Operation):
    """Terminal check against the known answer -- evaluation only.

    This never influences the reasoning; it exists so a benchmark harness can
    record whether the final thought was correct. Keeping it as an explicit
    GoO node (as the reference implementation does) means the check is
    recorded in the graph dump alongside everything else.
    """

    def __init__(
        self,
        check_fn: Callable[[Dict[str, Any]], bool],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "GroundTruth")
        self.check_fn = check_fn
        self.correct: Optional[bool] = None

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        if not inputs:
            self.correct = False
            return []
        # Judge the top-ranked thought -- by this point a KeepBest has
        # normally reduced the frontier to one.
        best = max(inputs, key=lambda t: t.score)
        self.correct = bool(self.check_fn(best.state))
        return inputs
