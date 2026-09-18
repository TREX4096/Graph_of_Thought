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

Cost discipline (matters on HPC, where GPU seconds are the budget)
-------------------------------------------------------------------
Every operation that talks to the model does so through **one batched call**
(``lm.query_batch``) covering all of its inputs, rather than one call per
input. On a GPU backend this is the difference between a saturated device and
an idle one.

To get the full benefit, a GoO should put sibling work in **one operation with
many input thoughts** rather than in many parallel single-input operations --
the Controller executes operations strictly one at a time, so four
single-input operations cannot batch with each other, while one operation
holding four thoughts can. ``KeepBestPerGroup`` and ``PairwiseAggregate``
exist to make that style expressible; see ``got/tasks/sorting/graphs.py``.

Each operation also carries a ``max_tokens`` budget and optional ``stop``
strings. Decode time is roughly linear in tokens produced, so capping a step
that needs 60 tokens at 60 rather than the global 1024 is a direct multiplier
on cost.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

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

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------
    @staticmethod
    def _new_thought(state: Dict[str, Any], operation: str, parents) -> Thought:
        """Create a thought, lift the Parser's validity flag, and wire edges."""
        t = Thought(
            state=state,
            valid=bool(state.get("valid", True)),
            operation=operation,
        )
        for p in parents:
            t.add_predecessor(p)
        return t

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
    """Generation transformation: each input thought -> k new thoughts.

    Paper: "one can generate one or more new thoughts based on an existing
    single thought v" (Section 3.2).

    All input thoughts are processed in **one batched model call**, so a
    single ``Generate`` holding four chunk thoughts costs one GPU round trip
    rather than four.

    Parameters
    ----------
    prompt_name:
        Which Prompter step to invoke (e.g. ``"sort"``, ``"split"``).
        The Prompter owns all task-specific wording; this operation stays
        generic.
    branching_factor:
        ``k`` -- how many independent samples to draw per input thought.
        k=1 makes this a plain CoT step; k>1 gives ToT-style branching.
    max_tokens:
        Cap on generated tokens for this step. Set it from what the step
        actually needs; ``None`` falls back to the backend default.
    stop:
        Stop strings ending generation early once the answer is complete.
    """

    def __init__(
        self,
        prompt_name: str = "generate",
        branching_factor: int = 1,
        name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name or f"Generate({prompt_name},k={branching_factor})")
        self.prompt_name = prompt_name
        self.branching_factor = branching_factor
        self.max_tokens = max_tokens
        self.stop = stop

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        produced: List[Thought] = []

        # Split inputs into local (free) work and work needing the model.
        llm_parents: List[Thought] = []
        llm_prompts: List[str] = []

        for parent in inputs:
            prompt = prompter.build(self.prompt_name, [parent.state], **kwargs)

            # A prompt may be "pure" -- deterministic local work needing no
            # model (splitting a list into chunks is plain Python). Returning
            # None from the Prompter signals this and saves an LLM call. The
            # paper's split step is likewise a structural decomposition, not
            # a reasoning step.
            if prompt is None:
                for st in parser.parse_local(self.prompt_name, parent.state, **kwargs):
                    produced.append(self._new_thought(st, self.prompt_name, [parent]))
                continue

            llm_parents.append(parent)
            llm_prompts.append(prompt)

        # One batched call for every prompt this operation needs.
        if llm_prompts:
            batched = lm.query_batch(
                llm_prompts,
                num_responses=self.branching_factor,
                max_tokens=self.max_tokens,
                stop=self.stop,
            )
            for parent, responses in zip(llm_parents, batched):
                for raw in responses:
                    st = parser.parse(self.prompt_name, [parent.state], raw, **kwargs)
                    produced.append(self._new_thought(st, self.prompt_name, [parent]))

        return produced


# ======================================================================
# Aggregation transformation -- the defining feature of GoT
# ======================================================================
class Aggregate(Operation):
    """Aggregation transformation: all input thoughts -> merged thought(s).

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
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name or f"Aggregate({prompt_name},k={num_merges})")
        self.prompt_name = prompt_name
        self.num_merges = num_merges
        self.max_tokens = max_tokens
        self.stop = stop

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        if not inputs:
            return []

        # All predecessor thoughts feed into every merge attempt.
        states = [t.state for t in inputs]
        prompt = prompter.build(self.prompt_name, states, **kwargs)

        if prompt is None:
            new_states = parser.aggregate_local(self.prompt_name, states, **kwargs)
        else:
            responses = lm.query_batch(
                [prompt],
                num_responses=self.num_merges,
                max_tokens=self.max_tokens,
                stop=self.stop,
            )[0]
            new_states = [
                parser.parse(self.prompt_name, states, raw, **kwargs)
                for raw in responses
            ]

        # Wire an edge from EVERY input -- this is the in-degree > 1 that
        # makes the structure a graph.
        return [
            self._new_thought(st, self.prompt_name, inputs) for st in new_states
        ]


class PairwiseAggregate(Operation):
    """Aggregate consecutive *pairs* of input thoughts, all in one batch.

    Why this exists
    ---------------
    A merge-sort style GoO needs several independent merges per level
    (chunk0+chunk1, chunk2+chunk3, ...). Expressing each as its own
    ``Aggregate`` operation is correct but wasteful on a GPU: the Controller
    runs operations one at a time, so those merges execute serially and each
    submits a batch of one.

    ``PairwiseAggregate`` performs the whole level as a single operation, so
    every pair's prompt goes to the model in **one batched call**. The
    resulting graph is identical -- each merged thought still has in-degree 2
    and is still a genuine aggregation.

    Each output thought is tagged with ``_group`` (its pair index) so a
    following ``KeepBestPerGroup`` can rank within each pair independently.
    """

    def __init__(
        self,
        prompt_name: str = "aggregate",
        num_merges: int = 1,
        name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name or f"PairwiseAggregate({prompt_name},k={num_merges})")
        self.prompt_name = prompt_name
        self.num_merges = num_merges
        self.max_tokens = max_tokens
        self.stop = stop

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        if not inputs:
            return []

        # Pair up consecutive thoughts. An odd leftover is carried forward
        # unchanged so no data is silently dropped.
        pairs: List[List[Thought]] = [
            inputs[i : i + 2] for i in range(0, len(inputs), 2)
        ]

        prompts: List[str] = []
        # Carry the group index alongside each pair rather than looking it up
        # later: list.index() on Thought lists is both O(n) and fragile.
        prompt_pairs: List[tuple] = []
        produced: List[Thought] = []

        for group, pair in enumerate(pairs):
            if len(pair) == 1:
                # Nothing to merge with: pass through, retagged for the next
                # ranking step. Costs no model call.
                st = dict(pair[0].state)
                st["_group"] = group
                produced.append(self._new_thought(st, "carry", pair))
                continue

            prompt = prompter.build(
                self.prompt_name, [t.state for t in pair], **kwargs
            )
            if prompt is None:
                for st in parser.aggregate_local(
                    self.prompt_name, [t.state for t in pair], **kwargs
                ):
                    st = dict(st)
                    st["_group"] = group
                    produced.append(self._new_thought(st, self.prompt_name, pair))
                continue

            prompts.append(prompt)
            prompt_pairs.append((group, pair))

        if prompts:
            batched = lm.query_batch(
                prompts,
                num_responses=self.num_merges,
                max_tokens=self.max_tokens,
                stop=self.stop,
            )
            for (group, pair), responses in zip(prompt_pairs, batched):
                for raw in responses:
                    st = parser.parse(
                        self.prompt_name, [t.state for t in pair], raw, **kwargs
                    )
                    st = dict(st)
                    st["_group"] = group
                    produced.append(self._new_thought(st, self.prompt_name, pair))

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

    Each refinement *round* is one batched call across all thoughts being
    refined, so refining eight thoughts for three rounds costs three model
    round trips, not twenty-four.

    Parameters
    ----------
    rounds:
        How many successive refinement passes to apply.
    """

    def __init__(
        self,
        prompt_name: str = "improve",
        rounds: int = 1,
        name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name or f"Improve({prompt_name},r={rounds})")
        self.prompt_name = prompt_name
        self.rounds = rounds
        self.max_tokens = max_tokens
        self.stop = stop

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        current = self.get_input_thoughts()
        if not current:
            return []

        # Refine every thought in lockstep, one batched call per round.
        for _ in range(self.rounds):
            prompts: List[str] = []
            targets: List[Thought] = []
            for node in current:
                prompt = prompter.build(self.prompt_name, [node.state], **kwargs)
                if prompt is None:
                    continue
                prompts.append(prompt)
                targets.append(node)

            if not prompts:
                break

            batched = lm.query_batch(
                prompts,
                num_responses=1,
                max_tokens=self.max_tokens,
                stop=self.stop,
            )

            refreshed: List[Thought] = []
            replaced = {id(t) for t in targets}
            for node, responses in zip(targets, batched):
                if not responses:
                    refreshed.append(node)
                    continue
                st = parser.parse(
                    self.prompt_name, [node.state], responses[0], **kwargs
                )
                refreshed.append(self._new_thought(st, self.prompt_name, [node]))

            # Carry through any thought that had no prompt this round.
            current = refreshed + [t for t in current if id(t) not in replaced]

        return current


# ======================================================================
# Scoring and ranking (Section 3.3)
# ======================================================================
class Score(Operation):
    """Assign a score to every incoming thought.

    Paper: "A score is modeled as a general function E(v, G, p_theta)"
    (Section 3.3). Two scoring modes exist, matching Section 4.3:

      * **local** -- a deterministic Python function. Sorting and set
        intersection use this ("use cases such as sorting use simple local
        scoring functions"). **Free and exact** -- and on HPC, dramatically
        cheaper than asking the model. Prefer it wherever the task permits.
      * **LLM-based** -- ask the model to rate the thought. Document merging
        needs this, since redundancy and information retention have no
        closed-form measure. All thoughts are scored in one batched call.

    Parameters
    ----------
    scoring_fn:
        Callable ``state -> float`` for local scoring. If None, the LLM is
        queried via ``prompt_name`` instead.
    n_votes:
        For LLM scoring, how many times to ask and average. The paper queries
        "3 times for each value, and take the average" for document merging,
        because single LLM judgements are noisy. Each extra vote costs a full
        generation, so raise it only where the noise actually matters.
    max_tokens:
        Defaults to 8 -- a score is a couple of digits, and letting the model
        run to a 1024-token default here is pure waste.
    """

    def __init__(
        self,
        scoring_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        prompt_name: str = "score",
        n_votes: int = 1,
        name: Optional[str] = None,
        max_tokens: Optional[int] = 8,
        stop: Optional[Sequence[str]] = ("\n",),
    ) -> None:
        super().__init__(name or "Score")
        self.scoring_fn = scoring_fn
        self.prompt_name = prompt_name
        self.n_votes = n_votes
        self.max_tokens = max_tokens
        self.stop = stop

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self.get_input_thoughts()
        if not inputs:
            return []

        if self.scoring_fn is not None:
            # Deterministic local scoring -- no model call, no cost.
            for t in inputs:
                t.score = float(self.scoring_fn(t.state))
                t.scored = True
            return inputs

        # LLM scoring: one batched call covering every thought.
        prompts = [
            prompter.build(self.prompt_name, [t.state], **kwargs) for t in inputs
        ]
        batched = lm.query_batch(
            prompts,
            num_responses=self.n_votes,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )

        for t, raws in zip(inputs, batched):
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

    Degrading instead of truncating
    -------------------------------
    An earlier version filtered to ``t.valid`` and returned ``[]`` when none
    survived. Against the mock backend that never fired -- mock output almost
    always parses -- but against a real 7B model it silently destroyed whole
    runs: one unparseable batch emptied this operation, every downstream
    operation then had no input thoughts, and the graph quietly stopped. The
    summary table showed ToT at 1.1 LLM calls instead of 3 with no error
    anywhere, which is the worst kind of bug: a wrong number that looks like
    a result.

    Ranking now prefers valid thoughts but falls back to the best invalid one
    rather than returning nothing. A badly-scored bad answer is far more
    informative than a silently truncated graph, and it keeps volume and
    latency measurable.
    """

    @staticmethod
    def _rankable(thoughts: List[Thought]) -> List[Thought]:
        """Valid thoughts if any exist, otherwise everything (see class doc)."""
        valid = [t for t in thoughts if t.valid]
        if valid:
            return valid
        if thoughts:
            logging.getLogger(__name__).warning(
                "no valid thoughts to rank; falling back to %d invalid one(s) "
                "so the graph keeps its shape", len(thoughts)
            )
        return list(thoughts)

    def __init__(self, n: int = 1, name: Optional[str] = None) -> None:
        super().__init__(name or f"KeepBest(n={n})")
        self.n = n

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        inputs = self._rankable(self.get_input_thoughts())
        if not inputs:
            return []

        ranked = sorted(inputs, key=lambda t: t.score, reverse=True)
        kept = []
        for src in ranked[: self.n]:
            c = src.copy(operation="keepbest")
            c.add_predecessor(src)
            kept.append(c)
        return kept


class KeepBestPerGroup(Operation):
    """Ranking within groups: keep the best ``n`` thoughts per group key.

    Why this exists
    ---------------
    A plain ``KeepBest`` over a mixed population would keep the globally best
    thoughts and discard entire chunks. To sort four chunks we need the best
    candidate *for each chunk*.

    The naive alternative is four separate sort/score/keep chains, one per
    chunk. That works, but it forces four serial single-prompt model calls
    where one batched call would do. Grouping lets a single ``Generate`` hold
    all four chunks -- batching them -- while this operation still ranks
    within each chunk independently.

    Parameters
    ----------
    group_key:
        State key identifying the group (e.g. ``"chunk_index"`` or
        ``"_group"``). Thoughts lacking the key fall into a shared group.
    n:
        How many to keep per group.
    """

    def __init__(
        self,
        group_key: str = "_group",
        n: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"KeepBestPerGroup({group_key},n={n})")
        self.group_key = group_key
        self.n = n

    def _execute(self, lm, prompter, parser, **kwargs) -> List[Thought]:
        # Same fallback as KeepBest: never empty the graph over a bad batch.
        # Here it matters even more -- emptying one chunk's group would leave
        # PairwiseAggregate with an odd number of inputs at the next level,
        # silently changing the merge tree's shape.
        inputs = KeepBest._rankable(self.get_input_thoughts())
        if not inputs:
            return []

        groups: Dict[Any, List[Thought]] = {}
        for t in inputs:
            groups.setdefault(t.state.get(self.group_key), []).append(t)

        kept: List[Thought] = []
        # Sort group keys for deterministic output ordering -- the pairing in
        # PairwiseAggregate depends on a stable order.
        for key in sorted(groups, key=lambda k: (k is None, k)):
            ranked = sorted(groups[key], key=lambda t: t.score, reverse=True)
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

    Useful for routing, but note the cost implication: splitting a population
    into per-branch Selectors forces those branches to execute as separate
    operations, which prevents batching. Prefer ``KeepBestPerGroup`` when the
    goal is per-group ranking rather than genuine routing.
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
