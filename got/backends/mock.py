"""
MockLM -- a deterministic, zero-cost fake LLM.
==============================================

Why a mock backend is essential here
------------------------------------
This machine has 3.5 GB of RAM available to WSL and no CUDA GPU. Running even
a 7B model locally is impossible. But *the thing we are actually replicating*
is not a language model -- it is the Graph of Thoughts control structure:
the Graph of Operations, thought aggregation, scoring, ranking, and the
Controller's execution order.

That machinery can be tested completely without a real model, provided the
fake model behaves like a real one in the ways that matter:

  * It is **fallible**. A perfect oracle would make GoT pointless -- the whole
    premise of the paper is that LLMs make errors on long inputs which
    decomposition + aggregation can repair. ``error_rate`` controls this.
  * It is **stochastic across samples**. Generate(k) must yield k *different*
    candidates, otherwise KeepBest has nothing to choose between.
  * It **degrades with input length**, mirroring the paper's core
    observation that LLMs fail on long sequences ("LLMs are unable to sort a
    sequence of such numbers correctly beyond a certain length consistently").

Reading the prompt correctly
----------------------------
Our task prompts are few-shot: they contain a worked *example* before the real
input. A naive "grab every bracketed list" parser would merge the example's
numbers into the answer, producing outputs far longer than the input. (This
was a real bug caught by the first smoke test -- a 32-element input came back
with 205 elements.)

The prompts are all built so the real payload comes **last**, so we extract
the last N lists, where N depends on the operation. ``_payload_lists`` does
this and is the single place that knows the convention.

Determinism
-----------
Seeded via ``random.Random(seed)``, so a given seed reproduces a run exactly.
This makes regression tests possible -- a real LLM cannot give you that.

Limitations (be honest about these in the report)
-------------------------------------------------
MockLM does *not* validate prompt wording, few-shot example quality, or any
genuinely linguistic behaviour. It validates control flow and bookkeeping
only. Quality numbers from MockLM are NOT paper-comparable; only runs against
a real open-weights model are.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence

from .base import AbstractLanguageModel


class MockLM(AbstractLanguageModel):
    """A fake LLM that solves the benchmark tasks with tunable imperfection.

    Parameters
    ----------
    error_rate:
        Base probability of corrupting the produced answer. 0.0 gives a
        perfect oracle (useful to confirm a pipeline can reach a perfect
        score at all); 0.15 is a reasonable default that leaves clear room
        for GoT's aggregation to demonstrate improvement.
    length_sensitivity:
        How much longer inputs increase the effective error rate. Set to 0
        to disable. With the default, a 64-element list is noticeably harder
        than a 16-element one, reproducing the paper's motivating failure.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        model_name: str = "mock",
        error_rate: float = 0.15,
        length_sensitivity: float = 0.004,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.error_rate = error_rate
        self.length_sensitivity = length_sensitivity
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Prompt reading
    # ------------------------------------------------------------------
    @staticmethod
    def _payload_lists(prompt: str, n: int) -> List[List[int]]:
        """Return the last ``n`` bracketed integer lists in ``prompt``.

        Our prompt templates always place the few-shot example first and the
        real input last, so the trailing lists are the actual payload. Taking
        them from the end is what keeps example numbers out of the answer.
        """
        matches = re.findall(r"\[[\s\d,\-]*\]", prompt)
        if not matches:
            return []
        tail = matches[-n:] if n > 0 else matches
        return [[int(x) for x in re.findall(r"-?\d+", m)] for m in tail]

    def _effective_error_rate(self, n_items: int) -> float:
        """Error rate for an input of ``n_items`` elements, capped at 0.9."""
        rate = self.error_rate + self.length_sensitivity * n_items
        return min(rate, 0.9)

    def _corrupt(self, values: List[int], rate: float) -> List[int]:
        """Apply realistic LLM-style errors to a correct answer.

        The three corruption modes mirror the exact failure modes the GoT
        paper's sorting score penalises (Section 5.1):
          * dropping an element    -> breaks frequency preservation (term Y)
          * duplicating an element -> also breaks term Y
          * swapping neighbours    -> breaks sortedness (term X)

        We draw the number of errors from the sequence length so that error
        *count* grows with length, matching real model behaviour, rather than
        corrupting every position independently.
        """
        out = list(values)
        if not out:
            return out

        n_errors = 0
        for _ in range(len(out)):
            if self._rng.random() < rate:
                n_errors += 1
        # Cap damage so a corrupted answer stays recognisably related to the
        # truth -- real models degrade, they do not emit noise.
        n_errors = min(n_errors, max(1, len(out) // 2)) if n_errors else 0

        for _ in range(n_errors):
            if not out:
                break
            mode = self._rng.choice(("drop", "dup", "swap"))
            idx = self._rng.randrange(len(out))
            if mode == "drop":
                out.pop(idx)
            elif mode == "dup":
                out.insert(idx, out[idx])
            elif mode == "swap" and len(out) > 1:
                j = min(idx + 1, len(out) - 1)
                out[idx], out[j] = out[j], out[idx]
        return out

    # ------------------------------------------------------------------
    # Task-specific simulated behaviour
    # ------------------------------------------------------------------
    def _answer_keyword_count(self, prompt: str) -> str:
        """Simulate counting country mentions in a passage."""
        countries = [
            "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany",
            "Italy", "Spain", "Norway", "Sweden", "Japan", "China", "India",
            "Australia", "Egypt", "Kenya", "Peru", "Chile", "Poland", "Greece",
        ]
        # Only count within the real passage, not the few-shot example.
        body = prompt.rsplit("Input:", 1)[-1]
        counts: Dict[str, int] = {}
        for c in countries:
            n = len(re.findall(rf"\b{c}\b", body))
            if n:
                counts[c] = n

        rate = self._effective_error_rate(sum(counts.values()) * 4)
        for c in list(counts):
            if self._rng.random() < rate:
                counts[c] = max(0, counts[c] + self._rng.choice((-1, 1)))
                if counts[c] == 0:
                    del counts[c]

        inner = ", ".join(f'"{k}": {v}' for k, v in sorted(counts.items()))
        return "{" + inner + "}"

    # ------------------------------------------------------------------
    # AbstractLanguageModel implementation
    # ------------------------------------------------------------------
    def _generate(
        self,
        prompt: str,
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Dispatch on prompt content and return ``num_responses`` samples.

        ``max_tokens`` and ``stop`` are accepted for interface compatibility
        and ignored: the mock emits exactly one short line, so it can never
        exceed a token budget or run past a stop string.

        Each sample is generated independently, so the k candidates of a
        Generate(k) operation genuinely differ -- which is what gives
        KeepBest and Aggregate something to work with.

        Dispatch is by distinctive phrases from our own prompt templates, so
        each branch knows exactly how many payload lists to expect.
        """
        low = prompt.lower()
        out: List[str] = []

        for _ in range(num_responses):
            # --- Scoring: the caller wants a bare number ------------------
            if "give a score" in low:
                out.append(str(self._rng.randint(0, 10)))
                continue

            # --- Keyword counting ----------------------------------------
            if "keyword" in low or "countries" in low or "country" in low:
                out.append(self._answer_keyword_count(prompt))
                continue

            # --- Set union: two payload lists ----------------------------
            # Must be tested BEFORE the generic branches. The union prompt
            # mentions neither "intersect" nor "merge", so without this it
            # fell through to the single-list default and silently dropped
            # one of its two inputs at every aggregation -- which made GoT
            # look worse than the IO baseline on set intersection.
            if "appears in either list" in low:
                lists = self._payload_lists(prompt, 2)
                if len(lists) < 2:
                    out.append(str(sorted(set(lists[0])) if lists else []))
                    continue
                correct = sorted(set(lists[0]) | set(lists[1]))
                rate = self._effective_error_rate(len(correct)) * 0.5
                out.append(str(self._corrupt(correct, rate)))
                continue

            # --- Set intersection: two payload lists ---------------------
            if "intersect" in low or "appears in both" in low:
                lists = self._payload_lists(prompt, 2)
                if len(lists) < 2:
                    out.append("[]")
                    continue
                correct = sorted(set(lists[0]) & set(lists[1]))
                rate = self._effective_error_rate(len(lists[0]) + len(lists[1]))
                out.append(str(self._corrupt(correct, rate)))
                continue

            # --- Refinement: "Input list" + "Incorrectly sorted list" -----
            # The first of the two payload lists is the original; the task is
            # to produce its correct sorting. Refinement is modelled as
            # easier than sorting from scratch (the model is shown its own
            # near-miss), so the error rate is halved.
            if "incorrectly sorted list" in low:
                lists = self._payload_lists(prompt, 2)
                if not lists:
                    out.append("[]")
                    continue
                original = lists[0]
                correct = sorted(original)
                rate = self._effective_error_rate(len(correct)) * 0.5
                out.append(str(self._corrupt(correct, rate)))
                continue

            # --- Merge two sorted lists: two payload lists ----------------
            if "merge the following two sorted lists" in low:
                lists = self._payload_lists(prompt, 2)
                if len(lists) < 2:
                    out.append(str(lists[0] if lists else []))
                    continue
                correct = sorted(lists[0] + lists[1])
                # Merging is easier than sorting from scratch: the inputs are
                # already ordered, so the model mostly interleaves them.
                rate = self._effective_error_rate(len(correct)) * 0.6
                out.append(str(self._corrupt(correct, rate)))
                continue

            # --- Document merging: no numeric structure -------------------
            if "merge" in low:
                out.append(
                    "MERGED DOCUMENT (mock): combined the supplied documents, "
                    "removing duplicated clauses and retaining unique terms."
                )
                continue

            # --- Default: sort a single list ------------------------------
            lists = self._payload_lists(prompt, 1)
            if not lists:
                out.append("[]")
                continue
            correct = sorted(lists[0])
            rate = self._effective_error_rate(len(correct))
            out.append(str(self._corrupt(correct, rate)))

        return out
