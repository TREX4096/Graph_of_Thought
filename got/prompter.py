"""
Prompter and Parser base classes.
=================================

Paper reference
---------------
Section 4.1 (Prompter): "The Prompter prepares the prompts to be sent to the
LLM. This module is responsible for the specifics of encoding the graph
structure within the prompt."

Section 4.2 (Parser): "The Parser extracts information from LLM thoughts. For
each such thought, the Parser constructs the thought state, which contains
this extracted information."

Together these two classes are the *entire* task-specific surface of GoT.
Everything else -- operations, controller, scoring machinery -- is generic.
To add a new task you write one Prompter and one Parser; you do not touch the
framework.

The ``build(name, states, **kwargs)`` dispatch convention
---------------------------------------------------------
Rather than one method per prompt type (which forces the framework to know
their names), operations pass a *string* naming the prompt they want. The
Prompter dispatches on it. This is how ``Generate(prompt_name="sort")`` stays
generic while producing a sorting-specific prompt.

Returning ``None`` from ``build`` marks the step as *local*: work that needs
no model. Splitting a 64-element list into four chunks is deterministic
Python; spending an LLM call on it would add cost and a failure mode for no
benefit. The paper's own sorting figure shows the first Generate as a
structural split, so treating it locally is faithful to the design.
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any, Dict, List, Optional


class AbstractPrompter(abc.ABC):
    """Builds prompt strings for a specific task."""

    @abc.abstractmethod
    def build(
        self, name: str, states: List[Dict[str, Any]], **kwargs
    ) -> Optional[str]:
        """Return the prompt text for step ``name``.

        Parameters
        ----------
        name:
            Which prompt to build, e.g. ``"sort"``, ``"aggregate"``, ``"score"``.
        states:
            The state dicts of the input thoughts. Aggregation passes several;
            generation passes one.

        Returns
        -------
        The prompt string, or ``None`` if this step is handled locally
        without an LLM call.
        """
        raise NotImplementedError


class AbstractParser(abc.ABC):
    """Extracts thought state from raw LLM output for a specific task."""

    @abc.abstractmethod
    def parse(
        self, name: str, states: List[Dict[str, Any]], raw: str, **kwargs
    ) -> Dict[str, Any]:
        """Turn the model's raw text into a new thought state.

        Implementations should be defensive: small open-weights models
        frequently wrap answers in prose or markdown fences. A parse failure
        should produce a state marked invalid, never an exception -- one bad
        response must not abort a multi-hour HPC job.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks for local (non-LLM) steps
    # ------------------------------------------------------------------
    def parse_local(
        self, name: str, state: Dict[str, Any], **kwargs
    ) -> List[Dict[str, Any]]:
        """Produce new states without an LLM call (used when build() -> None)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} declared prompt {name!r} as local but "
            "did not implement parse_local()."
        )

    def aggregate_local(
        self, name: str, states: List[Dict[str, Any]], **kwargs
    ) -> List[Dict[str, Any]]:
        """Merge states without an LLM call."""
        raise NotImplementedError(
            f"{self.__class__.__name__} declared aggregation {name!r} as local "
            "but did not implement aggregate_local()."
        )

    # ------------------------------------------------------------------
    # Shared text-wrangling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def strip_code_fences(text: str) -> str:
        """Remove markdown code fences that instruct-tuned models love to add.

        ``` ```json\n[1,2,3]\n``` ``` -> ``[1,2,3]``
        """
        text = text.strip()
        fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            return fence.group(1).strip()
        return text

    @classmethod
    def extract_list(cls, text: str) -> Optional[List[int]]:
        """Find the last bracketed integer list in ``text``.

        We take the *last* match rather than the first because chatty models
        often restate the input before giving their answer; the answer is
        almost always the final list mentioned.
        """
        text = cls.strip_code_fences(text)
        matches = re.findall(r"\[[\s\d,\-]*\]", text)
        if matches:
            try:
                return [int(n) for n in re.findall(r"-?\d+", matches[-1])]
            except ValueError:
                return None

        # No closing bracket. The usual cause is the generation hitting
        # ``max_tokens`` mid-list, and discarding these outright is what made
        # a whole graph collapse: the thought became invalid, KeepBest dropped
        # it, and every downstream operation silently had nothing to consume.
        #
        # Salvaging the partial list is strictly better. It is a *wrong*
        # answer, but the error-scope scorer already penalises missing
        # elements, so it ranks below a complete one exactly as it should --
        # and the graph keeps its shape, so volume and latency stay
        # measurable. Losing the last number guards against a half-emitted
        # digit ("1" of an intended "12").
        tail = re.search(r"\[[\s\d,\-]*$", text)
        if tail:
            nums = re.findall(r"-?\d+", tail.group(0))
            if len(nums) > 1:
                try:
                    return [int(n) for n in nums[:-1]]
                except ValueError:
                    return None
        return None

    @classmethod
    def extract_json_object(cls, text: str) -> Optional[Dict[str, Any]]:
        """Find and parse the last JSON object in ``text``."""
        text = cls.strip_code_fences(text)
        matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
        for candidate in reversed(matches):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def parse_score(raw: str) -> Optional[float]:
        """Pull a numeric score out of an LLM scoring response."""
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        return float(match.group()) if match else None
