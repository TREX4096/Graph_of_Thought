"""
Abstract LLM backend interface.
================================

Every "thought" in Graph of Thoughts (Besta et al., 2024) is produced by a call
to a language model.  The GoT framework itself is *model agnostic*: the
Controller, the Graph of Operations and the scoring logic never care which
model is behind the call.  We therefore funnel every model interaction through
the single narrow interface defined here.

Why this matters for this project
---------------------------------
We must support three very different execution environments:

  1. ``MockLM``      -- no model at all. Deterministic, instant, free.
                        Used to unit-test the *graph machinery* itself.
  2. ``LlamaCppLM``  -- a small quantised GGUF model on a laptop CPU.
                        Used for real-but-small local validation.
  3. ``VLLMLM`` / ``HFLM`` -- a full open-weights model on HPC GPUs.
                        Used for the actual paper-replication experiments.

Because all three implement ``AbstractLanguageModel``, a task written once
runs unchanged in all three settings.  Only a config flag changes.

Batching -- the single biggest HPC cost lever
---------------------------------------------
GPU time is the whole budget on a cluster. A GoT run issues *many small*
model calls, and issuing them one at a time leaves the GPU almost idle:
vLLM's continuous batching can hold dozens of sequences in flight, but only
if you hand it dozens of prompts at once.

So the primary API here is :meth:`query_batch`, which takes a *list* of
prompts. Backends that can genuinely batch (vLLM) override
:meth:`_generate_batch` and service the whole list in one GPU call. Backends
that cannot (llama.cpp on CPU) fall back to a loop, which is correct and no
slower than before.

Callers should prefer ``query_batch`` over ``query`` wherever more than one
prompt is known at the same time. The operations in ``got/operations.py`` all
do this.

Per-call generation limits
--------------------------
Decode time is roughly linear in tokens generated, so a global
``max_tokens=1024`` means a call that needs 40 tokens can still cost 1024 if
the model fails to emit a stop token. Both :meth:`query` and
:meth:`query_batch` accept per-call ``max_tokens`` and ``stop`` overrides, and
operations set them from what the step actually needs. On real HPC runs this
is frequently a 5-20x saving on decode.

Cost accounting
---------------
The GoT paper measures cost in tokens (Section 7). To reproduce those figures
we count tokens for *every* call, so token accounting lives in the base class
rather than in each backend. Open-source local models are monetarily free, but
token count is the hardware-independent proxy for cost that the paper plots.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class UsageStats:
    """Running tally of what a backend has consumed.

    Attributes
    ----------
    prompt_tokens:
        Total tokens sent *to* the model across all calls.
    completion_tokens:
        Total tokens generated *by* the model across all calls.
    n_calls:
        Number of prompts serviced. Counting prompts (not batches) keeps the
        figure comparable between a batched HPC run and a serial local one.
    n_batches:
        Number of distinct backend invocations. ``n_calls / n_batches`` is the
        mean batch size -- the number to watch when tuning HPC throughput.
    cost:
        Monetary cost. Stays 0.0 for local open-source models; kept so that
        the same reporting code works if an API backend is ever added.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0
    n_batches: int = 0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def mean_batch_size(self) -> float:
        return self.n_calls / self.n_batches if self.n_batches else 0.0

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        n_prompts: int = 1,
        cost: float = 0.0,
    ) -> None:
        """Record one completed backend invocation covering ``n_prompts``."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.n_calls += n_prompts
        self.n_batches += 1
        self.cost += cost

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["mean_batch_size"] = round(self.mean_batch_size, 2)
        return d


class AbstractLanguageModel(abc.ABC):
    """Base class every GoT backend must implement.

    Subclasses implement :meth:`_generate` (one prompt) and may optionally
    override :meth:`_generate_batch` (many prompts in one go) when the runtime
    supports true batching. Everything else -- token accounting, logging,
    caching -- is handled here so behaviour is identical across backends.
    """

    #: Set True by backends whose n>1 sampling shares the prompt's forward
    #: pass (vLLM). Used only for accurate prompt-token accounting.
    shares_prompt_across_samples: bool = False

    def __init__(
        self,
        model_name: str = "unnamed",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        cache: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = UsageStats()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # An optional exact-match response cache. GoT issues many *identical*
        # prompts (e.g. scoring the same thought repeatedly), so caching can
        # cut cost substantially. It is OFF by default because caching changes
        # sampling behaviour: with temperature > 0 we genuinely want k
        # *different* samples from one prompt, and a cache would collapse them
        # into one. Operations that want deterministic single answers (scoring
        # at temperature 0) can safely enable it.
        self._cache_enabled = cache
        self._cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _generate(
        self,
        prompt: str,
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Raw single-prompt model call. Return ``num_responses`` completions.

        Implementations must NOT do token accounting; :meth:`query` does it.
        """
        raise NotImplementedError

    def _generate_batch(
        self,
        prompts: Sequence[str],
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """Raw many-prompt call. Default: loop over :meth:`_generate`.

        Override this wherever the runtime can service many prompts in one
        invocation -- that is where the HPC speedup comes from.
        """
        return [
            self._generate(p, num_responses, max_tokens=max_tokens, stop=stop)
            for p in prompts
        ]

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for ``text``.

        The default is a deliberately crude heuristic (~4 characters per
        token) so the base class never depends on a tokenizer being
        available. Backends that own a real tokenizer override this.
        """
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Accounting helper
    # ------------------------------------------------------------------
    def _account(
        self, prompts: Sequence[str], results: Sequence[Sequence[str]]
    ) -> None:
        """Record token usage for a completed invocation.

        Prompt tokens are charged once per *sample* for backends that re-run
        the prompt per sample, but only once per *prompt* for backends whose
        n>1 sampling shares one forward pass over the prompt (vLLM). Getting
        this right matters because the paper's cost figures are token counts,
        and over-charging the prompt by a factor of k would misreport GoT's
        cost relative to single-sample baselines.
        """
        p_tok = 0
        c_tok = 0
        for prompt, responses in zip(prompts, results):
            n = max(1, len(responses))
            per_prompt = self._count_tokens(prompt)
            p_tok += per_prompt if self.shares_prompt_across_samples else per_prompt * n
            c_tok += sum(self._count_tokens(r) for r in responses)

        self.usage.add(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            n_prompts=len(prompts),
        )

    # ------------------------------------------------------------------
    # Public API used by the rest of GoT
    # ------------------------------------------------------------------
    def query(
        self,
        prompt: str,
        num_responses: int = 1,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Send one prompt and return ``num_responses`` samples.

        Prefer :meth:`query_batch` when several prompts are known at once --
        on a GPU backend that is the difference between a busy device and an
        idle one.
        """
        return self.query_batch(
            [prompt], num_responses=num_responses, max_tokens=max_tokens, stop=stop
        )[0]

    def query_batch(
        self,
        prompts: Sequence[str],
        num_responses: int = 1,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """Send many prompts at once; return one completion list per prompt.

        Parameters
        ----------
        prompts:
            Fully-rendered prompt texts from a task's Prompter.
        num_responses:
            How many independent completions per prompt -- the ``k`` of the
            paper's Generate transformation.
        max_tokens:
            Per-call cap on generated tokens. Defaults to the backend's
            ``max_tokens``. Setting this from what the step actually needs is
            the cheapest large saving available on HPC.
        stop:
            Stop strings. Ends decoding as soon as the answer is complete
            instead of running to the token cap.

        Returns
        -------
        ``[[completion, ...], ...]``, aligned with ``prompts``.
        """
        prompts = list(prompts)
        if not prompts:
            return []

        # --- Serve what we can from cache -----------------------------
        results: List[Optional[List[str]]] = [None] * len(prompts)
        pending_idx: List[int] = []
        pending: List[str] = []

        for i, p in enumerate(prompts):
            if self._cache_enabled:
                hit = self._cache.get(p)
                if hit is not None and len(hit) >= num_responses:
                    results[i] = hit[:num_responses]
                    continue
            pending_idx.append(i)
            pending.append(p)

        # --- Collapse duplicate prompts within this batch --------------
        # GoT frequently produces identical prompts in one operation (e.g.
        # two chunks that happen to be equal, or repeated scoring of the same
        # thought). Sending a duplicate costs full GPU time for a result we
        # already have. Deduplicating is only safe when we are not relying on
        # sampling variety, i.e. when a single response is requested.
        dedup = num_responses == 1 and len(set(pending)) < len(pending)
        if dedup:
            unique: List[str] = []
            first_pos: Dict[str, int] = {}
            mapping: List[int] = []
            for p in pending:
                if p not in first_pos:
                    first_pos[p] = len(unique)
                    unique.append(p)
                mapping.append(first_pos[p])
            to_send = unique
        else:
            mapping = list(range(len(pending)))
            to_send = pending

        if to_send:
            self.logger.debug(
                "query_batch model=%s prompts=%d (unique=%d) n=%d max_tokens=%s",
                self.model_name, len(pending), len(to_send), num_responses,
                max_tokens or self.max_tokens,
            )

            raw = self._generate_batch(
                to_send,
                num_responses,
                max_tokens=max_tokens or self.max_tokens,
                stop=stop,
            )
            self._account(to_send, raw)

            for slot, src in zip(pending_idx, mapping):
                results[slot] = list(raw[src])

            if self._cache_enabled:
                for p, r in zip(to_send, raw):
                    self._cache[p] = list(r)

        return [r if r is not None else [] for r in results]

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def reset_usage(self) -> None:
        """Zero the counters (used between benchmark instances)."""
        self.usage = UsageStats()

    def usage_report(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of consumption so far."""
        return {"model_name": self.model_name, **self.usage.to_dict()}

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"<{self.__class__.__name__} model={self.model_name!r} "
            f"calls={self.usage.n_calls} batches={self.usage.n_batches}>"
        )
