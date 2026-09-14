"""
Real open-source LLM backends. No API keys anywhere.
====================================================

Three concrete backends, matched to three hardware situations:

+---------------+------------------------+-----------------------------------+
| Backend       | Where it runs          | Typical model                     |
+===============+========================+===================================+
| LlamaCppLM    | This laptop, CPU only  | Qwen2.5-1.5B-Instruct Q4_K_M      |
|               | (3.5 GB RAM budget)    | (~1 GB on disk)                   |
+---------------+------------------------+-----------------------------------+
| HFLM          | HPC, 1 GPU             | Llama-3.1-8B-Instruct (fp16/4bit) |
+---------------+------------------------+-----------------------------------+
| VLLMLM        | HPC, 1-8 GPUs          | Llama-3.1-70B-Instruct, Qwen2.5   |
|               | highest throughput     |                                   |
+---------------+------------------------+-----------------------------------+

All three import their heavy dependency *lazily*, inside ``__init__``. This is
deliberate: ``import got`` must stay fast and must not fail on a machine where
vLLM is not installed (like this laptop). You only pay for what you use.

Chat templating
---------------
Instruction-tuned open models are trained with a specific chat template
(``<|im_start|>`` for Qwen, ``<|begin_of_text|>`` for Llama 3, etc). Feeding a
raw prompt without that scaffolding measurably degrades output quality and is
a classic source of "my replication got bad numbers" bugs. We therefore apply
the tokenizer's own ``apply_chat_template`` wherever one is available, rather
than hand-rolling the format.

HPC cost notes
--------------
Two settings in ``VLLMLM`` matter far more than anything else for cluster bills:

* **Batching.** ``_generate_batch`` hands vLLM the whole prompt list in one
  ``generate()`` call so its continuous batcher can keep the GPU saturated.
  Feeding prompts one at a time can leave a large GPU 90%+ idle.
* **Prefix caching.** Every prompt a task issues shares a long identical
  prefix (the instructions and the few-shot example). With
  ``enable_prefix_caching=True`` vLLM computes that prefix's KV cache once and
  reuses it, so prefill cost collapses to the few tokens that actually differ.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .base import AbstractLanguageModel

# A single shared system prompt. Kept short and neutral: GoT relies on the
# task prompts themselves carrying the instructions, and a verbose system
# prompt would add token cost to every single call. It is also identical
# across calls, which makes it free under prefix caching.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Follow the user's output format exactly. "
    "Do not add explanations unless explicitly asked."
)


class LlamaCppLM(AbstractLanguageModel):
    """Run a quantised GGUF model on CPU via ``llama-cpp-python``.

    This is the backend for *local* validation on a machine with no GPU.
    Quantised 1B-3B models fit comfortably in a few GB of RAM.

    There is no real batching here: llama.cpp on CPU processes sequences one
    at a time, so ``_generate_batch`` inherits the base class loop. That is
    fine -- this backend exists for correctness checking, not throughput.

    Parameters
    ----------
    model_path:
        Path to a ``.gguf`` file. See ``scripts/download_local_model.sh``.
    n_ctx:
        Context window. GoT prompts are small by design (that is the point of
        decomposition), so 4096 is ample and keeps the KV cache small -- an
        important consideration with only 3.5 GB of RAM.
    n_threads:
        CPU threads. Defaults to the machine's core count.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "llamacpp",
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)

        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "llama-cpp-python is required for LlamaCppLM. "
                "Install with: pip install llama-cpp-python"
            ) from exc

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model not found at {model_path!r}. "
                "Run scripts/download_local_model.sh to fetch one."
            )

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count(),
            n_gpu_layers=n_gpu_layers,  # 0 = pure CPU
            verbose=verbose,
        )

    def _count_tokens(self, text: str) -> int:
        """Exact token count using the model's own tokenizer."""
        try:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return super()._count_tokens(text)

    def _generate(
        self,
        prompt: str,
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Draw ``num_responses`` completions.

        llama.cpp has no native n>1 sampling, so we loop. Each iteration is a
        fresh sample; with temperature > 0 they differ, which is what
        Generate(k) requires.
        """
        out: List[str] = []
        for _ in range(num_responses):
            result = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stop=list(stop) if stop else None,
            )
            out.append(result["choices"][0]["message"]["content"].strip())
        return out


class HFLM(AbstractLanguageModel):
    """Run an open-weights model through HuggingFace ``transformers``.

    Use on the HPC when vLLM is unavailable or the model is small enough that
    raw ``transformers`` throughput suffices. Supports 4-bit loading through
    bitsandbytes so an 8B model fits on a single 16 GB card.

    ``_generate_batch`` does real left-padded batching, which is a large win
    over the base-class loop -- though still well short of vLLM's continuous
    batching. Prefer ``VLLMLM`` for long runs.
    """

    def __init__(
        self,
        model_id: str,
        model_name: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name or model_id, **kwargs)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers and torch are required for HFLM") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Batched generation needs a pad token and LEFT padding: with right
        # padding the model would continue from pad tokens rather than from
        # the real prompt end, silently corrupting every short sequence in
        # the batch.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {"device_map": device}
        if dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        if load_in_4bit:
            # 4-bit NF4 quantisation: the standard way to fit a large model on
            # a single GPU with minimal quality loss.
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._model.eval()

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _build_prompt(self, prompt: str) -> str:
        """Apply the model's chat template if it defines one."""
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _generate(
        self,
        prompt: str,
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        return self._generate_batch([prompt], num_responses, max_tokens, stop)[0]

    def _generate_batch(
        self,
        prompts: Sequence[str],
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """Generate for all prompts in a single padded batch."""
        texts = [self._build_prompt(p) for p in prompts]
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self._model.device)

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_tokens,
                temperature=max(self.temperature, 1e-5),
                do_sample=self.temperature > 0,
                num_return_sequences=num_responses,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Left padding means every row's generated part starts at the same
        # offset, so one slice works for the whole batch.
        prompt_len = inputs["input_ids"].shape[-1]
        decoded = [
            self._tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
            for o in outputs
        ]

        # `generate` returns num_return_sequences rows per prompt, in order.
        grouped: List[List[str]] = []
        for i in range(len(prompts)):
            chunk = decoded[i * num_responses : (i + 1) * num_responses]
            grouped.append([self._apply_stop(c, stop) for c in chunk])
        return grouped

    @staticmethod
    def _apply_stop(text: str, stop: Optional[Sequence[str]]) -> str:
        """Truncate at the first stop string.

        ``transformers`` has no simple batched stop-string support, so we cut
        in post-processing. This does not save GPU time (the tokens were
        already generated) but it does keep parsing robust. Use vLLM when
        stop-driven early exit matters for cost.
        """
        if not stop:
            return text
        cut = len(text)
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut].strip()


class VLLMLM(AbstractLanguageModel):
    """Run an open-weights model through vLLM -- the HPC workhorse.

    vLLM's paged attention and continuous batching make it dramatically
    faster than plain ``transformers`` for the many-small-calls pattern GoT
    produces. It also samples n>1 natively in a single batched call, which
    maps perfectly onto Generate(k).

    Cost-relevant settings
    ----------------------
    enable_prefix_caching:
        On by default here. Every GoT prompt for a task shares a long
        identical prefix (instructions + few-shot example); caching its KV
        state turns prefill for that prefix into a lookup. This is close to
        free and is the largest single saving available on prefill.
    tensor_parallel_size:
        Should equal the number of GPUs in the SLURM allocation for models
        too large for one card.
    max_num_seqs:
        Upper bound on concurrent sequences. Raising it increases GPU
        utilisation for GoT's many-small-calls pattern; lower it if you hit
        out-of-memory during long runs.
    """

    #: vLLM computes the prompt once and samples n continuations from it, so
    #: prompt tokens must be charged once, not n times.
    shares_prompt_across_samples = True

    def __init__(
        self,
        model_id: str,
        model_name: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        dtype: str = "auto",
        enable_prefix_caching: bool = True,
        max_num_seqs: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name or model_id, **kwargs)

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "vLLM is required for VLLMLM. On the HPC: pip install vllm"
            ) from exc

        self._SamplingParams = SamplingParams

        llm_kwargs: Dict[str, Any] = {
            "model": model_id,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "max_num_seqs": max_num_seqs,
        }
        # Older vLLM builds do not accept this kwarg; fall back rather than
        # crashing a queued HPC job over a version difference.
        try:
            self._llm = LLM(enable_prefix_caching=enable_prefix_caching, **llm_kwargs)
        except TypeError:
            self.logger.warning(
                "this vLLM build does not support enable_prefix_caching; "
                "continuing without it (prefill will cost more)"
            )
            self._llm = LLM(**llm_kwargs)

        self._tokenizer = self._llm.get_tokenizer()

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _build_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _generate(
        self,
        prompt: str,
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        return self._generate_batch([prompt], num_responses, max_tokens, stop)[0]

    def _generate_batch(
        self,
        prompts: Sequence[str],
        num_responses: int,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """One vLLM call for the entire prompt list.

        This is the method that keeps the GPU busy: vLLM schedules all
        ``len(prompts) * num_responses`` sequences together through its
        continuous batcher.
        """
        params = self._SamplingParams(
            n=num_responses,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stop=list(stop) if stop else None,
        )
        outputs = self._llm.generate(
            [self._build_prompt(p) for p in prompts], params
        )
        # vLLM preserves input order in its output list.
        return [[c.text.strip() for c in out.outputs] for out in outputs]
