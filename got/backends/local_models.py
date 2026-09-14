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
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import AbstractLanguageModel

# A single shared system prompt. Kept short and neutral: GoT relies on the
# task prompts themselves carrying the instructions, and a verbose system
# prompt would only add token cost to every single call.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Follow the user's output format exactly. "
    "Do not add explanations unless explicitly asked."
)


class LlamaCppLM(AbstractLanguageModel):
    """Run a quantised GGUF model on CPU via ``llama-cpp-python``.

    This is the backend for *local* validation on a machine with no GPU.
    Quantised 1B-3B models fit comfortably in a few GB of RAM.

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

    def _generate(self, prompt: str, num_responses: int) -> List[str]:
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
                max_tokens=self.max_tokens,
            )
            out.append(result["choices"][0]["message"]["content"].strip())
        return out


class HFLM(AbstractLanguageModel):
    """Run an open-weights model through HuggingFace ``transformers``.

    Use on the HPC when vLLM is unavailable or the model is small enough that
    raw ``transformers`` throughput suffices. Supports 4-bit loading through
    bitsandbytes so an 8B model fits on a single 16 GB card.
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

    def _generate(self, prompt: str, num_responses: int) -> List[str]:
        """Batch-sample ``num_responses`` completions in one forward pass."""
        text = self._build_prompt(prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 1e-5),
                do_sample=self.temperature > 0,
                num_return_sequences=num_responses,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Strip the prompt tokens; keep only what the model newly generated.
        prompt_len = inputs["input_ids"].shape[-1]
        return [
            self._tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
            for o in outputs
        ]


class VLLMLM(AbstractLanguageModel):
    """Run an open-weights model through vLLM -- the HPC workhorse.

    vLLM's paged attention and continuous batching make it dramatically
    faster than plain ``transformers`` for the many-small-calls pattern GoT
    produces. It also samples n>1 natively in a single batched call, which
    maps perfectly onto Generate(k).

    ``tensor_parallel_size`` should equal the number of GPUs in the SLURM
    allocation for models too large for one card.
    """

    def __init__(
        self,
        model_id: str,
        model_name: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        dtype: str = "auto",
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
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
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

    def _generate(self, prompt: str, num_responses: int) -> List[str]:
        """Native n-sampling: one call returns all k candidates."""
        params = self._SamplingParams(
            n=num_responses,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        outputs = self._llm.generate([self._build_prompt(prompt)], params)
        return [c.text.strip() for c in outputs[0].outputs]
