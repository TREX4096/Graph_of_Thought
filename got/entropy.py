"""
Step entropy: a computable criterion for which reasoning to discard.
====================================================================

Implements the measurement from Li et al., "Making Slow Thinking Faster:
Compressing LLM Chain-of-Thought via Step Entropy" (ICLR 2026,
arXiv:2508.03346), which reports that roughly 80% of low-entropy steps in
a long chain of thought can be removed with only minor accuracy loss,
while removing random or high-entropy steps hurts badly.

Why this module exists in a Graph-of-Thoughts project
-----------------------------------------------------
Our open question has been: *which vertices of a reasoning graph can be
dropped?* Marginal contribution answers it exactly but requires an
ablation run per vertex, which is far too expensive to use as an operator.
Step entropy is a candidate *cheap surrogate* - it falls out of the same
forward pass that generated the thought, at no extra cost.

The paper measures it on chains. Whether it transfers to graph vertices is
an empirical question, and answering it is the point of this module:
  * on a chain (GSM8K, MATH-500) we can reproduce the paper directly;
  * on a graph we can test entropy against measured marginal contribution.

Definitions
-----------
Token entropy over the model's next-token distribution given context c:

    H(t | c) = - sum_{w in V} p(w | c) log2 p(w | c)

Step entropy for step S_i of M_i tokens, length-normalised (the paper's
Eq. 3, which corrects for steps being different lengths):

    H(S_i) = (1 / M_i) * sum_j H(t_ij | c_ij)

A NOTE ON APPROXIMATION - read before quoting any number
--------------------------------------------------------
A serving engine does not hand back the full vocabulary distribution; vLLM
returns the top-k logprobs per position. We therefore compute entropy over
the top-k, renormalised to sum to one. This *under-estimates* true entropy,
and the error grows with how flat the real distribution is.

For our purpose that is acceptable, because pruning only needs steps
*ranked* by entropy, and the approximation is monotone in practice. It is
not acceptable to report these as absolute entropies in a paper. Raise
``logprobs`` if you need a tighter bound; k=20 captures the bulk of the
mass for a peaked distribution and much less for a flat one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: The paper segments the thinking content on blank lines.
STEP_DELIMITER = "\n\n"

#: Placeholder written in place of a pruned step. The paper finds this
#: degrades far less at high pruning ratios than deleting the step outright
#: or substituting [MASK] - the model is told that reasoning happened here
#: and was elided, rather than being shown a gap.
SKIP_TOKEN = "[SKIP]"


# ----------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_thinking(text: str) -> str:
    """Return the content between <think> tags, or the whole text if absent.

    Reasoning models emit their chain inside <think>...</think>. Instruct
    models such as Qwen2.5 do not, and simply write the chain followed by
    the answer - so falling back to the whole string keeps this usable for
    both, which matters because our GPU budget runs an instruct model.
    """
    m = THINK_BLOCK.search(text)
    return m.group(1).strip() if m else text.strip()


def segment_steps(text: str, delimiter: str = STEP_DELIMITER) -> List[str]:
    """Split a reasoning chain into steps on blank lines.

    The paper's own limitations section flags that results are sensitive to
    this choice, so it is a parameter rather than a constant. A model that
    writes one long paragraph yields a single step and nothing to prune -
    which is a real finding about that model, not a bug here.
    """
    parts = [p.strip() for p in text.split(delimiter)]
    return [p for p in parts if p]


# ----------------------------------------------------------------------
# Entropy
# ----------------------------------------------------------------------
def token_entropy(logprobs: Sequence[float]) -> float:
    """Entropy in bits over one position, from top-k logprobs (natural log).

    The top-k probabilities are renormalised before the sum, so this is the
    entropy of the truncated-and-rescaled distribution. See the module
    docstring on why that is a lower bound.
    """
    if not logprobs:
        return 0.0
    ps = [math.exp(lp) for lp in logprobs]
    total = sum(ps)
    if total <= 0:
        return 0.0
    ps = [p / total for p in ps]
    return -sum(p * math.log2(p) for p in ps if p > 0)


@dataclass
class StepMeasurement:
    """One reasoning step, with everything needed to decide its fate."""
    index: int
    text: str
    n_tokens: int
    entropy: float                       # length-normalised, bits/token
    total_entropy: float                 # summed over the step
    pruned: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def measure_steps(
    text: str,
    token_entropies: Sequence[float],
    token_texts: Optional[Sequence[str]] = None,
    delimiter: str = STEP_DELIMITER,
) -> List[StepMeasurement]:
    """Attribute per-token entropies to steps and normalise by step length.

    ``token_entropies`` is per generated token, in order. We align tokens to
    steps by walking the decoded token texts and accumulating characters
    until each step's length is covered. When token texts are unavailable we
    fall back to apportioning tokens by character share, which is coarse but
    adequate for ranking.
    """
    steps = segment_steps(text, delimiter)
    if not steps:
        return []

    if token_texts is not None and len(token_texts) == len(token_entropies):
        return _measure_aligned(steps, token_entropies, token_texts, delimiter)

    # Fallback: split the entropy sequence proportionally to step length.
    total_chars = sum(len(s) for s in steps) or 1
    out, cursor = [], 0
    for i, s in enumerate(steps):
        share = max(1, round(len(token_entropies) * len(s) / total_chars))
        chunk = list(token_entropies[cursor: cursor + share]) or [0.0]
        cursor += share
        out.append(StepMeasurement(
            index=i, text=s, n_tokens=len(chunk),
            entropy=sum(chunk) / len(chunk), total_entropy=sum(chunk),
        ))
    return out


def _measure_aligned(steps, token_entropies, token_texts, delimiter):
    """Exact alignment, walking decoded token strings against step text."""
    out = []
    tok_i, carry = 0, ""
    for i, step in enumerate(steps):
        need = len(step)
        got = len(carry)
        chunk: List[float] = []
        while got < need and tok_i < len(token_texts):
            chunk.append(token_entropies[tok_i])
            got += len(token_texts[tok_i])
            tok_i += 1
        # Tokens spanning the delimiter are charged to the step just closed.
        carry = ""
        if not chunk:
            chunk = [0.0]
        out.append(StepMeasurement(
            index=i, text=step, n_tokens=len(chunk),
            entropy=sum(chunk) / len(chunk), total_entropy=sum(chunk),
        ))
        # Skip the delimiter's own tokens.
        skipped = 0
        while tok_i < len(token_texts) and skipped < len(delimiter):
            if token_texts[tok_i].strip():
                break
            skipped += len(token_texts[tok_i])
            tok_i += 1
    return out


# ----------------------------------------------------------------------
# Pruning
# ----------------------------------------------------------------------
def select_to_prune(
    steps: Sequence[StepMeasurement],
    ratio: float = 0.8,
    strategy: str = "low",
    seed: int = 0,
) -> List[int]:
    """Choose which step indices to remove.

    Parameters
    ----------
    ratio:
        Fraction of steps to remove. The paper's headline uses 0.8.
    strategy:
        ``low``    - remove the lowest-entropy steps (the paper's method).
        ``high``   - remove the highest-entropy steps.
        ``random`` - remove a random subset.

    The last two exist because they are the controls that make the result
    meaningful. "Removing 80% of steps barely hurts" is only interesting if
    removing a *different* 80% hurts a lot - otherwise the chain was simply
    decorative and the entropy criterion did no work.
    """
    n = len(steps)
    if n == 0 or ratio <= 0:
        return []
    k = min(n, max(1, int(round(n * ratio))))

    if strategy == "random":
        import random
        return sorted(random.Random(seed).sample(range(n), k))
    reverse = strategy == "high"
    order = sorted(steps, key=lambda s: s.entropy, reverse=reverse)
    return sorted(s.index for s in order[:k])


def apply_pruning(
    steps: Sequence[StepMeasurement],
    prune_idx: Sequence[int],
    skip_token: str = SKIP_TOKEN,
    collapse: bool = True,
) -> str:
    """Rebuild the chain with pruned steps replaced by a skip marker.

    ``collapse`` folds a run of consecutive skips into one marker, which is
    what keeps the token saving large: eighty percent pruning on a
    forty-step chain otherwise leaves thirty-two separate markers.
    """
    drop = set(prune_idx)
    parts, last_skipped = [], False
    for s in steps:
        if s.index in drop:
            if not (collapse and last_skipped):
                parts.append(skip_token)
            last_skipped = True
        else:
            parts.append(s.text)
            last_skipped = False
    return STEP_DELIMITER.join(parts)


def pruning_report(
    steps: Sequence[StepMeasurement], prune_idx: Sequence[int]
) -> Dict[str, Any]:
    """Summary statistics for one pruning decision."""
    drop = set(prune_idx)
    kept = [s for s in steps if s.index not in drop]
    dropped = [s for s in steps if s.index in drop]
    tok_all = sum(s.n_tokens for s in steps) or 1
    tok_kept = sum(s.n_tokens for s in kept)
    return {
        "n_steps": len(steps),
        "n_pruned": len(dropped),
        "prune_ratio": len(dropped) / len(steps) if steps else 0.0,
        "tokens_before": tok_all,
        "tokens_after": tok_kept,
        "token_reduction": 1.0 - tok_kept / tok_all,
        "mean_entropy_kept": (sum(s.entropy for s in kept) / len(kept)) if kept else 0.0,
        "mean_entropy_pruned": (sum(s.entropy for s in dropped) / len(dropped)) if dropped else 0.0,
    }
