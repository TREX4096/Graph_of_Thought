"""
Prompts for the step-entropy experiment on GSM8K and MATH-500.
==============================================================

Three prompts, matching the three stages of the paper's static-pruning
protocol:

  1. COT_PROMPT          generate a full reasoning chain
  2. (segment + prune)   handled in got/entropy.py, no model call
  3. build_compressed_prompt  hand the pruned chain back and ask only for
                              the final answer

Stage 3 is the measurement that matters. The model is shown its own
reasoning with most steps replaced by [SKIP], and must still produce the
right answer. If accuracy survives, those steps were not carrying the
result.

ANSWER_ONLY_PROMPT is the no-reasoning control - the floor that any
compression scheme has to beat to be worth anything.
"""

from __future__ import annotations

COT_PROMPT = """Solve the following problem. Reason step by step, putting each step on its own line with a blank line between steps.

End your response with the final answer on its own line, in the form:
Final answer: <answer>

Problem: {question}
"""


ANSWER_ONLY_PROMPT = """Solve the following problem. Give only the final answer, with no reasoning.

Respond with exactly one line, in the form:
Final answer: <answer>

Problem: {question}
"""


# The compressed chain is presented as the model's own prior reasoning.
# Telling it explicitly that [SKIP] marks elided steps matters: the paper
# finds the marker degrades far less than a silent deletion, because the
# model is told reasoning happened rather than being shown a gap.
COMPRESSED_PROMPT = """Solve the following problem.

Below is a partial reasoning trace. Steps marked [SKIP] were omitted for brevity - assume they were completed correctly.

Problem: {question}

Reasoning:
{chain}

Based on the reasoning above, state the final answer on its own line, in the form:
Final answer: <answer>
"""


def build_compressed_prompt(question: str, chain: str) -> str:
    """Stage-3 prompt: pruned chain in, final answer out."""
    return COMPRESSED_PROMPT.format(question=question.strip(), chain=chain.strip())
