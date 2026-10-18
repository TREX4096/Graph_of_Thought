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


# Stage 3 is a CONTINUATION, not a new question.
#
# The paper places the compressed chain inside the think context - after the
# query, before </think> - so the model resumes its own partial reasoning and
# writes only the answer. An earlier version of this file instead opened a
# fresh user turn containing a "Reasoning:" block, which is a different task:
# read someone else's working rather than finish your own. On a reasoning
# model that is out of distribution, and it showed - the full-chain baseline
# came in 19 points below the paper's and every pruning strategy collapsed to
# the same accuracy, which is what you see when the supplied chain is being
# ignored.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def build_compressed_prefill(chain: str, think_tags: bool = True) -> str:
    """Assistant-turn prefill holding the pruned chain.

    Returned text is appended after the generation prompt, so the model
    continues from it. With ``think_tags`` the chain is wrapped so a
    reasoning model sees a completed think block and moves to the answer;
    without them it is plain prefilled reasoning, which suits an instruct
    model that never emits think tags at all.
    """
    body = chain.strip()
    if think_tags:
        return f"{THINK_OPEN}\n{body}\n{THINK_CLOSE}\n\nFinal answer:"
    return f"{body}\n\nFinal answer:"


def build_compressed_prompt(question: str, chain: str) -> str:
    """Deprecated: kept so older result scripts still import cleanly.

    Use :func:`build_compressed_prefill` with the prompt from
    :data:`COT_PROMPT`. See the note above on why a separate user turn
    measures the wrong thing.
    """
    return COT_PROMPT.format(question=question.strip())
