"""
Answer extraction and comparison for GSM8K and MATH-500.
========================================================

Grading maths answers by string equality is wrong often enough to change a
headline number. "1/2", "\\frac{1}{2}" and "0.5" are one answer written
three ways; "18" and "$18" differ only in a currency symbol the model was
never asked to omit. A scorer that calls those wrong will understate
accuracy uniformly - and worse, it will understate it *unevenly* between
the full and the compressed chain, because the compressed chain tends to
produce terser formatting.

So this module normalises before comparing, and falls back to numeric
equality when both sides parse as numbers.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

# Model is instructed to end with this marker; we also accept \boxed{}.
ANSWER_MARKER = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*", re.I)
BOXED = re.compile(r"\\boxed\{")


def _last_boxed(text: str) -> Optional[str]:
    """Content of the final \\boxed{...}, respecting nested braces."""
    starts = [m.end() for m in BOXED.finditer(text)]
    if not starts:
        return None
    i, depth, buf = starts[-1], 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf).strip()


NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Models occasionally wrap the answer in markup - we have seen
# '<span class="final-answer">448/15625'. The value inside was correct,
# so that was our extraction failing, not the model. Strip tags first.
HTML_TAG = re.compile(r"<[^>]{0,80}>")


def extract_answer(text: str) -> str:
    """Pull the final answer out of a model completion.

    Order matters: a boxed expression is the most explicit signal, then an
    'answer is' marker, then the last number in the text. The last-number
    fallback is deliberately last - it is right for GSM8K and frequently
    wrong for MATH-500, where the answer may be an expression.
    """
    if not text:
        return ""
    text = HTML_TAG.sub("", text)
    boxed = _last_boxed(text)
    if boxed:
        return boxed.strip()

    tail = text.strip().splitlines()
    for line in reversed(tail[-6:]):            # answers live near the end
        m = ANSWER_MARKER.search(line)
        if m:
            cand = line[m.end():].strip().rstrip(".").strip()
            if cand:
                return cand

    # If what remains is already a bare answer - a fraction, a surd, a short
    # expression - return it whole. Falling through to "last number in the
    # text" would turn 448/15625 into 15625 and 11\sqrt{2} into 2, picking
    # a fragment of the answer rather than the answer.
    stripped = text.strip()
    if stripped and len(stripped) <= 40 and "\n" not in stripped:
        # Ignore LaTeX command names when deciding whether this is prose.
        words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[a-zA-Z]+", "", stripped))
        if not words:                       # no prose - it is the answer itself
            return stripped

    nums = NUMBER.findall(text)
    return nums[-1].replace(",", "") if nums else ""


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------
_STRIP = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\:|\\ "), ""),
    (re.compile(r"\\text\{[^}]*\}"), ""),
    (re.compile(r"\\mbox\{[^}]*\}"), ""),
    (re.compile(r"\s+"), ""),
    (re.compile(r"^\$+|\$+$"), ""),
    (re.compile(r"^\\\((.*)\\\)$"), r"\1"),
    (re.compile(r"\\%|%"), ""),
    (re.compile(r"^\{(.*)\}$"), r"\1"),
]
_DEGREES = re.compile(r"\^\\?circ|\\degree")


# "x=5" and "5" are the same answer; the variable belongs to the
# question, not to the model. Only a single letter being assigned is
# stripped, so an equation that genuinely IS the answer survives.
_ASSIGN = re.compile(r"^[a-z]\s*=\s*", re.I)

# "3 treeks" is 3. Only a trailing alphabetic word after a bare number
# is dropped, so LaTeX commands and symbolic answers are untouched.
_TRAILING_UNIT = re.compile(r"^(-?[\d.,]+)\s*[a-z]{2,}\.?$", re.I)


def normalise_answer(a: str) -> str:
    """Canonicalise a LaTeX or plain answer for comparison."""
    if a is None:
        return ""
    s = str(a).strip()
    s = HTML_TAG.sub("", s)
    s = _DEGREES.sub("", s)
    m = _TRAILING_UNIT.match(s)
    if m:
        s = m.group(1)
    s = _ASSIGN.sub("", s)
    for pat, rep in _STRIP:
        s = pat.sub(rep, s)
    s = s.replace(",", "")
    # \dfrac and \tfrac are \frac with different display hints.
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Trailing ".0" on an integer-valued float.
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()


FRAC = re.compile(r"^\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")
SIMPLE_FRAC = re.compile(r"^(-?[\d.]+)/(-?[\d.]+)$")


def _as_number(s: str) -> Optional[float]:
    """Interpret a normalised answer as a number, if it is one."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    for pat in (FRAC, SIMPLE_FRAC):
        m = pat.match(s)
        if m:
            try:
                return float(Fraction(float(m.group(1))) / Fraction(float(m.group(2))))
            except (ValueError, ZeroDivisionError):
                return None
    return None


def _strip_braces(s: str) -> str:
    return re.sub(r"\{([^{}]*)\}", r"\1", s)


def answers_match(produced: str, truth: str, tol: float = 1e-6) -> bool:
    """True when two answers denote the same value.

    Tries, in order: exact match after normalisation, then numeric equality
    within ``tol``. The numeric path is what catches 1/2 against 0.5.
    """
    p, t = normalise_answer(produced), normalise_answer(truth)
    if not p or not t:
        return False
    if p == t:
        return True
    pn, tn = _as_number(p), _as_number(t)
    if pn is not None and tn is not None:
        return abs(pn - tn) <= tol * max(1.0, abs(tn))
    # Brace style must not decide correctness: 11\\sqrt{2} is 11\\sqrt2.
    # Last resort, on fully normalised strings only, so it cannot
    # collapse genuinely different expressions.
    return _strip_braces(p) == _strip_braces(t)
