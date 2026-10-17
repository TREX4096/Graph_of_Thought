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


def extract_answer(text: str) -> str:
    """Pull the final answer out of a model completion.

    Order matters: a boxed expression is the most explicit signal, then an
    'answer is' marker, then the last number in the text. The last-number
    fallback is deliberately last - it is right for GSM8K and frequently
    wrong for MATH-500, where the answer may be an expression.
    """
    if not text:
        return ""
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


def normalise_answer(a: str) -> str:
    """Canonicalise a LaTeX or plain answer for comparison."""
    if a is None:
        return ""
    s = str(a).strip()
    s = _DEGREES.sub("", s)
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
    return False
