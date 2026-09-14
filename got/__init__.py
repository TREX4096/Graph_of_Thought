"""
Graph of Thoughts (GoT) -- a replication of Besta et al., AAAI 2024.

Public API
----------
Building a GoT program means three things:

  1. Pick a backend      -- got.backends.MockLM / LlamaCppLM / HFLM / VLLMLM
  2. Build a GoO         -- wire Operation objects with add_predecessor()
  3. Run it              -- Controller(lm, prompter, parser, leaves).run()

See got/tasks/sorting/ for a fully worked example.
"""

from .thought import Thought
from .operations import (
    Operation,
    InputOp,
    Generate,
    Aggregate,
    PairwiseAggregate,
    Improve,
    Score,
    KeepBest,
    KeepBestPerGroup,
    KeepValid,
    Selector,
    GroundTruth,
)
from .controller import Controller
from .prompter import AbstractPrompter, AbstractParser
from .metrics import volume, latency, graph_metrics, theoretical_bounds

__version__ = "0.1.0"

__all__ = [
    "Thought", "Operation", "InputOp", "Generate", "Aggregate", "Improve",
    "Score", "KeepBest", "KeepBestPerGroup", "KeepValid", "Selector", "GroundTruth",
    "PairwiseAggregate",
    "Controller", "AbstractPrompter", "AbstractParser",
    "volume", "latency", "graph_metrics", "theoretical_bounds",
]
