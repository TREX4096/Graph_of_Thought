"""
The latency-volume tradeoff (GoT paper, Section 6 and Table 2).
===============================================================

This is one of the paper's headline theoretical contributions, and it is
directly measurable on a graph we have actually built -- so we implement it
rather than merely citing it.

Definitions (Section 6)
-----------------------
* **Latency** of a thought t: the number of hops in the graph of thoughts
  needed to reach t from the input. I.e. the length of the *shortest* path
  from a source vertex to t.

* **Volume** of a thought t: "the number of preceding LLM thoughts that could
  have impacted t. Formally, the volume of t is the number of thoughts from
  which there exists a path to t in the graph of thoughts."

Volume measures how much accumulated reasoning is available to a thought.
Latency measures how long you waited for it. A good scheme wants high volume
(well-informed answers) at low latency (cheap, parallel).

The paper's analytical result (Table 2), with total cost fixed at Theta(N):

    Scheme    Latency      Volume
    -------   ----------   ------------
    CoT       N            N
    CoT-SC    N/k          N/k
    ToT       log_k N      O(log_k N)
    GoT       log_k N      N

The insight: ToT buys low latency by *giving up* volume -- a leaf of a tree
only sees its own root-to-leaf path, which is log_k N thoughts. GoT recovers
full volume N at the same latency, because aggregation lets every thought
reach the final answer. That is the quantitative case for using a graph.

Empirical verification
----------------------
``volume()`` and ``latency()`` below compute these on a real run's graph, so
``scripts/verify_latency_volume.py`` can confirm that a GoT run genuinely
achieves higher volume than an equivalent ToT run. This turns a table in a
paper into a reproduced experimental result -- exactly what a replication
should do.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional

from .thought import Thought


def volume(thought: Thought) -> int:
    """Number of thoughts from which a path exists to ``thought``.

    Computed by a reverse BFS over predecessor edges. The thought itself is
    excluded, matching the paper's "preceding LLM thoughts".
    """
    seen = set()
    queue = deque(thought.predecessors)
    while queue:
        node = queue.popleft()
        if node.id in seen:
            continue
        seen.add(node.id)
        queue.extend(node.predecessors)
    return len(seen)


def latency(thought: Thought) -> int:
    """Hops along the shortest path from a source vertex to ``thought``.

    A source is any thought with no predecessors -- in practice the Input
    vertex. Shortest (not longest) path is used because latency asks "how
    soon could this thought have been produced", and independent branches
    execute in parallel.
    """
    # BFS backwards; the first source we reach is at minimum distance.
    depth: Dict[int, int] = {thought.id: 0}
    queue = deque([thought])
    best: Optional[int] = None

    while queue:
        node = queue.popleft()
        d = depth[node.id]
        if not node.predecessors:
            best = d if best is None else min(best, d)
            continue
        for p in node.predecessors:
            if p.id not in depth or depth[p.id] > d + 1:
                depth[p.id] = d + 1
                queue.append(p)

    return best if best is not None else 0


def graph_metrics(thoughts: Iterable[Thought]) -> Dict[str, float]:
    """Aggregate latency/volume statistics over a whole reasoning graph.

    Returns the metrics for the *final* thoughts (those with no successors),
    since those are the answers the user actually receives.
    """
    thoughts = list(thoughts)
    if not thoughts:
        return {}

    finals = [t for t in thoughts if not t.successors] or thoughts

    volumes = [volume(t) for t in finals]
    latencies = [latency(t) for t in finals]

    return {
        "n_thoughts": len(thoughts),
        "n_final_thoughts": len(finals),
        "max_volume": max(volumes),
        "mean_volume": sum(volumes) / len(volumes),
        "max_latency": max(latencies),
        "mean_latency": sum(latencies) / len(latencies),
        # The headline ratio: volume gained per unit of latency paid.
        # GoT should score far higher here than ToT on the same task.
        "volume_per_latency": (
            max(volumes) / max(latencies) if max(latencies) > 0 else float(max(volumes))
        ),
    }


def theoretical_bounds(n: int, k: int) -> Dict[str, Dict[str, float]]:
    """Table 2 of the paper, evaluated numerically for given N and k.

    Useful for plotting the analytical curves alongside measured values.

    Parameters
    ----------
    n:
        Total thought budget N (the paper fixes total cost at Theta(N)).
    k:
        Branching factor.
    """
    import math

    log_k_n = math.log(n, k) if k > 1 else float(n)

    return {
        "CoT":    {"latency": float(n),      "volume": float(n)},
        "CoT-SC": {"latency": n / k,         "volume": n / k},
        "ToT":    {"latency": log_k_n,       "volume": log_k_n},
        "GoT":    {"latency": log_k_n,       "volume": float(n)},
    }
