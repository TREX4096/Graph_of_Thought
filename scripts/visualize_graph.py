"""
Draw the reasoning graph produced by a run.
===========================================

Produces the picture that makes GoT's contribution obvious at a glance: for
IO/CoT/CoT-SC/ToT the graph is a tree (every node has one parent), while for
GoT edges converge -- vertices with in-degree > 1 are the aggregations.

Also plots the latency-volume tradeoff of Table 2, measured on the graphs we
actually built rather than taken from the paper.

Usage
-----
    python scripts/visualize_graph.py --out docs/figures
    python scripts/visualize_graph.py --schemes got tot --length 32 --out docs/figures

Outputs PNGs suitable for dropping straight into the project report.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")            # headless: no display on HPC compute nodes
import matplotlib.pyplot as plt
import networkx as nx

from got import Controller, graph_metrics, latency, theoretical_bounds, volume
from got.backends import MockLM
from got.tasks.sorting import SortingParser, SortingPrompter
from got.tasks.sorting.graphs import SCHEMES


# Colour by role so the structure reads without a legend lookup.
COLOURS = {
    "input": "#4C72B0",       # blue  -- the problem
    "aggregate": "#C44E52",   # red   -- GoT's distinctive operation
    "keepbest": "#55A868",    # green -- pruning / ranking
    "other": "#B0B0B0",       # grey  -- ordinary generation
}


def build_nx_graph(thoughts) -> nx.DiGraph:
    """Convert our Thought objects into a networkx DiGraph for layout."""
    g = nx.DiGraph()
    for t in thoughts:
        g.add_node(
            t.id,
            operation=t.operation,
            score=t.score,
            is_aggregate=t.is_aggregate,
        )
    for t in thoughts:
        for p in t.predecessors:
            g.add_edge(p.id, t.id)
    return g


def layer_positions(g: nx.DiGraph) -> Dict[int, tuple]:
    """Lay nodes out in layers by depth from the source.

    A hierarchical layout is essential here: a force-directed layout would
    scramble the merge tree and hide the very structure we are trying to show.
    """
    # Longest-path depth puts a node below ALL of its ancestors, which keeps
    # aggregation edges pointing downwards.
    depth: Dict[int, int] = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        depth[node] = 0 if not preds else max(depth[p] for p in preds) + 1

    by_layer: Dict[int, List[int]] = {}
    for node, d in depth.items():
        by_layer.setdefault(d, []).append(node)

    pos = {}
    for d, nodes in by_layer.items():
        nodes.sort()
        width = max(len(nodes), 1)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2.0) * 1.4
            pos[node] = (x, -d)
    return pos


def draw_scheme(scheme: str, numbers: List[int], out_dir: str, args) -> Dict:
    """Run one scheme with the mock backend and save a picture of its graph."""
    lm = MockLM(error_rate=0.15, seed=args.seed)
    builder = SCHEMES[scheme]
    if scheme == "got":
        leaves = builder(numbers, num_chunks=args.num_chunks,
                         branching_factor=args.branching_factor,
                         aggregation_attempts=args.aggregation_attempts)
    else:
        leaves = builder(numbers)

    ctrl = Controller(lm, SortingPrompter(), SortingParser(), leaves)
    ctrl.run(num_chunks=args.num_chunks)

    thoughts = ctrl.all_thoughts()
    g = build_nx_graph(thoughts)
    pos = layer_positions(g)

    node_colours = []
    for node in g.nodes():
        op = g.nodes[node]["operation"]
        if op == "input":
            node_colours.append(COLOURS["input"])
        elif g.nodes[node]["is_aggregate"]:
            node_colours.append(COLOURS["aggregate"])
        elif op == "keepbest":
            node_colours.append(COLOURS["keepbest"])
        else:
            node_colours.append(COLOURS["other"])

    m = graph_metrics(thoughts)
    fig, ax = plt.subplots(figsize=(max(7, len(pos) * 0.35), 7))

    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color="#888888", arrows=True,
        arrowsize=9, width=0.9, alpha=0.75,
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colours, node_size=260,
        edgecolors="white", linewidths=1.0,
    )

    n_agg = sum(1 for t in thoughts if t.is_aggregate)
    ax.set_title(
        f"{scheme.upper()} -- reasoning graph\n"
        f"{len(thoughts)} thoughts, {n_agg} aggregations | "
        f"max volume {m['max_volume']}, max latency {m['max_latency']}",
        fontsize=11,
    )
    ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   color=COLOURS["input"], label="input"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   color=COLOURS["other"], label="generated thought"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   color=COLOURS["keepbest"], label="kept (ranked best)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   color=COLOURS["aggregate"], label="aggregation (in-degree > 1)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"graph_{scheme}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

    return {"scheme": scheme, **m, "n_aggregations": n_agg}


def plot_latency_volume(measured: List[Dict], out_dir: str) -> None:
    """Scatter measured (latency, volume) per scheme, with Table 2 alongside.

    The interesting visual: GoT sits far above ToT on the volume axis at
    comparable latency -- which is precisely the claim of Section 6.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for m in measured:
        ax1.scatter(m["max_latency"], m["max_volume"], s=140, zorder=3)
        ax1.annotate(
            m["scheme"].upper(),
            (m["max_latency"], m["max_volume"]),
            textcoords="offset points", xytext=(8, 6), fontsize=10,
        )
    ax1.set_xlabel("latency (hops to final thought)")
    ax1.set_ylabel("volume (ancestor thoughts)")
    ax1.set_title("Measured on real runs")
    ax1.grid(alpha=0.3)

    bounds = theoretical_bounds(n=64, k=4)
    for name, b in bounds.items():
        ax2.scatter(b["latency"], b["volume"], s=140, zorder=3)
        ax2.annotate(name, (b["latency"], b["volume"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax2.set_xlabel("latency")
    ax2.set_ylabel("volume")
    ax2.set_title("Paper Table 2 (N=64, k=4)")
    ax2.grid(alpha=0.3)

    fig.suptitle("The latency-volume tradeoff (GoT Section 6)", fontsize=13)
    path = os.path.join(out_dir, "latency_volume.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise GoT reasoning graphs")
    ap.add_argument("--schemes", nargs="+",
                    default=["io", "cot", "cot_sc", "tot", "got"])
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--branching-factor", type=int, default=3)
    ap.add_argument("--aggregation-attempts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    numbers = [rng.randint(0, 9) for _ in range(args.length)]

    print(f"Visualising {len(args.schemes)} schemes on a {args.length}-element input")
    measured = [draw_scheme(s, numbers, args.out, args) for s in args.schemes]
    plot_latency_volume(measured, args.out)

    print("\nMeasured structure:")
    print(f"{'scheme':8s} {'thoughts':>9s} {'aggs':>6s} {'volume':>8s} {'latency':>8s}")
    for m in measured:
        print(f"{m['scheme']:8s} {m['n_thoughts']:9d} {m['n_aggregations']:6d} "
              f"{m['max_volume']:8d} {m['max_latency']:8d}")


if __name__ == "__main__":
    main()
