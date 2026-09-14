"""
Generate the analytical figures used in explanation.md.
=======================================================

These are not decoration: each one plots a quantity that is derived
symbolically in the write-up, so the reader can check the algebra against the
curve. All four are computed from closed-form expressions -- no LLM involved.

Outputs to docs/figures/:
    theory_volume_latency.png   Table 2 as functions of N
    theory_decomposition.png    why splitting beats one long call
    theory_best_of_k.png        order statistics of best-of-k selection
    theory_cost_quality.png     the cost a given success probability buys
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOURS = {
    "CoT": "#4C72B0",
    "CoT-SC": "#DD8452",
    "ToT": "#55A868",
    "GoT": "#C44E52",
    "IO": "#8C8C8C",
}


# ----------------------------------------------------------------------
# Figure 1 -- Table 2 as continuous functions of N
# ----------------------------------------------------------------------
def fig_volume_latency(out_dir: str, k: int = 4) -> None:
    """Plot latency(N) and volume(N) for each scheme, per GoT Section 6.

    Latency:  CoT = N,  CoT-SC = N/k,  ToT = GoT = log_k N
    Volume:   CoT = N,  CoT-SC = N/k,  ToT = log_k N,  GoT = N
    """
    N = np.logspace(1, 4, 200)
    logk = np.log(N) / np.log(k)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    # --- latency ---
    ax1.loglog(N, N, label="CoT", color=COLOURS["CoT"], lw=2)
    ax1.loglog(N, N / k, label="CoT-SC", color=COLOURS["CoT-SC"], lw=2)
    ax1.loglog(N, logk, label="ToT", color=COLOURS["ToT"], lw=2, ls="--")
    ax1.loglog(N, logk, label="GoT", color=COLOURS["GoT"], lw=2.5, ls=":")
    ax1.set_title("Latency  (lower is better)")
    ax1.set_xlabel("total thought budget $N$")
    ax1.set_ylabel("hops to final thought")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25, which="both")

    # --- volume ---
    ax2.loglog(N, N, label="CoT", color=COLOURS["CoT"], lw=2)
    ax2.loglog(N, N / k, label="CoT-SC", color=COLOURS["CoT-SC"], lw=2)
    ax2.loglog(N, logk, label="ToT", color=COLOURS["ToT"], lw=2)
    ax2.loglog(N, N, label="GoT", color=COLOURS["GoT"], lw=2.5, ls=":")
    ax2.set_title("Volume  (higher is better)")
    ax2.set_xlabel("total thought budget $N$")
    ax2.set_ylabel("ancestor thoughts")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25, which="both")

    # --- the ratio: what you actually want ---
    ax3.loglog(N, N / N, label="CoT", color=COLOURS["CoT"], lw=2)
    ax3.loglog(N, (N / k) / (N / k), label="CoT-SC", color=COLOURS["CoT-SC"], lw=2)
    ax3.loglog(N, logk / logk, label="ToT", color=COLOURS["ToT"], lw=2)
    ax3.loglog(N, N / logk, label="GoT", color=COLOURS["GoT"], lw=2.5)
    ax3.set_title("Volume / Latency  (higher is better)")
    ax3.set_xlabel("total thought budget $N$")
    ax3.set_ylabel(r"$V/L$")
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(alpha=0.25, which="both")

    fig.suptitle(
        r"GoT Table 2 as functions of $N$  ($k=%d$):  "
        r"only GoT holds $V=\Theta(N)$ at $L=\Theta(\log_k N)$" % k,
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "theory_volume_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
# Figure 2 -- why decomposition beats one long call
# ----------------------------------------------------------------------
def fig_decomposition(out_dir: str, lam: float = 0.035) -> None:
    r"""Compare P(success) for a monolithic call vs merge-sort decomposition.

    Model: the probability an LLM handles a length-$n$ sequence correctly
    decays geometrically in the number of items it must track,

        p(n) = e^{-\lambda n}

    **Monolithic (IO):**      P = p(n)

    **Decomposed (GoT)** with $m$ chunks, best-of-$k$ per step, exact scoring:
    a step succeeds if at least one of $k$ samples is right,

        q(n) = 1 - (1 - p(n))^k

    so all $m$ chunks succeed with $q(n/m)^m$, and the $m-1$ merges of a
    binary tree each succeed with $q_{merge}$:

        P = q(n/m)^m  \cdot  \prod_{levels} q(\cdot)^{\#merges}

    The curve shows the crossover: decomposition wins once $n$ is large
    enough that $p(n)$ has collapsed but $p(n/m)$ has not.
    """
    n = np.arange(4, 161)
    m, k = 4, 3

    def p(x):
        return np.exp(-lam * np.asarray(x, dtype=float))

    def best_of_k(prob, kk):
        return 1.0 - (1.0 - prob) ** kk

    # Monolithic
    p_io = p(n)

    # Decomposed: m chunks, then log2(m) merge levels
    p_chunk = best_of_k(p(n / m), k) ** m
    p_merge = np.ones_like(n, dtype=float)
    size = n / m
    remaining = m
    while remaining > 1:
        size = size * 2
        n_merges = remaining // 2
        # Merging two sorted lists is easier than sorting: halve the rate.
        p_merge *= best_of_k(p(size * 0.5), 10) ** n_merges
        remaining //= 2
    p_got = p_chunk * p_merge

    # ToT-like: best-of-k on the whole sequence, no decomposition
    p_tot = best_of_k(p_io, k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax1.plot(n, p_io, label="IO: $p(n)$", color=COLOURS["IO"], lw=2)
    ax1.plot(n, p_tot, label=r"ToT: $1-(1-p(n))^k$", color=COLOURS["ToT"], lw=2)
    ax1.plot(n, p_got, label="GoT: decomposed", color=COLOURS["GoT"], lw=2.5)
    ax1.axvline(64, color="k", ls=":", lw=1, alpha=0.6)
    ax1.annotate("n = 64\n(paper's setting)", xy=(64, 0.5), xytext=(78, 0.62),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.set_xlabel("input length $n$")
    ax1.set_ylabel(r"$P(\mathrm{success})$")
    ax1.set_title(r"Success probability   ($\lambda=%.3f$, $m=%d$, $k=%d$)"
                  % (lam, m, k))
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    # Advantage ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_io > 1e-12, p_got / p_io, np.nan)
    ax2.semilogy(n, ratio, color=COLOURS["GoT"], lw=2.5)
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("input length $n$")
    ax2.set_ylabel(r"$P_{GoT}\,/\,P_{IO}$")
    ax2.set_title("Decomposition advantage grows with length")
    ax2.grid(alpha=0.25, which="both")

    fig.suptitle("Why merge-sort decomposition helps: the exponential is taken "
                 "on $n/m$, not $n$", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "theory_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
# Figure 3 -- order statistics of best-of-k
# ----------------------------------------------------------------------
def fig_best_of_k(out_dir: str) -> None:
    r"""Plot $1-(1-p)^k$ and its diminishing returns.

    With an **exact** scorer, generating $k$ candidates and keeping the best
    succeeds whenever at least one candidate is correct:

        q_k(p) = 1 - (1-p)^k

    The marginal gain of the $k$-th sample is

        \frac{\partial q_k}{\partial k} = -(1-p)^k \ln(1-p)

    which decays geometrically -- the justification for the paper's modest
    $k=3$ on chunks, and for treating $k$ as the first cost knob to turn down.
    """
    p = np.linspace(0.01, 0.99, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for kk in (1, 2, 3, 5, 10, 20):
        ax1.plot(p, 1 - (1 - p) ** kk, lw=2, label=f"k = {kk}")
    ax1.plot(p, p, "k:", lw=1, alpha=0.5)
    ax1.set_xlabel("single-sample success probability $p$")
    ax1.set_ylabel(r"$q_k = 1-(1-p)^k$")
    ax1.set_title("Best-of-$k$ with an exact scorer")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    ks = np.arange(1, 21)
    for pv, style in ((0.1, "-"), (0.3, "--"), (0.5, ":")):
        gain = (1 - (1 - pv) ** ks) - (1 - (1 - pv) ** (ks - 1))
        ax2.plot(ks, gain, style, lw=2, label=f"p = {pv}")
    ax2.set_xlabel("$k$")
    ax2.set_ylabel("marginal gain of the $k$-th sample")
    ax2.set_title("Diminishing returns: cost is linear in $k$, benefit is not")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.set_xticks([1, 5, 10, 15, 20])

    fig.suptitle("Order statistics behind Generate(k) + KeepBest(1)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "theory_best_of_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
# Figure 4 -- cost model
# ----------------------------------------------------------------------
def fig_cost_quality(out_dir: str) -> None:
    r"""Token cost of each scheme as a function of input length.

    Decode cost dominates. For sorting, an answer of $n$ digits costs
    $\Theta(n)$ tokens, so with $m$ chunks, branching $k$ and $k_a$
    aggregation attempts:

        C_{IO}   = n
        C_{CoT}  = 2n
        C_{SC}   = k\,n
        C_{ToT}  = k\,n + (d-1)\,n
        C_{GoT}  = \underbrace{m\,k\,(n/m)}_{chunks}
                 + \underbrace{k_a \sum_{\ell=1}^{\log_2 m} \tfrac{m}{2^\ell}\cdot\tfrac{2^\ell n}{m}}_{merges}
                 = k\,n + k_a\, n \log_2 m

    The merge term is the price of aggregation -- and the reason
    ``aggregation_attempts`` is the first knob to turn down on a cluster.
    """
    n = np.arange(16, 257, 8)
    m, k, ka, d = 4, 3, 10, 3

    c_io = n.astype(float)
    c_cot = 2.0 * n
    c_sc = k * n
    c_tot = k * n + (d - 1) * n
    c_got = k * n + ka * n * math.log2(m)
    c_got_cheap = k * n + 3 * n * math.log2(m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax1.plot(n, c_io, label="IO", color=COLOURS["IO"], lw=2)
    ax1.plot(n, c_cot, label="CoT", color=COLOURS["CoT"], lw=2)
    ax1.plot(n, c_sc, label=f"CoT-SC (k={k})", color=COLOURS["CoT-SC"], lw=2)
    ax1.plot(n, c_tot, label=f"ToT (k={k}, d={d})", color=COLOURS["ToT"], lw=2)
    ax1.plot(n, c_got, label=f"GoT ($k_a$={ka})", color=COLOURS["GoT"], lw=2.5)
    ax1.plot(n, c_got_cheap, label="GoT ($k_a$=3)", color=COLOURS["GoT"],
             lw=2, ls="--", alpha=0.7)
    ax1.set_xlabel("input length $n$")
    ax1.set_ylabel("decode tokens (arbitrary units)")
    ax1.set_title("Cost is linear in $n$; the constant is what differs")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    kas = np.arange(1, 21)
    cost = k * 64 + kas * 64 * math.log2(m)
    ax2.plot(kas, cost / cost[0], color=COLOURS["GoT"], lw=2.5)
    ax2.set_xlabel(r"aggregation attempts $k_a$")
    ax2.set_ylabel("relative cost  (n = 64)")
    ax2.set_title(r"$k_a$ is the dominant GoT cost knob")
    ax2.grid(alpha=0.25)
    ax2.set_xticks([1, 5, 10, 15, 20])
    for kk in (3, 5, 10):
        ax2.annotate(f"$k_a$={kk}", xy=(kk, cost[kk - 1] / cost[0]),
                     xytext=(kk + 0.6, cost[kk - 1] / cost[0] - 0.35),
                     fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))

    fig.suptitle("Token-cost model for sorting", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "theory_cost_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate analytical figures")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Generating analytical figures:")
    fig_volume_latency(args.out)
    fig_decomposition(args.out)
    fig_best_of_k(args.out)
    fig_cost_quality(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
