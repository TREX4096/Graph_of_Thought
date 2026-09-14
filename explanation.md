# explanation.md — Mathematical Deep Dive and Design Rationale

This document develops the theory behind the four papers **formally**, derives the
results rather than quoting them, and records why the implementation is built the way
it is.

Reading order follows the project instructions, and it is the right one: each paper is a
direct response to a mathematical limitation of the previous.

```
Chain-of-Thought  →  Tree of Thoughts  →  Graph of Thoughts
   (2022)               (2023)                (2024)
      │                                          ▲
      └──────── Multimodal-CoT (2023) ───────────┘
                (orthogonal: adds vision, not structure)
```

**Contents**

1. [Notation](#1-notation)
2. [The autoregressive bottleneck — stated formally](#2-the-autoregressive-bottleneck--stated-formally)
3. [Paper 1 — Chain-of-Thought](#3-paper-1--chain-of-thought)
4. [Paper 2 — Tree of Thoughts](#4-paper-2--tree-of-thoughts)
5. [Paper 3 — Multimodal Chain-of-Thought](#5-paper-3--multimodal-chain-of-thought)
6. [Paper 4 — Graph of Thoughts](#6-paper-4--graph-of-thoughts)
7. [Why decomposition works — the error model](#7-why-decomposition-works--the-error-model)
8. [Best-of-k and the role of exact scoring](#8-best-of-k-and-the-role-of-exact-scoring)
9. [The latency–volume theorem, with proof](#9-the-latencyvolume-theorem-with-proof)
10. [The scoring functions — formal properties](#10-the-scoring-functions--formal-properties)
11. [Cost model](#11-cost-model)
12. [Glossary](#12-glossary)
13. [Paper → code map](#13-paper--code-map)
14. [Implementation decisions](#14-implementation-decisions)
15. [Bugs found, and what they taught me](#15-bugs-found-and-what-they-taught-me)
16. [What I verified, and what I did not](#16-what-i-verified-and-what-i-did-not)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $p_\theta$ | the language model, with parameters $\theta$ |
| $x$ | problem input |
| $y$ | final answer |
| $z_i$ | the $i$-th intermediate thought |
| $z_{1\ldots n}$ | a sequence of $n$ thoughts |
| $s = [x, z_{1\ldots i}]$ | a *state* — input plus thoughts so far (ToT) |
| $G=(V,E)$ | the reasoning graph (GoT); $V$ thoughts, $E$ dependencies |
| $k$ | branching factor — samples drawn per step |
| $k_a$ | aggregation attempts — samples drawn per merge |
| $m$ | number of chunks the input is split into |
| $N$ | total thought budget (total LLM calls) |
| $n$ | problem size (e.g. list length) |
| $\mathcal{E}(v,G,p_\theta)$ | score of thought $v$ |
| $\mathcal{R}(G,p_\theta,h)$ | the $h$ top-ranked thoughts |
| $V(t)$, $L(t)$ | volume and latency of thought $t$ |

---

## 2. The autoregressive bottleneck — stated formally

A decoder-only transformer defines a distribution over token sequences by the chain rule
of probability:

$$
p_\theta(w_1,\ldots,w_T) \;=\; \prod_{t=1}^{T} p_\theta\!\left(w_t \mid w_{<t}\right)
$$

Two facts follow, and together they are the reason all four papers exist.

**Fact 1 — compute per token is constant.** A forward pass through an $L$-layer model with
hidden width $d$ costs $\Theta(L d^2 + L\,T d)$ per token, *independent of how hard the
question is*. A problem needing more computation than one token's worth cannot get it,
unless more tokens are emitted.

**Fact 2 — decisions are irrevocable.** Sampling $w_t$ conditions everything after it.
There is no operator in this factorisation for "revise $w_{t-5}$". The distribution is
strictly left-to-right.

Writing $y$ for the answer and $z$ for intermediate work, **direct prompting** asks for

$$
y \sim p_\theta(y \mid x)
$$

and the model must compress all reasoning into the forward passes producing $y$. If $y$ is
a single token, that is exactly one forward pass' worth of computation for arbitrarily
hard $x$.

**Every subsequent idea in these papers is a different answer to: what structure should
the intermediate $z$ have?**

| Paper | Structure of $z$ | Formally |
|---|---|---|
| CoT | chain | $z_1 \to z_2 \to \cdots \to z_n \to y$ |
| CoT-SC | $k$ disjoint chains | $k$ i.i.d. samples of the above |
| ToT | tree | $V$ with $\deg^-(v) \le 1$ |
| GoT | DAG | $V$ with $\deg^-(v)$ unbounded |

That last row — the removal of the in-degree constraint — is the entire contribution of
Graph of Thoughts.

---

## 3. Paper 1 — Chain-of-Thought

> Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*,
> NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

### 3.1 The formal object

CoT introduces intermediate thoughts $z_1,\ldots,z_n$ between $x$ and $y$, each sampled
conditioned on everything before it:

$$
z_i \sim p_\theta^{\mathrm{CoT}}\!\left(z_i \mid x, z_{1\ldots i-1}\right),
\qquad
y \sim p_\theta^{\mathrm{CoT}}\!\left(y \mid x, z_{1\ldots n}\right)
$$

In practice the whole thing is sampled as one continuous sequence:

$$
[z_{1\ldots n}, y] \sim p_\theta^{\mathrm{CoT}}(z_{1\ldots n}, y \mid x)
$$

Note what this does **not** specify: the decomposition of $z$ into steps is left
ambiguous — is $z_i$ a phrase, a sentence, a paragraph? ToT's first contribution is to
make that choice explicit.

### 3.2 Why it works — a computational argument

Compare the total computation available.

*Direct:* answer of $T_y$ tokens ⟹ $\Theta(T_y)$ forward passes.

*CoT:* answer preceded by $T_z$ reasoning tokens ⟹ $\Theta(T_z + T_y)$ forward passes.

Since $T_z \gg T_y$ for a hard problem, CoT buys roughly a factor $T_z/T_y$ more
computation — **without changing $\theta$ at all**. The paper states this as its first
listed property:

> "chain of thought, in principle, allows models to decompose multi-step problems into
> intermediate steps, which means that **additional computation can be allocated** to
> problems that require more reasoning steps."

There is a second, information-theoretic reading. The context window acts as an external
memory. Writing $z_i$ into the context makes it available to every later step at
$O(1)$ retrieval cost via attention, whereas an unwritten intermediate result must be
re-derived inside the residual stream at every layer. **The chain of thought is
computation, not just explanation.**

### 3.3 The worked example

Standard prompting on the cafeteria problem yields *"The answer is 27"* — wrong.
CoT prompting yields:

> "The cafeteria had 23 apples originally. They used 20 to make lunch. So they had
> 23 − 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9."

Each arithmetic step is a separate sub-computation with its result written down.

### 3.4 Results and the emergence threshold

| Model | GSM8K solve rate |
|---|---|
| Prior SOTA (fine-tuned GPT-3 175B + verifier) | 33% |
| PaLM 540B, standard prompting | 18% |
| **PaLM 540B, CoT prompting** | **57%** |

Critically, CoT is an **emergent ability of scale**. Writing $A(\theta)$ for CoT accuracy
gain, empirically

$$
A(\theta) \approx 0 \quad\text{for } |\theta| \lesssim 10^{10}, \qquad
A(\theta) > 0 \quad\text{for } |\theta| \gtrsim 10^{10}
$$

Below threshold, models produce fluent but logically invalid chains and then follow them
confidently to wrong answers — CoT can be *worse* than direct prompting.

**Direct consequence for this project:** locally we can fit only a 1.5B model, which is
an order of magnitude below the threshold. This is why the local path uses a mock backend
for logic validation and defers quality measurement to the HPC with an 8B+ model. It is a
documented property of the method, not a defect in the implementation.

### 3.5 CoT-SC (Self-Consistency)

> Wang et al., ICLR 2023. Not one of our four PDFs, but the baseline both ToT and GoT
> measure against.

Sample $k$ chains independently and take the modal answer:

$$
\left[z^{(i)}_{1\ldots n}, y^{(i)}\right] \sim p_\theta^{\mathrm{CoT}}(\cdot \mid x),
\quad i=1\ldots k,
\qquad
\hat y \;=\; \arg\max_{y} \; \#\{\, i : y^{(i)} = y \,\}
$$

**Why it helps:** many reasoning paths reach one correct answer, but errors are
idiosyncratic. Correct answers concentrate; wrong answers disperse. If each chain is
correct with probability $p$ and errors are i.i.d. across a large answer space, the mode
is correct with probability approaching 1 as $k$ grows.

**Its two limitations**, both named by ToT:

1. **No local exploration** — within a chain there is still no branching.
2. **Voting requires a small answer space.** If $y$ is a 64-element list, no two samples
   will ever be identical, so $\#\{i : y^{(i)} = y\} = 1$ for every sample and the mode is
   meaningless.

Limitation 2 is decisive for sorting, which is why our `cot_sc` baseline selects by
*score* rather than by majority — the only sensible adaptation to a large output space.

---

## 4. Paper 2 — Tree of Thoughts

> Yao et al., NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)

### 4.1 Framing

The paper opens with dual-process theory: **System 1** (fast, automatic, associative)
versus **System 2** (slow, deliberate, planning). Autoregressive generation is System 1;
ToT bolts on a System 2.

Formally it adopts Newell & Simon's problem-space model: search a tree whose nodes are
partial solutions and whose edges are operators.

### 4.2 The four design questions

A ToT instantiation is the tuple $(G, V, \text{search})$ answering four questions. GoT
inherits and extends this structure, so it is worth stating precisely.

**(1) Thought decomposition.** A state is $s = [x, z_{1\ldots i}]$. Thought granularity is
a genuine trade-off:

$$
\underbrace{\text{too small}}_{\text{unevaluable}} \;\ll\; |z_i| \;\ll\; \underbrace{\text{too large}}_{\text{ungeneratable}}
$$

| Task | One thought is |
|---|---|
| Game of 24 | one equation, `13 - 9 = 4 (left: 4, 4, 10)` |
| Creative Writing | a paragraph-level plan |
| Crosswords | one word |

**(2) Thought generator $G(p_\theta, s, k)$.** Two strategies:

$$
\text{(a) i.i.d. sampling:}\quad z^{(j)} \sim p_\theta^{\mathrm{CoT}}(z_{i+1} \mid s),\quad j=1\ldots k
$$
$$
\text{(b) propose-all:}\quad \left[z^{(1)},\ldots,z^{(k)}\right] \sim p_\theta^{\mathrm{propose}}\!\left(z^{(1\ldots k)}_{i+1} \mid s\right)
$$

(a) suits rich thought spaces where independent samples are naturally diverse; (b) suits
constrained spaces, because conditioning on the other candidates prevents duplicates.
Our `Generate(branching_factor=k)` implements (a), correct for sorting where each
candidate is a long structured object.

**(3) State evaluator $V(p_\theta, S)$.** ToT's cleverest move. Classical search needs a
heuristic; those are normally hand-programmed (Deep Blue) or learned (AlphaGo). ToT
proposes a third: **ask the LLM**.

$$
\text{(a) value:}\quad V(p_\theta,S)(s) \sim p_\theta^{\mathrm{value}}(v \mid s)
\qquad
\text{(b) vote:}\quad V(p_\theta,S)(s) = \mathbb{1}[s = s^*],\; s^* \sim p_\theta^{\mathrm{vote}}(s^* \mid S)
$$

The licensing observation:

> "Such valuations do not need to be perfect, and only need to be **approximately helpful
> for decision making**."

**(4) Search.** BFS with beam width $b$ (keep best $b$ per level), or DFS with pruning
threshold $v_{th}$ and backtracking.

$$
S_t \;=\; \arg\max_{S \subset S'_t,\; |S|=b} \; \sum_{s \in S} V_t(s)
$$

### 4.3 Results

| Method | Game of 24 |
|---|---|
| GPT-4 + IO | 7.3% |
| GPT-4 + CoT | 4.0% |
| GPT-4 + CoT-SC ($k$=100) | 9.0% |
| **GPT-4 + ToT ($b$=5)** | **74%** |

An 18× improvement over CoT.

### 4.4 The structural limitation

A tree is precisely a connected acyclic graph in which

$$
\deg^-(v) \le 1 \quad \text{for all } v \in V
$$

Three consequences, all fatal for the problems GoT targets:

1. **Merging is inexpressible.** An operation with $k$ inputs and 1 output requires
   $\deg^-(v) = k > 1$. Not a tree node. There is no workaround.
2. **Discarded branches are lost.** Pruning at a node removes its entire subtree from
   the information available to the answer.
3. **Refinement loops are inexpressible.** Trees are acyclic, so $(v,v) \notin E$.

Section 9 below shows this is not merely an expressiveness inconvenience — it costs ToT a
factor of $N/\log_k N$ in *volume*.

---

## 5. Paper 3 — Multimodal Chain-of-Thought

> Zhang et al., TMLR 2024. [arXiv:2302.00923](https://arxiv.org/abs/2302.00923)

### 5.1 Why it is in the set

This paper is **not** on the CoT → ToT → GoT structural axis. It extends the *input
modality*, not the reasoning topology. I read it to establish the boundary of the family,
and it contains one lesson that transfers directly.

### 5.2 The two-stage factorisation

Standard one-stage CoT with vision would sample rationale and answer jointly. MM-CoT
factors them and conditions the second on the first:

$$
\text{Stage 1:}\quad R \sim p_\theta\!\left(R \mid Q, C, I\right)
$$
$$
\text{Stage 2:}\quad A \sim p_\theta\!\left(A \mid Q, C, I, R\right)
$$

where $Q$ = question, $C$ = text context, $I$ = image features, $R$ = rationale,
$A$ = answer. Both stages are fine-tuned (T5-based) with vision features fused in — a real
difference from the other three papers, which are all training-free.

### 5.3 The finding that matters

The authors first tried one-stage and found that models under 1B parameters generate
**hallucinated rationales** — plausible but false reasoning — and the answer stage then
faithfully follows them to a wrong answer. *Providing a rationale made accuracy worse.*
Grounding the rationale in actual image features fixed it; their <1B model then beat
GPT-3.5 on ScienceQA.

### 5.4 The transferable lesson

> **A wrong reasoning step is worse than no reasoning step, because downstream steps
> trust it.**

Write $P(\text{correct})$ for a two-stage pipeline:

$$
P(A \text{ correct}) = P(A \mid R\ \text{good})\,P(R\ \text{good}) + P(A \mid R\ \text{bad})\,P(R\ \text{bad})
$$

When $P(A \mid R\ \text{bad}) \ll P(A \mid \text{no } R)$ — i.e. bad reasoning actively
misleads — the pipeline is worse than no reasoning at all unless $P(R\ \text{good})$ is
high.

**This is why GoT scores and ranks before aggregating.** In a graph a bad thought does not
merely produce one bad answer; it propagates into *every* aggregation that consumes it.
Hence in our implementation a `KeepBest` sits between every `Generate` and the
`Aggregate` that follows.

---

## 6. Paper 4 — Graph of Thoughts

> Besta et al., AAAI 2024. [arXiv:2308.09687](https://arxiv.org/abs/2308.09687)

### 6.1 Formal definition

GoT is the tuple $(G, \mathcal{T}, \mathcal{E}, \mathcal{R})$:

| Symbol | Name | Type |
|---|---|---|
| $G = (V, E)$ | reasoning process | directed graph, $E \subseteq V \times V$ |
| $\mathcal{T}$ | transformations | $\mathcal{T}(G, p_\theta) \to G'$ |
| $\mathcal{E}$ | evaluator | $\mathcal{E}(v, G, p_\theta) \to \mathbb{R}$ |
| $\mathcal{R}$ | ranking | $\mathcal{R}(G, p_\theta, h) \to V^h$ |

A vertex holds a solution (initial, intermediate or final); its form is task-dependent.
An edge $(t_1,t_2)$ means $t_2$ was constructed **using $t_1$ as direct input**.

> (The paper overloads $E$ for both the edge set and the evaluator. Context disambiguates;
> I flag it because it is genuinely confusing on first reading.)

A transformation is a pair of added and removed sets:

$$
\mathcal{T}(G, p_\theta) = (V^+, V^-, E^+, E^-),
\qquad
G' = \bigl((V \cup V^+)\setminus V^-,\; (E \cup E^+)\setminus E^-\bigr)
$$

### 6.2 The three transformations

**Generation** — $1 \to k$:

$$
V^+ = \{v_1^+,\ldots,v_k^+\},\qquad E^+ = \{(v,v_1^+),\ldots,(v,v_k^+)\}
$$

```
        v
      / | \
    v₁⁺ v₂⁺ v₃⁺
```

Subsumes CoT-SC and ToT branching. **Nothing new.**

**Aggregation** — $k \to 1$. ★ The key one ★

$$
V^+ = \{v^+\},\qquad E^+ = \{(v_1,v^+),\ldots,(v_k,v^+)\}
$$

```
    v₁   v₂   v₃   v₄
      \   \   /   /
          v⁺          ← deg⁻(v⁺) = 4
```

Since $\deg^-(v^+) = k$, and a tree requires $\deg^-\le 1$:

$$
k > 1 \;\Longrightarrow\; G \text{ is not a tree.}
$$

**This is a theorem, not a preference.** The moment you want to merge, you need a graph.

**Refinement** — self-loop:

$$
V^+ = \varnothing,\qquad E^+ = \{(v,v)\}
$$

Also impossible in a tree, which is acyclic.

### 6.3 Scoring and ranking

$$
\mathcal{E}(v, G, p_\theta) \in \mathbb{R},
\qquad
\mathcal{R}(G,p_\theta,h) = \operatorname*{arg\,top-}_{v \in V}{}^{h}\; \mathcal{E}(v,G,p_\theta)
$$

Note $\mathcal{E}$ takes the **whole graph** $G$, not just $v$ — "scores may be relative
to other thoughts."

Crucially, scoring may be **local and exact** where the task permits:

> "use cases such as sorting use **simple local scoring functions**."

For sorting the score is computable in Python: free, exact, zero-variance. This is a
large practical advantage over ToT, which relies on noisy LLM-based state evaluation, and
on HPC it is also a large *cost* advantage (§11).

### 6.4 Architecture — GoO vs GRS

```
 ┌────────────────────────────────────────────────────────────┐
 │  CONTROLLER                                                │
 │   ┌──────────────────────┐   ┌────────────────────────┐    │
 │   │ GoO                  │   │ GRS                    │    │
 │   │ Graph of Operations  │   │ Graph Reasoning State  │    │
 │   │ ── STATIC ──         │   │ ── DYNAMIC ──          │    │
 │   │ the execution plan,  │   │ the thoughts produced, │    │
 │   │ built once upfront   │   │ updated as it runs     │    │
 │   └──────────────────────┘   └────────────────────────┘    │
 └───────┬──────────────┬──────────────┬─────────────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌─────▼──────┐
    │ Prompter│   │  Parser   │  │  Scoring & │
    │ builds  │   │ extracts  │  │ Validation │
    │ prompts │   │ state     │  │            │
    └────┬────┘   └─────▲─────┘  └────────────┘
         │              │
         └──────► LLM ──┘
```

The distinction that drove my implementation:

- **GoO is the plan.** A DAG of *operations*. Static, built before execution.
- **GRS is the state.** A DAG of *thoughts*, with scores and validity. Dynamic.

One GoO shape executed on 100 inputs produces 100 different GRSs.

### 6.5 Use case: Sorting (§5.1)

Sort digits 0–9 **with duplicates**. The paper's diagnosis of why LLMs fail is precise:

> "unable to sort a sequence of such numbers correctly beyond a certain length
> consistently **because duplicate counts do not match**."

The decomposition is merge sort. With $m$ chunks the recursion is

$$
T(n) \;=\; \underbrace{m \cdot \text{sort}(n/m)}_{\text{chunk sorts}} \;+\; \underbrace{\sum_{\ell=1}^{\log_2 m} \tfrac{m}{2^\ell}\cdot\text{merge}\!\left(\tfrac{2^\ell n}{m}\right)}_{\text{merge tree}}
$$

**The GoO (paper Figure 4), $n=64$, $m=4$:**

```
[64 numbers]
     │  split → 4 chunks of 16        (local, no LLM)
     ├──────────┬──────────┬──────────┐
   chunk0     chunk1     chunk2     chunk3
     │ k=3       │ k=3      │ k=3      │ k=3
   Score      Score      Score      Score       (exact, free)
  KeepBest   KeepBest   KeepBest   KeepBest     (N=1)
     └────┬─────┘          └────┬─────┘
      Aggregate kₐ=10       Aggregate kₐ=10
        Score                 Score
       KeepBest              KeepBest
          └──────────┬──────────┘
              Aggregate kₐ=10
                 Score → KeepBest
                   │
              [64 sorted]
```

Note the asymmetry: $k=3$ for chunk sorting, $k_a=10$ for merging. Merging gets more
attempts because that is where global correctness is decided — a merge error corrupts
everything, while a chunk error is contained. §11 shows this is also the dominant cost
term.

### 6.6 Use case: Set Intersection (§5.2)

Split $B$ only; intersect each piece against all of $A$; union the results. Valid because
intersection distributes over union:

$$
A \cap \left(\bigcup_{i=1}^{m} B_i\right) \;=\; \bigcup_{i=1}^{m} \left(A \cap B_i\right)
$$

So the aggregation is an **exact union**. This illustrates the paper's general point
nicely: *the right graph decomposition follows from the algebraic structure of the
problem*, and GoT supplies the vocabulary for whatever that structure turns out to be.

### 6.7 Use cases: Keyword Counting (§5.3) and Document Merging (§5.4)

**Keyword counting** splits text into passages and sums sub-counts — aggregation is
addition, again exact:

$$
\mathrm{count}(w, T) = \sum_{j=1}^{m} \mathrm{count}(w, T_j)
$$

**Document merging** is the one task with no exact scorer, so the LLM judges. Query
redundancy $r$ and retention $i$, three times each, average, then combine by **harmonic
mean**:

$$
\text{score} = \frac{2ri}{r+i}
$$

The harmonic mean is chosen because it is dominated by the smaller argument: you cannot
win by deleting everything ($r=10$, $i=0 \Rightarrow 0$) or by concatenating everything
($i=10$, $r=0 \Rightarrow 0$). The arithmetic mean would reward both degenerate
strategies with 5.

### 6.8 Headline result

> "increasing the quality of sorting by **62% over ToT**, while simultaneously reducing
> costs by **>31%**."

---

## 7. Why decomposition works — the error model

The papers assert that decomposition helps. Here is the argument made quantitative.

### 7.1 The model

Let $p(n)$ be the probability the model handles a length-$n$ instance correctly. Empirically
accuracy decays roughly geometrically in the number of items that must be tracked
simultaneously, so model it as

$$
p(n) \;=\; e^{-\lambda n}
$$

with $\lambda > 0$ a model-quality constant (smaller $\lambda$ = stronger model). This
captures the paper's own observation that failures set in "beyond a certain length".

### 7.2 Monolithic versus decomposed

**IO (monolithic):**

$$
P_{\mathrm{IO}}(n) \;=\; p(n) \;=\; e^{-\lambda n}
$$

**ToT (best-of-$k$, still monolithic):** with an exact scorer, success needs at least one
good sample among $k$:

$$
P_{\mathrm{ToT}}(n) \;=\; 1 - \bigl(1 - e^{-\lambda n}\bigr)^{k}
$$

**GoT (decomposed).** Write $q_k(p) = 1-(1-p)^k$. All $m$ chunks must succeed, then every
merge must succeed:

$$
P_{\mathrm{GoT}}(n) \;=\; \underbrace{\Bigl[q_k\!\left(e^{-\lambda n/m}\right)\Bigr]^{m}}_{\text{chunks}}
\cdot
\underbrace{\prod_{\ell=1}^{\log_2 m} \Bigl[q_{k_a}\!\left(e^{-\lambda_{\mathrm{m}} 2^{\ell} n/m}\right)\Bigr]^{m/2^{\ell}}}_{\text{merge tree}}
$$

where $\lambda_{\mathrm{m}} < \lambda$ because merging two *already sorted* lists is an
easier operation than sorting from scratch — the model mostly interleaves.

### 7.3 The key inequality

The decisive structural fact is that the exponential is evaluated at $n/m$, not $n$:

$$
p(n/m) = e^{-\lambda n/m} = \bigl[p(n)\bigr]^{1/m} \;\gg\; p(n)
\qquad\text{for } m>1,\ \lambda n \gg 1
$$

For $\lambda = 0.035$, $n = 64$, $m = 4$:

$$
p(64) = e^{-2.24} \approx 0.106,
\qquad
p(16) = e^{-0.56} \approx 0.571
$$

A single chunk is **5.4× more likely** to be handled correctly than the whole input. With
$k=3$ best-of-3 on each chunk, $q_3(0.571) = 1-0.429^3 \approx 0.921$, and all four chunks
succeed with $0.921^4 \approx 0.72$ — versus $0.106$ monolithically.

![Decomposition maths](docs/figures/theory_decomposition.png)

*Left: success probability against input length for the three schemes, at
$\lambda=0.035$, $m=4$, $k=3$. Right: the ratio $P_{\mathrm{GoT}}/P_{\mathrm{IO}}$, which
grows with $n$ — decomposition matters more the harder the instance. Generated by
`scripts/make_theory_figures.py`.*

### 7.4 Why there is an optimum $m$

More chunks means easier sub-problems but more merges, and each merge is a fresh chance to
fail. Taking logs of the chunk term,

$$
\log P_{\text{chunks}} = m \log q_k\!\left(e^{-\lambda n/m}\right)
$$

increases with $m$, while the merge term contributes $m-1$ failure opportunities and
decreases with $m$. The product has an interior maximum. The paper's choice of $m=4$ for
$n=64$ (chunks of 16) sits near it for GPT-3.5-class models; **the optimum shifts with
model strength**, so a weaker open model may prefer larger $m$ (smaller chunks). This is a
concrete, cheap experiment to run on the HPC, and one of the more interesting things this
codebase can measure.

---

## 8. Best-of-$k$ and the role of exact scoring

### 8.1 With a perfect scorer

`Generate(k)` followed by `KeepBest(1)` succeeds iff at least one of $k$ i.i.d. samples is
correct:

$$
q_k(p) \;=\; 1 - (1-p)^k
$$

The marginal value of the $k$-th sample is

$$
\frac{\partial q_k}{\partial k} \;=\; -(1-p)^k \ln(1-p) \;=\; \Theta\!\left((1-p)^k\right)
$$

— **geometric decay**. Cost, meanwhile, is exactly linear in $k$. So the return per token
falls off fast, which justifies the paper's modest $k=3$ on chunks and makes $k$ the first
knob to turn down when the budget bites.

![Best-of-k order statistics](docs/figures/theory_best_of_k.png)

*Left: $q_k(p)$ for several $k$. Right: marginal gain of the $k$-th sample — geometric
decay against linear cost.*

### 8.2 With an imperfect scorer

This is where exact local scoring earns its place. Suppose the scorer picks the truly-best
candidate only with probability $\sigma$ (and otherwise picks at random). Then roughly

$$
P(\text{success}) \;\approx\; \sigma\, q_k(p) \;+\; (1-\sigma)\, p
$$

As $\sigma \to 1$ we recover $q_k(p)$; as $\sigma \to 1/k$ (random choice) the benefit of
generating $k$ candidates evaporates entirely. **Generating candidates is only worth
paying for if you can tell which one is good.**

For sorting, $\sigma = 1$ exactly, because the error-scope function is computable in
closed form. This is the single largest reason the sorting pipeline works well, and it is
why §14 treats "prefer local exact scoring" as a design rule rather than an optimisation.

---

## 9. The latency–volume theorem, with proof

Section 6 of the paper is its theoretical contribution. I derive it here rather than
quoting it.

### 9.1 Definitions

For a thought $t$ in reasoning graph $G$:

$$
L(t) \;=\; \min_{\,s \in \mathrm{Sources}(G)} \operatorname{dist}(s, t)
\qquad\text{(latency: shortest path from an input)}
$$

$$
V(t) \;=\; \bigl|\{\, u \in V \;:\; \exists \text{ a path } u \rightsquigarrow t \,\}\bigr|
\qquad\text{(volume: ancestors)}
$$

Latency answers *how long did I wait*; volume answers *how much accumulated reasoning
informs this answer*. We want high $V$ at low $L$. Fix total cost at $\Theta(N)$ thoughts
for every scheme, with branching factor $k$.

### 9.2 Scheme by scheme

**CoT — a single chain of $N$ thoughts.**
The final thought sits at the end: $L = N$. Every earlier thought is an ancestor:
$V = N$.

$$
L_{\mathrm{CoT}} = N, \qquad V_{\mathrm{CoT}} = N
$$

**CoT-SC — $k$ disjoint chains from one root, each of length $N/k$.**
Each chain is $k$ times shorter, so $L = N/k$. But the chains never interact: a thought's
ancestors are only its own chain, so $V = N/k$.

$$
L_{\mathrm{CoT\text{-}SC}} = N/k, \qquad V_{\mathrm{CoT\text{-}SC}} = N/k
$$

Splitting divides latency by $k$ — *and divides volume by $k$ too*. Speed was bought by
discarding information.

**ToT — a complete $k$-ary tree with $N$ nodes.**
Depth is $\log_k N$, so $L = \log_k N$. Now the crucial step: consider a **leaf** $t$. In a
tree every node has in-degree $\le 1$, so the ancestor set of $t$ is exactly the unique
path from the root to $t$:

$$
V_{\mathrm{ToT}}(t) \;=\; \bigl|\mathrm{path}(\mathrm{root} \to t)\bigr| \;=\; \Theta(\log_k N)
$$

$$
L_{\mathrm{ToT}} = \log_k N, \qquad V_{\mathrm{ToT}} = O(\log_k N)
$$

**This is the punchline about trees.** A tree with $N$ nodes gives its leaf access to only
$\log_k N$ of them. The other $N - \log_k N$ thoughts — nearly all the work — have **no
path** to the answer and therefore cannot influence it. ToT explores a great deal and then
throws almost all of it away.

**GoT — a $k$-ary tree joined at its leaves to a mirrored $k$-ary tree with reversed
edges.**

```
       INPUT              ← fan out (tree half): depth log_k N
        /|\
       ● ● ●
      /|\ /|\
     ● ● ● ● ●            ← N leaves explored in parallel
      \|/ \|/
       ● ● ●              ← fan back IN: every node here is an AGGREGATION
        \|/
       OUTPUT
```

*Latency.* Both halves have depth $\log_k N$, so $L = 2\log_k N = \Theta(\log_k N)$.

*Volume.* Take the output $t^*$. Let $u$ be any thought in the graph. In the fan-out half,
$u$ lies on a root-to-leaf path, so there is a path $u \rightsquigarrow \ell$ to some leaf
$\ell$. In the mirrored half, every leaf has a path $\ell \rightsquigarrow t^*$, because
the mirror's edges are reversed and its aggregations converge on $t^*$. Concatenating,
$u \rightsquigarrow t^*$ exists for **every** $u$. Hence

$$
V_{\mathrm{GoT}}(t^*) \;=\; N
$$

$$
\boxed{\;L_{\mathrm{GoT}} = \Theta(\log_k N), \qquad V_{\mathrm{GoT}} = N\;}
$$

### 9.3 Table 2

| Scheme | Latency | Volume | $V/L$ |
|---|---|---|---|
| CoT | $N$ | $N$ | $1$ |
| CoT-SC | $N/k$ | $N/k$ | $1$ |
| ToT | $\log_k N$ | $O(\log_k N)$ | $\Theta(1)$ |
| **GoT** | $\log_k N$ | $N$ | $\Theta(N/\log_k N)$ |

![Volume and latency](docs/figures/theory_volume_latency.png)

*Table 2 plotted as continuous functions of $N$ at $k=4$. GoT is the only scheme whose
volume-per-latency ratio grows with the budget.*

**The one-sentence summary:** *aggregation is what lets information flow back together
after it fans out; a tree can only fan out.*

### 9.4 Empirical verification

`got/metrics.py` computes $V$ and $L$ by BFS over the graphs our runs actually build. On a
32-element sorting instance:

| Scheme | thoughts | aggregations | $V$ | $L$ |
|---|---|---|---|---|
| IO | 2 | 0 | 1 | 1 |
| CoT | 3 | 0 | 2 | 2 |
| CoT-SC | 7 | 0 | 2 | 2 |
| ToT | 9 | 0 | **6** | **6** |
| **GoT** | 39 | **15** | **18** | **7** |

GoT achieves **3× ToT's volume for one extra hop**, and is the only scheme with non-zero
aggregation count. The theory is reproduced, not merely cited.

![GoT reasoning graph](docs/figures/graph_got.png)

*An actual GoT reasoning graph from this implementation. Red vertices have
$\deg^- > 1$ — the aggregations. The diamond shape is the fan-out/fan-in construction of
§9.2 made concrete.*

---

## 10. The scoring functions — formal properties

### 10.1 Sorting (§5.1)

For input $a = [a_1 \ldots a_n]$ and output $b = [b_1 \ldots b_m]$:

$$
\mathrm{error\text{-}scope}(a,b) \;=\; X + Y
$$

$$
X \;=\; \sum_{i=1}^{m-1} \operatorname{sgn}\!\bigl(\max(b_i - b_{i+1},\,0)\bigr),
\qquad
Y \;=\; \sum_{d=0}^{9} \Bigl|\; \bigl|\{b_p : b_p = d\}\bigr| - \bigl|\{a_q : a_q = d\}\bigr| \;\Bigr|
$$

**Reading $X$.** $\max(b_i-b_{i+1},0)$ is positive iff $b_i > b_{i+1}$, and $\operatorname{sgn}$
collapses it to $1$. So

$$
X \;=\; \bigl|\{\, i : b_i > b_{i+1} \,\}\bigr|
$$

the count of **adjacent descents** — not total inversions. A list sorted except for one
swapped neighbouring pair scores $X = 1$, whereas its inversion count would also be 1;
but a reversed list scores $X = m-1$ where the inversion count is $\binom{m}{2}$. $X$ is
therefore a *local* disorder measure, bounded by $m-1$.

**Reading $Y$.** Writing $\mu_a, \mu_b$ for the multiset multiplicity functions,

$$
Y \;=\; \bigl\| \mu_b - \mu_a \bigr\|_1
$$

the $\ell_1$ distance between count vectors. This is exactly the failure the paper
diagnosed: dropped elements and hallucinated duplicates.

**Proposition.** $X + Y = 0 \iff b = \mathrm{sorted}(a)$.

*Proof.* ($\Leftarrow$) If $b$ is $a$ sorted, it is non-decreasing so $X=0$, and it is a
permutation of $a$ so $\mu_b = \mu_a$ and $Y=0$.
($\Rightarrow$) $Y = 0$ gives $\mu_b = \mu_a$, so $b$ is a permutation of $a$ (in
particular $m = n$). $X = 0$ gives $b_i \le b_{i+1}$ for all $i$, so $b$ is non-decreasing.
A non-decreasing permutation of $a$ is unique and equals $\mathrm{sorted}(a)$. $\blacksquare$

**Why both terms are necessary.** Neither alone is sound:

| Failure | $X$ | $Y$ | Caught by |
|---|---|---|---|
| $b$ ascending but 3 elements dropped | 0 | 3 | $Y$ only |
| $b$ has right multiset, wrong order | $>0$ | 0 | $X$ only |

**Sign convention.** Our framework ranks by descending score, so we use the paper's
positive form:

$$
\mathrm{score}(b) \;=\; \max\bigl(n - \mathrm{error\text{-}scope}(a,b),\; 0\bigr)
$$

The paper additionally clips with $\min(\mathrm{error\text{-}scope}, n)$ **for plotting
only** ("to improve the clarity of plots, as some baselines result in large numbers of
outliers"). We keep clipping optional and off by default for analysis.

### 10.2 Set intersection (§5.2)

For inputs $A, B$ and output $C$ (a *list*, possibly with repeats):

$$
\mathrm{error\text{-}scope} \;=\; X_1 + X_2 + X_d
$$

$$
X_1 = \bigl|\,\mathrm{set}(C) \setminus (A \cap B)\,\bigr|,
\qquad
X_2 = \bigl|\,(A \cap B) \setminus \mathrm{set}(C)\,\bigr|,
\qquad
X_d = |C| - |\mathrm{set}(C)|
$$

$X_1 + X_2$ is the **symmetric difference** $\bigl|\mathrm{set}(C) \,\triangle\, (A\cap B)\bigr|$,
a genuine metric on sets. $X_d$ counts duplicates, needed "because the LLM expresses the
set as a list in natural language" — a list can repeat where a set cannot.

By the same argument as above, $X_1+X_2+X_d = 0$ iff $C$ is exactly $A \cap B$ with no
repeats.

---

## 11. Cost model

On a cluster the budget is GPU-seconds, and GPU-seconds are dominated by **decoded
tokens** (prefill is one parallel pass; decode is a sequential loop).

### 11.1 Token cost per scheme

For sorting, an answer listing $j$ digits costs $\Theta(j)$ tokens. Writing $c$ for tokens
per element, $m$ chunks, branching $k$, aggregation attempts $k_a$, ToT depth $d$:

$$
\begin{aligned}
C_{\mathrm{IO}} &= c\,n \\[2pt]
C_{\mathrm{CoT}} &= 2\,c\,n \\[2pt]
C_{\mathrm{CoT\text{-}SC}} &= k\,c\,n \\[2pt]
C_{\mathrm{ToT}} &= k\,c\,n + (d-1)\,c\,n \\[2pt]
C_{\mathrm{GoT}} &= \underbrace{m \cdot k \cdot c\,\tfrac{n}{m}}_{\text{chunk sorts}}
 \;+\; \underbrace{k_a \sum_{\ell=1}^{\log_2 m} \frac{m}{2^{\ell}} \cdot c\,\frac{2^{\ell} n}{m}}_{\text{merge tree}}
 \;=\; k\,c\,n \;+\; k_a\, c\, n \log_2 m
\end{aligned}
$$

The merge sum is worth noting: at level $\ell$ there are $m/2^\ell$ merges each producing
$2^\ell n/m$ elements, so **every level costs the same $c\,n$** — a classic merge-sort
property. Hence the clean $\log_2 m$ factor.

$$
\frac{C_{\mathrm{GoT}}}{C_{\mathrm{IO}}} \;=\; k + k_a \log_2 m
$$

With the paper's $k=3$, $k_a=10$, $m=4$: a factor of $3 + 20 = 23$. **The $k_a \log_2 m$
term dominates**, which identifies `aggregation_attempts` as the first knob to turn.

![Cost model](docs/figures/theory_cost_quality.png)

*Left: decode cost against input length. Right: relative cost against $k_a$ at $n=64$ —
the dominant GoT cost knob.*

### 11.2 Throughput, and why batching matters

Wall-clock GPU time is

$$
T_{\mathrm{GPU}} \;\approx\; \frac{C_{\text{decode}}}{\tau(\beta)} \;+\; \frac{C_{\text{prefill}}}{\tau_{\text{pre}}}
$$

where $\tau(\beta)$ is throughput at batch size $\beta$. The critical fact:
$\tau$ **increases steeply with $\beta$** until the GPU saturates, because decoding is
memory-bandwidth-bound and a larger batch amortises each weight read across more
sequences. Serving prompts one at a time leaves a large GPU mostly idle.

This is why the implementation batches (see §14.8): one `Generate` holding four chunk
thoughts submits $4k$ sequences in a single call instead of four calls of $k$.

Prefix caching gives a second saving. All prompts for a task share a long identical
prefix $P$ (instructions + few-shot example), so with caching

$$
C_{\text{prefill}} \;=\; |P| + \sum_i |u_i| \quad\text{instead of}\quad \sum_i \bigl(|P| + |u_i|\bigr)
$$

where $u_i$ is the instance-specific tail. With $|P| \gg |u_i|$ this is close to an
$N$-fold reduction in prefill.

### 11.3 Pricing a job before submitting it

`scripts/estimate_cost.py` runs the full GoO on the free mock backend — identical control
flow, identical prompt counts, no GPU — and projects tokens, batches and GPU-seconds. The
*counts are exact*; only $\tau$ is modelled. Example for 100 instances of `sorting_64`
on an 8B model:

| scheme | $k_a$ | batches | sequences | decode tokens | GPU time |
|---|---|---|---|---|---|
| io | – | 100 | 100 | 7,200 | 0.1 min |
| cot | – | 200 | 200 | 14,400 | 0.1 min |
| cot_sc | – | 100 | 300 | 21,600 | 0.2 min |
| tot | – | 300 | 500 | 36,000 | 0.3 min |
| **got** | **10** | 300 | 4,200 | 192,960 | **1.5 min** |
| got | 5 | 300 | 2,700 | 113,760 | 0.9 min |
| got | 3 | 300 | 2,100 | 82,080 | 0.6 min |

The whole comparison is single-digit minutes of compute. On a cluster the *allocation*
(model load, queue, node hold) will dominate the bill, not the inference — so the
practical advice is to batch many configurations into one job rather than submitting many
short ones.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Aggregation** | GoT transformation merging $k$ thoughts into 1; creates $\deg^-(v)>1$. Impossible in a tree. |
| **Backtracking** | Abandoning an unpromising branch; introduced by ToT. |
| **Beam search** | Keeping best $b$ candidates per level; ToT's BFS. |
| **Branching factor $k$** | Candidates generated per thought. |
| **CoT** | Chain of Thought: intermediate reasoning steps in the prompt. |
| **CoT-SC** | Self-Consistency: $k$ chains, majority vote. |
| **DAG** | Directed Acyclic Graph — the shape of a GoT reasoning graph. |
| **Emergent ability** | Capability absent below a scale threshold; CoT needs $\gtrsim$10B params. |
| **Error scope** | Task-specific error metric; $X+Y$ for sorting. |
| **Few-shot / ICL** | In-context learning: examples in the prompt, no weight updates. |
| **GoO** | Graph of Operations — the **static** execution plan. |
| **GRS** | Graph Reasoning State — the **dynamic** thoughts produced. |
| **Hallucinated rationale** | Plausible but wrong reasoning; dangerous because later steps trust it. |
| **In-degree $\deg^-$** | Incoming edges. $\deg^->1 \iff$ aggregation $\iff$ genuinely a graph. |
| **Latency $L(t)$** | Shortest-path hops from input to $t$. |
| **Multiset** | Set allowing duplicates; sorting must preserve it (term $Y$). |
| **Parser** | Extracts structured thought state from raw LLM text. |
| **Prefill / decode** | Parallel pass over the prompt vs. sequential token generation. Decode dominates cost. |
| **Prefix caching** | Reusing the KV cache of a shared prompt prefix. |
| **Prompter** | Builds prompt strings; owns task-specific wording. |
| **Refinement** | Improving a thought in place; formally a self-loop $(v,v)$. |
| **Score $\mathcal{E}$** | $\mathcal{E}(v,G,p_\theta)\to\mathbb{R}$; exact Python (sorting) or LLM query (merging). |
| **Ranking $\mathcal{R}$** | Top-$h$ thoughts by score. |
| **State (ToT)** | $s=[x,z_{1\ldots i}]$ — a partial solution. |
| **System 1 / 2** | Fast-automatic vs slow-deliberate cognition; ToT's framing. |
| **Thought** | A coherent unit of intermediate reasoning; a vertex in $G$. |
| **Volume $V(t)$** | Number of thoughts with a path *to* $t$. |
| **$X$ (sorting)** | Count of adjacent descents — sortedness. |
| **$Y$ (sorting)** | $\ell_1$ distance between multiset count vectors. |

---

## 13. Paper → code map

| Concept | Section | Implementation |
|---|---|---|
| Thought (vertex) | GoT 3.1 | [`got/thought.py`](got/thought.py) — `Thought` |
| Edge (dependency) | GoT 3.1 | `Thought.add_predecessor()` |
| Generation | GoT 3.2 | [`got/operations.py`](got/operations.py) — `Generate` |
| **Aggregation** | GoT 3.2 | **`Aggregate`, `PairwiseAggregate`** |
| Refinement | GoT 3.2 | `Improve` |
| Evaluator $\mathcal{E}$ | GoT 3.3 | `Score` |
| Ranking $\mathcal{R}$ | GoT 3.3 | `KeepBest`, `KeepBestPerGroup` |
| Prompter | GoT 4.1 | [`got/prompter.py`](got/prompter.py) |
| Parser | GoT 4.2 | [`got/prompter.py`](got/prompter.py) |
| Scoring & Validation | GoT 4.3 | `Score` + `KeepValid` + `state["valid"]` |
| Controller | GoT 4.4 | [`got/controller.py`](got/controller.py) |
| **GoO** (static) | GoT 4.5 | the `Operation` DAG in `got/tasks/*/graphs.py` |
| **GRS** (dynamic) | GoT 4.5 | `Operation.thoughts`, `Controller.all_thoughts()` |
| Sorting score $X+Y$ | GoT 5.1 | [`got/tasks/sorting/scoring.py`](got/tasks/sorting/scoring.py) |
| Sorting GoO (Fig. 4) | GoT 5.1 | `got_sorting_goo` |
| Set intersection | GoT 5.2 | [`got/tasks/set_intersection/`](got/tasks/set_intersection/) |
| Latency & Volume | GoT 6 | [`got/metrics.py`](got/metrics.py) |
| Table 2 | GoT 6 | `metrics.theoretical_bounds()` |
| ToT BFS/beam | ToT Alg. 1 | `tot_goo()` |
| CoT-SC | CoT-SC | `cot_sc_goo()` |
| Cost model (§11) | — | [`scripts/estimate_cost.py`](scripts/estimate_cost.py) |

---

## 14. Implementation decisions

### 14.1 Three interchangeable backends

Hardware: 7.3 GB RAM (3.5 GB in WSL), 8 cores, **AMD integrated GPU — no CUDA**. A 7B
model in fp16 needs $\approx 2 \times 7\times10^9 = 14$ GB. It cannot run here.

| Backend | Where | Model | Purpose |
|---|---|---|---|
| `MockLM` | laptop | none | validate graph logic, free and instant |
| `LlamaCppLM` | laptop | Qwen2.5-1.5B Q4 (~1 GB) | real-but-small end-to-end check |
| `HFLM`/`VLLMLM` | HPC | Llama-3.1-8B/70B | paper-comparable numbers |

### 14.2 The mock must be fallible, not an oracle

A perfect mock would score 100% for every scheme and prove nothing. It must fail **the way
a real LLM fails**. So `MockLM` corrupts with modes matching the score's penalties (drop
and duplicate break $Y$; neighbour swap breaks $X$), **degrades with length** per the
$e^{-\lambda n}$ model of §7, and treats merging ($\times 0.6$) and refinement
($\times 0.5$) as easier than sorting — which is precisely *why* decomposition helps.

Setting `error_rate=0` gives a perfect oracle, isolating framework bugs from model errors.
There is a test for exactly that.

**Honest limitation:** MockLM validates control flow and bookkeeping only. Mock numbers
are **not** paper-comparable.

### 14.3 Structural steps do not call the LLM

Splitting a 64-element list into 4 chunks is deterministic Python. Spending a call on it
adds cost *and* a failure mode for zero benefit — a model that miscounts while chunking
corrupts $Y$ before reasoning starts. `Prompter.build()` returns `None` to mark a step
local; verified free by `test_split_is_local_and_free`.

### 14.4 Refinement as a fresh vertex, not a literal self-loop

The paper defines refinement as $E^+=\{(v,v)\}$. A literal self-loop makes $G$ cyclic,
breaking topological sort, BFS volume computation, and layout. We materialise the refined
result as a new vertex with the original as predecessor: acyclic, same provenance, same
information content. Bookkeeping, not behaviour.

### 14.5 Scoring separate from validation

A thought can be well-formed but poor (valid, low score) or malformed (invalid — prose
where a list was required). Separating them lets `KeepValid` drop garbage before it
pollutes the ranking. This matters far more with open 1.5B models than with GPT-4.

### 14.6 Defensive parsing

`extract_list()` takes the **last** bracketed list — chatty models restate the input first.
Code fences are stripped. A parse failure yields an invalid thought, never an exception:
one bad response must not abort a multi-hour HPC job.

### 14.7 Five schemes, one engine

`io`, `cot`, `cot_sc`, `tot`, `got` share Controller, backend, prompts and scorer, and
**differ only in graph topology**. Any measured difference is attributable to structure
alone. Tests assert the four baselines have zero aggregations and GoT has some.

### 14.8 Batching, and why the GoO shape had to change

The first version gave each chunk its own `Selector → Generate → Score → KeepBest` chain.
The reasoning *graph* was right, but the cost was not: the Controller runs operations one
at a time, so four single-input `Generate`s meant four serial calls each submitting a
batch of one.

Folding them into a single `Generate` over all chunk thoughts, followed by
`KeepBestPerGroup`, produces **identical thoughts and edges** while collapsing 4 serial
calls into 1 batched call of 4 prompts ($4k$ sequences). Same for `PairwiseAggregate` per
merge level. Measured on 64-element sorting: GoT went from **7 GPU round trips to 3**,
with 12 concurrent sequences in the first batch instead of 3.

Per-step token budgets matter for the same reason. A 16-element chunk answer needs
$\approx 2\times16+32 = 64$ tokens; leaving the global 1024 default in place risks paying
16× for nothing when a model fails to emit a stop token.

---

## 15. Bugs found, and what they taught me

### Bug 1 — few-shot examples leaking into answers

**Symptom:** a 32-element input produced a **205-element** output.
**Cause:** `MockLM` scraped *every* bracketed list from the prompt, including the
16-element few-shot example, and merged them all.
**Fix:** prompts place the real payload last; the mock takes the last $N$ lists.
**Lesson:** mirrors a real failure mode — instruction-tuned models do sometimes copy
exemplars into answers. Taking the **last** match in `extract_list()` guards the real
version.

### Bug 2 — aggregated thoughts scored against the wrong target ★

**Symptom:** GoT scored **0/32** while visibly emitting a near-perfect sorted list.
**Cause:** `sorting_score` compares `current` against `original`, and a merged thought
inherited `original` from **only its first parent** — so a 32-element result was compared
against a 16-element chunk. $Y$ then reported ~16 spurious elements and the score floored.
**Fix:** define `original` recursively:

$$
\mathrm{orig}(v) = \begin{cases}
\text{the chunk} & v \text{ from split} \\
\mathrm{orig}(\mathrm{parent}(v)) & v \text{ from sort/refine} \\
\mathrm{orig}(p_1) \uplus \mathrm{orig}(p_2) & v \text{ from aggregation}
\end{cases}
$$

with $\uplus$ multiset union. At the root of the merge tree this reconstitutes the full
input multiset.

**Lesson — the important one:** in a tree, "what should I contain?" has one answer from one
parent. **In a graph, provenance must be combined across all parents.** Aggregation changes
not just topology but how metadata propagates. This bug class *cannot occur* in CoT or ToT.

### Bug 3 — aggregation silently discarding half its input

**Symptom:** on set intersection GoT scored **worse than IO** (error 13.60 vs 3.90).
**Cause:** the union prompt contains neither "intersect" nor "merge", so `MockLM` fell
through to its generic single-list branch and read only one of the two payload lists.
Across a 3-level tree only $\approx 1/4$ of elements survived.
**Fix:** an explicit union branch, tested before the generic ones.
**Result:** error **3.90 → 1.10 ($-72\%$)**, accuracy **0% → 40%**.
**Lesson:** a silent fallback is worse than a crash. With a *real* model the analogous
failure — misreading an aggregation prompt and echoing one input — would be equally
silent. This is exactly why `Score` sits after every `Aggregate`: an aggregation that
dropped half its input scores badly and gets pruned.

### Bug 4 — grouping metadata lost across the parser (found during the batching refactor)

**Symptom:** after folding the per-chunk chains into one batched `Generate`, GoT returned
a correctly sorted list of only **a quarter** of the input.
**Cause:** `KeepBestPerGroup` ranks within `chunk_index`, but `SortingParser.parse()`
built a fresh state dict and dropped that key. All sorted chunks fell into one group, so
`KeepBestPerGroup(n=1)` kept a single chunk and discarded the other three.
**Fix:** carry grouping keys (`chunk_index`, `_group`) through every parse.
**Lesson:** a cost optimisation that changes *how work is grouped* silently changes
*what identifies a group*. The unit test asserting a perfect oracle reaches
`sorted(numbers)` caught it immediately — which is the argument for having that test at
all.

---

## 16. What I verified, and what I did not

### Verified ✅

- **Formal structures.** Volume, latency, in-degree and the DAG property computed on real
  graphs; 46 tests passing.
- **Table 2 qualitatively.** GoT reaches **3× ToT's volume for one extra hop**; the only
  scheme with aggregations.
- **Scoring formulae.** $X+Y$ and $X_1+X_2+X_d$ implemented verbatim and tested against
  hand-computed cases, including the $X+Y=0 \iff$ correct proposition of §10.1.
- **Relative ordering.** Mock runs give IO 26 < CoT 28 < CoT-SC 29 < ToT 30 < **GoT 31**
  (score out of 32) — the paper's predicted ordering.
- **Aggregation does the work.** Set intersection: $-72\%$ error vs IO; the entire gain
  vanished when aggregation broke (Bug 3).
- **Framework correctness independent of model quality.** With `error_rate=0` the pipeline
  reaches a provably perfect answer.
- **Cost model counts.** Batches and sequence counts from `estimate_cost.py` match what
  the runner actually issues.

### NOT verified ⚠️

- **Absolute quality numbers.** The mock is not a language model. The "62% over ToT" claim
  **cannot** be confirmed without real LLM runs. Mock accuracy is a diagnostic, not a
  result.
- **The cost claim.** The paper reports GoT *reducing* cost >31% vs ToT; our GoT uses more
  tokens than our ToT baseline, because our ToT is deliberately lean ($b=1$, $d=3$) while
  the paper's is much wider. A fair comparison needs the paper's ToT budget.
- **Throughput $\tau$.** Modelled, not measured. Calibrate from a pilot run before trusting
  the money column of §11.3.
- **Prompt quality.** Untested against a real model. Small open models are far more
  format-sensitive than GPT-4; expect prompt iteration on the HPC.
- **Keyword counting and document merging.** Datasets generated and MockLM supports
  keyword counting, but the GoO builders are not yet written.
- **The optimal $m$ (§7.4).** Predicted to shift with model strength; not yet measured.

### Summary

**Established:** a correct, tested, documented implementation of the GoT framework that
provably builds the graph structures the paper describes, reproduces its latency–volume
theorem empirically, prices its own HPC jobs, and runs identically on a laptop and a
cluster with open-source models.

**Remaining:** running it against a real open-weights model at scale. Everything needed is
in place.

---

## References

1. Wei, J. et al. **Chain-of-Thought Prompting Elicits Reasoning in Large Language
   Models.** NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
2. Yao, S. et al. **Tree of Thoughts: Deliberate Problem Solving with Large Language
   Models.** NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) ·
   [code](https://github.com/princeton-nlp/tree-of-thought-llm)
3. Zhang, Z. et al. **Multimodal Chain-of-Thought Reasoning in Language Models.**
   TMLR 2024. [arXiv:2302.00923](https://arxiv.org/abs/2302.00923)
4. Besta, M. et al. **Graph of Thoughts: Solving Elaborate Problems with Large Language
   Models.** AAAI 2024. [arXiv:2308.09687](https://arxiv.org/abs/2308.09687) ·
   [code](https://github.com/spcl/graph-of-thoughts)
5. Wang, X. et al. **Self-Consistency Improves Chain of Thought Reasoning in Language
   Models.** ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
6. Newell, A. & Simon, H. **Human Problem Solving.** Prentice-Hall, 1972.
