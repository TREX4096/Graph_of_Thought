# explanation.md — Reasoning, Interpretation, and Deep Dive

This document records **how I read the four papers, what every term means, and why
the implementation is built the way it is**. It is written to be read top-to-bottom
by someone who has not read the papers.

The reading order is the one the project instructions specify, and it is the right
one, because each paper is a direct response to a limitation of the previous:

```
Chain-of-Thought  →  Tree of Thoughts  →  Graph of Thoughts
   (2022)               (2023)                (2024)
      │                                          ▲
      └──────── Multimodal-CoT (2023) ───────────┘
                (orthogonal branch:
                 adds vision, not structure)
```

**Contents**

1. [The one idea that connects all four papers](#1-the-one-idea-that-connects-all-four-papers)
2. [Paper 1 — Chain-of-Thought Prompting](#2-paper-1--chain-of-thought-prompting)
3. [Paper 2 — Tree of Thoughts](#3-paper-2--tree-of-thoughts)
4. [Paper 3 — Multimodal Chain-of-Thought](#4-paper-3--multimodal-chain-of-thought)
5. [Paper 4 — Graph of Thoughts (the focus)](#5-paper-4--graph-of-thoughts-the-focus)
6. [The latency–volume tradeoff, explained properly](#6-the-latencyvolume-tradeoff-explained-properly)
7. [Glossary of every term](#7-glossary-of-every-term)
8. [How the papers map onto this codebase](#8-how-the-papers-map-onto-this-codebase)
9. [Implementation decisions and why I made them](#9-implementation-decisions-and-why-i-made-them)
10. [Bugs found during development (and what they taught me)](#10-bugs-found-during-development-and-what-they-taught-me)
11. [What I verified, and what I did not](#11-what-i-verified-and-what-i-did-not)

---

## 1. The one idea that connects all four papers

A language model generates text **one token at a time, left to right**. Each token is
chosen based on everything before it. There is no "undo", no "try both branches", no
"think about it and come back".

For a task like "what is the capital of France?", that's fine. For a task like "sort
these 64 numbers" or "play this game of 24", it is not — because those tasks require
you to *explore*, *evaluate*, and sometimes *abandon* a line of work.

All four papers attack this same problem, and they differ only in **what structure they
impose on the intermediate reasoning**:

| Paper | Structure of reasoning | What it can newly do |
|---|---|---|
| CoT | A **chain** | Show intermediate steps at all |
| CoT-SC | **k independent chains** | Sample several answers, vote |
| ToT | A **tree** | Branch, evaluate, backtrack, prune |
| GoT | An arbitrary **graph (DAG)** | **Aggregate** separate lines of reasoning; refine in loops |
| MM-CoT | A chain, but **two-stage + vision** | Reason over images, not just text |

The progression CoT → ToT → GoT is a progression in **graph topology**. That is the
single most important thing to understand about this project. Multimodal-CoT is on a
different axis entirely — it changes the *input modality*, not the *reasoning shape*.

---

## 2. Paper 1 — Chain-of-Thought Prompting

> Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*,
> NeurIPS 2022. (arXiv:2201.11903)

### The core claim

If you show a model a few examples that include **the intermediate reasoning steps**,
not just the answer, it will imitate that behaviour and produce its own reasoning —
and it gets dramatically more questions right.

### The mechanism, concretely

**Standard few-shot prompting** gives the model input→output pairs:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many does he have now?
A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch
   and bought 6 more, how many apples do they have?
A:                                    ← model must answer in one leap
```
The model answers **27**. Wrong.

**Chain-of-thought prompting** gives input→*reasoning*→output triples:

```
Q: Roger has 5 tennis balls. ...
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is
   6 tennis balls. 5 + 6 = 11. The answer is 11.        ← the chain of thought

Q: The cafeteria had 23 apples. ...
A:
```
The model now produces: *"The cafeteria had 23 apples originally. They used 20 to make
lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The
answer is 9."* Correct.

### Why it works — the deeper reason

This is the part worth internalising, because **it is the justification for everything
that follows in ToT and GoT**:

A transformer does a **fixed amount of computation per token**. A hard problem may need
more computation than one token's worth. By emitting intermediate tokens, the model
gives itself **more forward passes to work with**, and each intermediate result is
written into the context where later steps can read it.

In other words: **the context window is being used as a scratchpad / working memory.**
The reasoning chain is not just an explanation for humans — it is computation.

The paper states this as its first listed property:

> "chain of thought, in principle, allows models to **decompose multi-step problems into
> intermediate steps**, which means that **additional computation can be allocated** to
> problems that require more reasoning steps."

### Headline results

- PaLM 540B on GSM8K (grade-school math word problems): **18% → 57%** solve rate.
- This beat the prior state of the art, which was a *fine-tuned* GPT-3 175B **with a
  verifier** (33%).
- No gradient updates, no training data. Just eight hand-written exemplars.

### Emergence — the critical caveat

CoT is an **emergent ability of scale**. Below roughly 10B parameters it does not help
and often *hurts* — small models produce fluent-sounding but logically broken chains,
and then confidently follow them to a wrong answer.

**Why this matters directly for our project:** we are running open-source models, and
locally we can only fit a 1.5B model. A 1.5B model is *below the emergence threshold*.
This is not a flaw in our implementation — it is a documented property of the method,
and it is exactly why the local path uses a mock backend for logic validation and
defers real quality measurement to the HPC with an 8B+ model.

### The limitations that motivate the next paper

1. **One chain, one shot.** If the first step is wrong, everything after it is wrong.
   There is no recovery.
2. **No exploration.** The model never considers an alternative first step.
3. **No evaluation.** Nothing ever asks "is this partial answer any good?"

---

### CoT-SC (Self-Consistency) — the intermediate step

> Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, ICLR 2023.

Not one of our four PDFs, but both ToT and GoT treat it as the baseline to beat, so it
must be understood.

**The idea:** sample `k` independent chains at temperature > 0, then take a **majority
vote** over the final answers.

Formally: sample `[z₁…ₙ⁽ⁱ⁾, y⁽ⁱ⁾] ~ p(z, y | x)` for `i = 1…k`, then return
`argmax_y #{i : y⁽ⁱ⁾ = y}`.

**Why it helps:** there are many valid reasoning paths to one correct answer, but
errors are idiosyncratic. Correct answers cluster; wrong answers scatter.

**Its two limitations** (ToT names both explicitly):
1. **No local exploration.** Within any single chain there is still no branching.
2. **Voting needs a small answer space.** "Most frequent answer" is meaningless when
   the answer is a 64-element list or a paragraph of prose — no two samples will ever
   be identical.

The second point is decisive for our sorting task, and it's why our `cot_sc` baseline
uses *best-scored* selection rather than majority voting.

---

## 3. Paper 2 — Tree of Thoughts

> Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*,
> NeurIPS 2023. (arXiv:2305.10601)

### The framing: System 1 vs System 2

The paper opens with dual-process theory from cognitive science:

- **System 1** — fast, automatic, associative. ≈ token-by-token generation.
- **System 2** — slow, deliberate, planning. ≈ what LLMs lack.

ToT's goal is to bolt a System 2 onto a System 1 model. It borrows from Newell &
Simon's classical problem-solving work: humans search a **combinatorial problem space**,
a tree whose nodes are partial solutions and whose branches are operators.

The paper names the two shortcomings of CoT precisely:

> "1) **Locally**, they do not explore different continuations within a thought process
> — the branches of the tree. 2) **Globally**, they do not incorporate any type of
> planning, lookahead, or backtracking."

### The four design questions

ToT is a *framework*, and instantiating it means answering four questions. This
structure is worth memorising — GoT inherits and extends it.

#### 1. Thought decomposition — what is one "thought"?

A **state** is `s = [x, z₁…ᵢ]` — the input plus the thoughts so far.

A thought's size is a genuine engineering tradeoff:
- Too small (one token): the model can't evaluate whether it's promising.
- Too big (a whole book): the model can't generate diverse, coherent candidates.

The paper's own instantiations show the range:

| Task | One thought is... |
|---|---|
| Game of 24 | one equation, e.g. `13 - 9 = 4 (left: 4, 4, 10)` |
| Creative Writing | a paragraph-level writing plan |
| Crosswords | a word for one clue |

#### 2. Thought generator `G(p_θ, s, k)` — how to propose candidates

Two strategies:

- **(a) Sample i.i.d.** — call the same CoT prompt `k` times at temperature > 0.
  Best when the thought space is *rich* (paragraphs), where independent samples are
  naturally diverse.
- **(b) Propose sequentially** — one "propose prompt" that emits all `k` candidates at
  once. Best when the space is *constrained* (a single word, one equation), because
  seeing the other candidates in-context stops the model repeating itself.

*This distinction shows up in our code:* `Generate(branching_factor=k)` implements
strategy (a), which is the right choice for sorting (each candidate sorting is a long
structured object).

#### 3. State evaluator `V(p_θ, S)` — the heuristic

This is ToT's cleverest contribution. Classical search needs a heuristic; those are
normally **hand-programmed** (Deep Blue) or **learned** (AlphaGo). ToT proposes a third
option: **ask the LLM to evaluate the state**.

Two modes:

- **(a) Value each state independently** — prompt for a scalar (1–10) or a class
  (`sure` / `likely` / `impossible`). Achieved via *lookahead* ("can 5, 5, 14 reach 24?
  yes: 5+5+14") plus *commonsense* ("1 2 3 are too small to reach 24").
- **(b) Vote across states** — show the model all candidates and ask which is best.
  Used when quality is comparative rather than absolute (e.g. "which passage is more
  coherent?").

Key insight, quoted because it's the licence for the whole approach:

> "Such valuations do not need to be perfect, and only need to be **approximately
> helpful for decision making**."

#### 4. Search algorithm

- **BFS (Algorithm 1)** — keep the best `b` states per level. Used when the tree is
  shallow (T ≤ 3). This is a *beam search*.
- **DFS (Algorithm 2)** — go deep on the most promising state; if the evaluator says a
  state is hopeless (`V(s) ≤ v_th`), **prune the subtree and backtrack**.

### Results

| Method | Game of 24 success |
|---|---|
| GPT-4 + IO prompting | 7.3% |
| GPT-4 + CoT | 4.0% |
| GPT-4 + CoT-SC (k=100) | 9.0% |
| **GPT-4 + ToT (b=5)** | **74%** |

An 18× improvement over CoT. Search matters enormously for this class of problem.

### The limitation that motivates GoT

A tree has exactly one property that turns out to be fatal: **every node has exactly
one parent.**

Consequences:
- Two promising branches can **never be combined**. You must pick one and discard the
  other, throwing away whatever was good in the loser.
- There is no way to express "merge these four sorted chunks into one sorted list",
  because that operation has **four inputs and one output**.
- Pruned subtrees are gone forever.

GoT's entire contribution follows from removing this restriction.

---

## 4. Paper 3 — Multimodal Chain-of-Thought

> Zhang et al., *Multimodal Chain-of-Thought Reasoning in Language Models*, TMLR 2024.
> (arXiv:2302.00923)

### Why this paper is in the set

It is **not** part of the CoT → ToT → GoT structural progression. It is an orthogonal
extension: *what if the input includes images?* I read it to understand the boundary of
the family, and it turns out to contain one lesson that genuinely transfers.

### The problem

Textbooks have figures. Science questions have diagrams. A text-only CoT cannot reason
about them. The naive fix — caption the image, append the caption — loses information,
because a caption is a lossy summary of a figure.

### The two-stage framework

The paper's core proposal separates what CoT normally fuses:

```
        ┌──────────────────────────────────────────────┐
Stage 1 │ (Question, Context, Image)  →  Rationale      │   rationale generation
        └──────────────────────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────────┐
Stage 2 │ (Question, Context, Image, Rationale) → Answer│   answer inference
        └──────────────────────────────────────────────┘
```

Both stages are fine-tuned models (T5-based), and vision features are fused into both.
Note that this is **fine-tuning**, not prompting — a real difference from the other
three papers, which are all training-free.

### The finding that matters

The authors first tried the obvious one-stage approach and found something striking:
models **under 1B parameters generate *hallucinated rationales*** — plausible-sounding
reasoning that is factually wrong — and then the answer stage faithfully follows the
bad reasoning to a wrong answer. Giving the model a rationale made it *worse*.

Adding vision features fixed this: with the image actually available, the rationales
became grounded, and accuracy jumped. Their <1B model beat GPT-3.5 and human average on
ScienceQA.

### The transferable lesson

> **A reasoning step that is wrong is worse than no reasoning step at all, because
> downstream steps trust it.**

This is directly relevant to GoT and to our implementation. In a graph of thoughts, a
bad thought doesn't just produce a bad answer — it **propagates through every
aggregation that consumes it**. That is precisely why GoT scores and ranks thoughts
before merging them, and why our `KeepBest` sits between every `Generate` and the
`Aggregate` that follows.

It is also why our local 1.5B model is expected to underperform: it is in exactly the
hallucinated-rationale regime this paper documents.

---

## 5. Paper 4 — Graph of Thoughts (the focus)

> Besta et al., *Graph of Thoughts: Solving Elaborate Problems with Large Language
> Models*, AAAI 2024. (arXiv:2308.09687)

### The central claim

> "The key idea and primary advantage of GoT is the ability to model the information
> generated by an LLM as an **arbitrary graph**, where units of information ('LLM
> thoughts') are **vertices**, and **edges** correspond to **dependencies** between
> these vertices."

The human-reasoning motivation, from the introduction:

> "one could explore a certain chain of reasoning, backtrack and start a new one, then
> realize that a certain idea from the previous chain could be **combined** with the
> currently explored one, and **merge them both into a new solution**, taking advantage
> of their strengths and eliminating their weaknesses."

That word — **merge** — is the whole paper.

### Formal definition

GoT is the tuple **(G, T, E, R)**:

| Symbol | Name | Meaning |
|---|---|---|
| **G** = (V, E) | reasoning process | directed graph; `V` = thoughts, `E` = dependencies |
| **T** | transformations | the operations that modify `G` |
| **E** | evaluator | `E(v, G, p_θ) → score` |
| **R** | ranking | `R(G, p_θ, h) → h best thoughts` |

A **vertex** holds a solution — initial, intermediate, or final. Its form is
task-dependent: a sequence of numbers for sorting, a paragraph for writing.

A **directed edge (t₁, t₂)** means "thought t₂ was constructed **using t₁ as direct
input**" — i.e. the LLM was explicitly instructed to use t₁ when generating t₂.

Note `E` is used for both the edge set and the evaluator in the paper. Context
disambiguates; I mention it because it confused me on first reading.

### The three thought transformations

This is the heart of the paper. Each transformation adds vertices `V⁺` and edges `E⁺`.

#### Generation — 1 → k

```
V⁺ = {v₁⁺, …, v_k⁺}
E⁺ = {(v, v₁⁺), …, (v, v_k⁺)}
```

```
        v
      / | \
    v₁⁺ v₂⁺ v₃⁺
```
Generate `k` new thoughts from one existing thought. **This is what CoT-SC and ToT
already do.** Nothing new — GoT simply subsumes it.

#### Aggregation — k → 1  ★ THE KEY ONE ★

```
V⁺ = {v⁺}
E⁺ = {(v₁, v⁺), …, (v_k, v⁺)}
```

```
    v₁   v₂   v₃   v₄
      \   \   /   /
       \   \ /   /
          v⁺            ← in-degree 4
```

> "one can **aggregate arbitrary thoughts into new ones**, to combine and reinforce the
> advantages of these thoughts, while eliminating their disadvantages."

**Why this is impossible in a tree:** a tree node has in-degree ≤ 1, by definition. A
vertex with in-degree 4 is not a tree node. The moment you want to merge, you need a
graph. There is no way around it.

The paper also notes this generalises beyond single thoughts: by adding outgoing edges
from the *final* thoughts of several chains, you aggregate **entire reasoning paths**.

#### Refinement — self-loop

```
V⁺ = {}
E⁺ = {(v, v)}
```

```
      ┌───┐
      ▼   │
      v ──┘
```
Improve a thought in place. Also impossible in a tree — trees are acyclic.

### Scoring and ranking

**Score:** `E(v, G, p_θ)`. Note the signature includes the **whole graph G**, not just
the vertex. The paper explains why: "in some evaluation scenarios, scores may be
relative to other thoughts." Maximum generality.

**Rank:** `R(G, p_θ, h)` returns the `h` highest-scoring thoughts. The paper says it
"most often" just takes the top `h` by score — a simple strategy that works.

Crucially, scoring can be **local and exact** where the task permits:

> "use cases such as sorting use **simple local scoring functions**."

For sorting, the score is computable in Python. This is free, exact, and noise-free —
a large practical advantage over ToT, which leans on LLM-based state evaluation.

### System architecture (Section 4)

```
 ┌───────────────────────────────────────────────────────────┐
 │  CONTROLLER                                                │
 │    ┌──────────────────────┐   ┌────────────────────────┐   │
 │    │ GoO                  │   │ GRS                    │   │
 │    │ Graph of Operations  │   │ Graph Reasoning State  │   │
 │    │ ── STATIC ──         │   │ ── DYNAMIC ──          │   │
 │    │ the execution plan   │   │ the thoughts produced  │   │
 │    │ built once, upfront  │   │ updated as it runs     │   │
 │    └──────────────────────┘   └────────────────────────┘   │
 └────────┬──────────────┬──────────────┬────────────────────┘
          │              │              │
     ┌────▼────┐   ┌─────▼─────┐  ┌─────▼──────┐
     │ Prompter│   │  Parser   │  │  Scoring & │
     │         │   │           │  │ Validation │
     │ builds  │   │ extracts  │  │            │
     │ prompts │   │ thought   │  │ verifies + │
     │         │   │ state     │  │ scores     │
     └────┬────┘   └─────▲─────┘  └────────────┘
          │              │
          └──────► LLM ──┘
```

The **GoO / GRS distinction** is the architectural insight and I want to state it
clearly because it drove my implementation:

- **GoO is the plan.** Static. Built before execution. "Split into 4, sort each 3 ways,
  keep the best, merge pairwise 10 ways, keep the best." It is a DAG of *operations*.
- **GRS is the state.** Dynamic. "Operation 7 produced these 3 thoughts, with these
  scores, and this one was invalid." It is a DAG of *thoughts*.

One GoO shape, executed on 100 different inputs, produces 100 different GRSs.

### Use case: Sorting (Section 5.1)

The task: sort numbers 0–9 **with duplicates**. The paper is explicit about why LLMs
fail:

> "The considered LLMs are unable to sort a sequence of such numbers correctly beyond a
> certain length consistently **because duplicate counts do not match**."

That is a precise diagnosis and it shapes the scoring function.

**The decomposition is merge sort:**
1. Split the input into subarrays.
2. Sort each subarray (easy — they're short).
3. Merge sorted subarrays pairwise (easier than sorting — inputs are already ordered).

**The GoO from Figure 4, for 64 numbers:**

```
[64 numbers]
     │  Generate — split into 4 chunks of 16
     ├──────────┬──────────┬──────────┐
   chunk0     chunk1     chunk2     chunk3
     │ k=3       │ k=3      │ k=3      │ k=3     ← 3 candidate sortings each
   Score      Score      Score      Score
  KeepBest   KeepBest   KeepBest   KeepBest      ← N=1: best of the 3
     │          │          │          │
     └────┬─────┘          └────┬─────┘
      Aggregate k=10        Aggregate k=10       ← 10 merge attempts each
        Score                 Score
       KeepBest              KeepBest
          └──────────┬──────────┘
              Aggregate k=10
                 Score
                KeepBest
                   │
              [64 sorted]
```

Note the asymmetry: **k=3 for sorting, k=10 for merging.** Merging gets more attempts
because that's where global correctness is decided — a merge error corrupts the whole
result, while a chunk error is contained.

**The scoring function** — this is the formula I implemented verbatim:

```
error-scope = X + Y

X = Σᵢ₌₁^{m-1} sgn(max(bᵢ − bᵢ₊₁, 0))
Y = Σᵢ₌₀^{9} | |{b_p : b_p = i}| − |{a_q : a_q = i}| |
```

Reading it term by term:

- **X — sortedness.** `sgn(max(bᵢ − bᵢ₊₁, 0))` is 1 exactly when `bᵢ > bᵢ₊₁`, i.e. an
  adjacent descent. So X counts **adjacent inversions**, not total inversions. A list
  sorted except for one swapped neighbouring pair scores X = 1.
- **Y — multiset preservation.** For each digit 0–9, `|count_in_output − count_in_input|`.
  Catches dropped elements and hallucinated duplicates — the exact failure mode
  diagnosed above.

**Why you need both:** a model could return a perfectly ascending list that quietly
lost three elements (X = 0, but wrong). Or the right multiset in the wrong order
(Y = 0, but wrong). Only `X + Y = 0` means genuinely correct.

**Sign convention:** the paper converts to a positive "higher is better" score with
`max(n − error-scope, 0)`, and clips with `min(error-scope, n)` **for plotting only**
("to improve the clarity of plots, as some baselines result in large numbers of
outliers"). I kept clipping optional and off by default for analysis.

### Use case: Set Intersection (Section 5.2)

Split **set B** into subsets; intersect each against the **whole of A**; union the
results.

The reason this is valid is algebraic:

```
A ∩ (B₁ ∪ B₂ ∪ … ∪ B_m) = (A ∩ B₁) ∪ (A ∩ B₂) ∪ … ∪ (A ∩ B_m)
```

Intersection distributes over union, so the aggregation step is an **exact union**.
This is a nice illustration of the paper's broader point: *the right graph
decomposition follows from the algebraic structure of the problem*, and GoT gives you
the vocabulary to express whatever that structure turns out to be.

**Scoring:**
```
error-scope = X1 + X2 + Xd
  X1 = |C \ (A ∩ B)|    spurious elements
  X2 = |(A ∩ B) \ C|    missing elements
  Xd = duplicates in C
```
`Xd` exists "because the LLM expresses the set as a list in natural language" — a list
can repeat where a set cannot.

### Use case: Keyword Counting (Section 5.3)

Split text into passages, count country mentions per passage, **sum** the sub-counts.
Score = Σ |computed_count − true_count| over keywords.

The aggregation here is arithmetic addition — again exact, again a different merge
semantics from the previous two tasks.

### Use case: Document Merging (Section 5.4)

Merge several overlapping NDAs into one, minimising duplication while maximising
information retention.

This is the one task with **no exact scorer**, so the LLM is the judge:
- Query for *redundancy* (10 = none) and *information retention* (10 = all).
- **Ask 3 times for each and average** — single LLM judgements are noisy.
- Combine with the **harmonic mean**, which punishes a bad score on either axis
  (you cannot win by deleting everything, or by concatenating everything).

### Headline results

> "increasing the quality of sorting by **62% over ToT**, while simultaneously reducing
> costs by **>31%**."

Quality *and* cost, simultaneously — that's the claim worth testing.

---

## 6. The latency–volume tradeoff, explained properly

Section 6 is the paper's theoretical contribution and, in my view, the most elegant
part. It took me a couple of passes to understand, so here it is carefully.

### Definitions

- **Latency** of a thought `t` = number of hops from the input to `t` — the length of
  the **shortest path**. *"How long did I wait for this answer?"*
- **Volume** of a thought `t` = *"the number of preceding LLM thoughts that could have
  impacted t"* — formally, **the number of thoughts from which there exists a path to
  `t`**. *"How much accumulated reasoning informs this answer?"*

You want **high volume** (a well-informed answer) at **low latency** (fast, parallel).
These normally trade off against each other.

### The analysis

Fix total cost at Θ(N) thoughts for every scheme, with branching factor `k`:

| Scheme | Latency | Volume | Why |
|---|---|---|---|
| CoT | **N** | **N** | One long chain: the last thought sees all N, but you waited N steps |
| CoT-SC | N/k | N/k | k parallel chains: each is k× shorter, **but each sees only its own chain** |
| ToT | **log_k N** | **O(log_k N)** | A k-ary tree: fast, **but a leaf only sees its own root-to-leaf path** |
| **GoT** | **log_k N** | **N** | ✅ **both** |

### The key realisations

**CoT-SC's hidden cost.** Splitting into `k` chains divides latency by `k` — but it
divides volume by `k` too. The chains never talk to each other, so each answer is
informed by only 1/k of the work done. You bought speed by throwing away information.

**ToT's hidden cost.** This is the one that surprised me. A tree with N nodes is fast
to descend (log_k N). But look at a **leaf**: its ancestors are just the nodes on the
single path back to the root — that's log_k N nodes. The *other* N − log_k N thoughts
in the tree, all that exploration, **cannot influence it at all**, because there's no
path from them to the leaf. ToT does a lot of work and then discards most of it.

**How GoT gets both.** The paper's construction: a complete k-ary tree joined at its
leaves to a **mirrored k-ary tree with its edges reversed**.

```
       INPUT              ← fan out (the tree): latency log_k N
        /|\
       / | \
      ●  ●  ●
     /|\ /|\ /|\
    ● ● ● ● ● ● ●         ← N leaves, all explored in parallel
     \|/ \|/ \|/
      ●  ●  ●             ← fan back IN (the mirror): aggregation
       \ | /
        \|/
       OUTPUT             ← every one of the N thoughts has a path here
```

The mirrored half is made entirely of **aggregations**. Because every leaf has a path
to the output through them, volume = N. Because the mirror is also log_k N deep, total
latency stays Θ(log_k N).

> "GoT is the only scheme to come with both a low latency of log_k N and a high volume
> N. This is enabled by the fact that GoT **harnesses aggregations of thoughts**, making
> it possible to reach the final thought from any other intermediate thought."

**This is the whole argument for graphs in one sentence:** aggregation is what lets
information flow back together after it fans out. A tree can only fan out.

### I verified this empirically

`got/metrics.py` computes volume and latency by BFS over the actual graphs our runs
produce. On a 32-element sorting problem:

| Scheme | thoughts | aggregations | **volume** | **latency** |
|---|---|---|---|---|
| IO | 2 | 0 | 1 | 1 |
| CoT | 3 | 0 | 2 | 2 |
| CoT-SC | 7 | 0 | 2 | 2 |
| ToT | 9 | 0 | **6** | **6** |
| **GoT** | 39 | **15** | **18** | **7** |

Exactly the predicted pattern: GoT achieves **3× the volume of ToT for one extra hop of
latency**, and it is the only scheme with a non-zero aggregation count. The theory is
reproduced, not just cited.

---

## 7. Glossary of every term

Alphabetical. These are the terms that must be understood to read the papers or this
codebase.

| Term | Meaning |
|---|---|
| **Aggregation** | GoT transformation merging k thoughts into 1. Creates a vertex with **in-degree > 1**. Impossible in a tree. The defining feature of GoT. |
| **Backtracking** | Abandoning an unpromising branch and returning to an earlier state. Introduced by ToT. |
| **Beam search** | Keeping the best `b` candidates at each level. ToT's BFS is beam search with beam width `b`. |
| **Branching factor (k)** | How many candidates to generate from one thought. |
| **Chain of Thought (CoT)** | A series of intermediate natural-language reasoning steps between input and output. |
| **CoT-SC** | Self-Consistency: sample k chains, take the most frequent answer. |
| **DAG** | Directed Acyclic Graph. The actual shape of a GoT reasoning graph. |
| **Emergent ability** | A capability absent in small models that appears above a scale threshold. CoT is one; it needs ≳10B params. |
| **Error scope** | GoT's task-specific error metric. For sorting, `X + Y`. |
| **Few-shot / ICL** | In-context learning: giving examples in the prompt instead of fine-tuning. |
| **GoO** | **Graph of Operations**. The **static** execution plan — a DAG of operations, built before the run. |
| **GRS** | **Graph Reasoning State**. The **dynamic** state — the thoughts actually produced, with scores and validity. |
| **Hallucinated rationale** | Plausible-sounding but wrong reasoning. Documented in MM-CoT for <1B models; dangerous because later steps trust it. |
| **In-degree** | Number of incoming edges. **In-degree > 1 ⟺ aggregation ⟺ genuinely a graph.** |
| **Latency** | Hops from input to a thought (shortest path). |
| **Multiset** | A set allowing duplicates. Sorting must preserve the input multiset — term Y of the score. |
| **Parser** | Module extracting structured thought state from raw LLM text. |
| **Prompter** | Module building prompt strings; owns all task-specific wording. |
| **Refinement** | GoT transformation improving a thought in place; formally a self-loop `(v, v)`. |
| **Score / Evaluator E** | `E(v, G, p_θ) → ℝ`. May be exact Python (sorting) or an LLM query (document merging). |
| **Ranking R** | `R(G, p_θ, h)` → the h best thoughts. Usually just top-h by score. |
| **State (ToT)** | `s = [x, z₁…ᵢ]` — input plus thoughts so far. A partial solution. |
| **System 1 / System 2** | Fast-automatic vs slow-deliberate cognition. ToT's framing for what LLMs lack. |
| **Thought** | A coherent unit of intermediate reasoning; a **vertex** in GoT. |
| **Thought decomposition** | Choosing how big one thought should be. Too small → unevaluable; too big → incoherent. |
| **Volume** | Number of thoughts with a path *to* a given thought. How much reasoning informs it. |
| **X (sorting)** | Count of adjacent descents — measures sortedness. |
| **Y (sorting)** | Sum of per-digit frequency mismatches — measures multiset preservation. |

---

## 8. How the papers map onto this codebase

Every framework concept has exactly one home in the code:

| Paper concept | Section | Implementation |
|---|---|---|
| Thought (vertex) | GoT 3.1 | [`got/thought.py`](got/thought.py) — `Thought` |
| Edge (dependency) | GoT 3.1 | `Thought.add_predecessor()` |
| Generation transformation | GoT 3.2 | [`got/operations.py`](got/operations.py) — `Generate` |
| **Aggregation transformation** | GoT 3.2 | [`got/operations.py`](got/operations.py) — **`Aggregate`** |
| Refinement transformation | GoT 3.2 | [`got/operations.py`](got/operations.py) — `Improve` |
| Evaluator `E` | GoT 3.3 | [`got/operations.py`](got/operations.py) — `Score` |
| Ranking `R` | GoT 3.3 | [`got/operations.py`](got/operations.py) — `KeepBest` |
| Prompter | GoT 4.1 | [`got/prompter.py`](got/prompter.py) — `AbstractPrompter` |
| Parser | GoT 4.2 | [`got/prompter.py`](got/prompter.py) — `AbstractParser` |
| Scoring & Validation | GoT 4.3 | `Score` + `KeepValid` + `state["valid"]` |
| Controller | GoT 4.4 | [`got/controller.py`](got/controller.py) — `Controller` |
| **GoO** (static plan) | GoT 4.5 | The `Operation` DAG built in `got/tasks/*/graphs.py` |
| **GRS** (dynamic state) | GoT 4.5 | `Operation.thoughts` + `Controller.all_thoughts()` |
| Sorting score `X + Y` | GoT 5.1 | [`got/tasks/sorting/scoring.py`](got/tasks/sorting/scoring.py) |
| Sorting GoO (Figure 4) | GoT 5.1 | [`got/tasks/sorting/graphs.py`](got/tasks/sorting/graphs.py) — `got_sorting_goo` |
| Set intersection | GoT 5.2 | [`got/tasks/set_intersection/`](got/tasks/set_intersection/) |
| Latency & Volume | GoT 6 | [`got/metrics.py`](got/metrics.py) |
| Table 2 bounds | GoT 6 | `metrics.theoretical_bounds()` |
| ToT BFS/beam | ToT Alg. 1 | `tot_goo()` — refine + score + `KeepBest(b)` per level |
| CoT-SC k chains | CoT-SC | `cot_sc_goo()` — `Generate(k)` + `KeepBest(1)` |
| CoT chain | CoT | `cot_goo()` — `Generate(1)` + `Improve` |

---

## 9. Implementation decisions and why I made them

These are choices where the paper left room, and I want the reasoning on record.

### 9.1 Three interchangeable LLM backends

**Constraint:** this laptop has 7.3 GB RAM (3.5 GB visible to WSL), 8 cores, and an
**AMD integrated GPU — no CUDA**. A 7B model in fp16 needs ~14 GB. It cannot run here.

**Decision:** put every model call behind `AbstractLanguageModel` and provide:

| Backend | Where | Model | Purpose |
|---|---|---|---|
| `MockLM` | laptop | none | validate graph logic, free and instant |
| `LlamaCppLM` | laptop | Qwen2.5-1.5B Q4 (~1 GB) | real-but-small end-to-end check |
| `HFLM` / `VLLMLM` | HPC | Llama-3.1-8B/70B | paper-comparable numbers |

The same task code runs on all three. Only a config flag changes.

### 9.2 The mock backend must be *fallible*, not an oracle

This is the subtlest design decision in the project.

A mock that sorts perfectly would make every scheme score 100% and prove nothing. The
mock must **fail the way a real LLM fails**, or the pipeline isn't being tested.

So `MockLM`:
- corrupts answers with modes matching the score's penalties — **drop** and **duplicate**
  (break term Y), **swap neighbours** (break term X);
- **degrades with input length** (`length_sensitivity`), reproducing the paper's core
  observation that LLMs fail past a certain length;
- treats **merging as easier than sorting** (×0.6 error) and **refinement as easier
  still** (×0.5) — which is *why* the merge-sort decomposition helps at all;
- is **seeded**, so runs are exactly reproducible — something no real LLM offers.

Setting `error_rate=0.0` turns it into a perfect oracle, which isolates framework bugs
from model errors. There's a test for exactly that
([`test_zero_error_rate_is_a_perfect_oracle`](tests/test_graph.py)).

**Honest limitation:** MockLM validates control flow and bookkeeping only. It says
nothing about prompt quality. **Mock numbers are not paper-comparable.**

### 9.3 Structural steps don't call the LLM

Splitting a 64-element list into 4 chunks is deterministic Python. Spending an LLM call
on it would add cost *and* a failure mode for zero benefit — a model that miscounts
while chunking corrupts term Y before reasoning even starts.

**Mechanism:** `Prompter.build()` returns `None` to mark a step as local, and the
operation routes to `Parser.parse_local()` instead. Verified free by
[`test_split_is_local_and_free`](tests/test_graph.py).

This is faithful to the paper — Figure 4 shows the first Generate as a structural split.

### 9.4 Refinement as a fresh vertex, not a literal self-loop

The paper defines refinement as `E⁺ = {(v, v)}` — a genuine self-loop. I materialise the
refined result as a **new vertex whose predecessor is the original**.

**Why:** a literal self-loop makes the graph cyclic, which breaks topological sorting,
BFS-based volume computation, and visualisation. Representing it as a new vertex keeps
the graph acyclic while preserving the full provenance chain. The distinction is
bookkeeping, not behaviour — the information content is identical.

### 9.5 Scoring is separate from validation

A thought can be **well-formed but poor** (valid, low score) or **malformed entirely**
(invalid — the model returned prose where a list was required).

Keeping these separate lets `KeepValid` discard garbage before it pollutes the ranking.
This matters far more with open 1.5B models than with the paper's GPT-3.5/4, which
almost always return parseable output.

### 9.6 Defensive parsing everywhere

`extract_list()` takes the **last** bracketed list in the response, not the first —
chatty models restate the input before answering. Code fences are stripped. A parse
failure produces an invalid thought, never an exception: **one bad response must not
abort a multi-hour HPC job.** `run_benchmark.py` wraps each instance in try/except for
the same reason.

### 9.7 Five schemes sharing one engine

`io`, `cot`, `cot_sc`, `tot`, `got` all use the same Controller, backend, prompts and
scorer. **They differ only in graph topology.** That's a deliberate experimental design:
any measured difference is attributable to structure alone, not to prompt or
implementation differences. Tests assert that the four baselines contain **zero**
aggregations and that GoT contains some.

---

## 10. Bugs found during development (and what they taught me)

Recording these because each one exposed something real about the method.

### Bug 1 — Few-shot examples leaking into answers

**Symptom:** a 32-element input produced a **205-element** output.

**Cause:** `MockLM` scraped *every* bracketed list from the prompt — including the
16-element list in the **few-shot example** — and merged them all into its answer.

**Fix:** all prompt templates place the real payload **last**, so the mock takes the
last *N* lists, where *N* depends on the operation (`_payload_lists`).

**Lesson:** this is a mock-specific bug, but it mirrors a real failure mode —
instruction-tuned models genuinely do sometimes copy few-shot exemplars into their
answers. Taking the **last** match in `extract_list()` guards against the real version.

### Bug 2 — Aggregated thoughts scored against the wrong target ★

**Symptom:** GoT scored **0/32** while visibly producing a near-perfect sorted list.

**Cause:** `sorting_score` compares `current` against `original`. For a merged thought I
was inheriting `original` from **only the first parent** — so a 32-element merged result
was being compared against a 16-element chunk. Term Y then reported ~16 "extra"
elements and the score floored at zero.

**Fix:** define `original` **recursively down the graph**:
```
chunk from split       →  original = that chunk
sort of a chunk        →  original = parent's original
merge of A and B       →  original = A.original + B.original
```
At the root of the merge tree, the concatenation reconstitutes the full input multiset.

**Lesson — and this is the important one:** in a tree, "what was I supposed to produce?"
has one answer inherited from one parent. **In a graph, provenance must be combined from
all parents.** Aggregation doesn't just change the topology; it changes how metadata
propagates. This bug is *specific to graphs* and could not occur in CoT or ToT.

### Bug 3 — Aggregation silently discarding half its input

**Symptom:** on set intersection, GoT scored **worse than the IO baseline** (error 13.60
vs 3.90) — the opposite of the expected result.

**Cause:** the union prompt contains neither "intersect" nor "merge", so `MockLM` fell
through to its generic "sort one list" branch, read only **one** payload list, and threw
the other away. Every aggregation lost half its data; across a 3-level merge tree only
~1/4 of elements survived.

**Fix:** an explicit union branch, tested before the generic ones.

**Lesson:** a silent fallback is worse than a crash. And it's a reminder that with
*real* models the analogous failure — a model that misreads the aggregation prompt and
returns only one input — would be equally silent. This is precisely why `Score` sits
after every `Aggregate`: an aggregation that dropped half its input scores badly and
gets pruned.

After the fix: **error 1.10 vs 3.90 (−72%), accuracy 40% vs 0%.**

---

## 11. What I verified, and what I did not

Being explicit about this, because a replication's value depends on it.

### Verified ✅

- **The formal structures.** Volume, latency, in-degree, and the DAG property are
  computed on real graphs and unit-tested (46 tests passing).
- **Table 2's qualitative claim.** GoT achieves **3× ToT's volume for one extra hop** on
  a real 32-element run. The only scheme with aggregations.
- **The scoring formulae.** `X + Y` for sorting and `X1 + X2 + Xd` for intersection are
  implemented verbatim from the paper and tested against hand-computed cases.
- **The relative ordering of schemes.** On mock runs: IO 26 < CoT 28 < CoT-SC 29 <
  ToT 30 < **GoT 31** (score out of 32) — the paper's predicted ordering.
- **That aggregation is what does the work.** Set intersection: −72% error vs the IO
  baseline; the whole gain vanished when aggregation was broken (Bug 3).
- **Framework correctness independent of model quality.** With `error_rate=0.0` the
  pipeline reaches a provably perfect answer.

### NOT verified ⚠️

- **Absolute quality numbers.** The mock backend is not a language model. The "62%
  improvement over ToT" claim **cannot** be confirmed without real LLM runs on the HPC.
  Mock accuracy figures are diagnostics, not results.
- **The cost claim.** The paper reports GoT **reducing** cost >31% vs ToT. In our runs
  GoT uses *more* tokens than our ToT baseline (4995 vs 724 per instance). This is not a
  contradiction — our ToT baseline is deliberately lean (beam width 1, depth 3), while
  the paper's ToT configuration is much wider. A fair cost comparison requires matching
  the paper's ToT budget, which is HPC work.
- **Prompt quality.** Every prompt is untested against a real model. Small open models
  are far more format-sensitive than GPT-4, and prompt iteration on the HPC should be
  expected.
- **Keyword counting and document merging.** Datasets are generated and MockLM supports
  keyword counting, but the GoO builders for these two tasks are not yet written.

### The honest summary

**What this project has established:** a correct, tested, well-documented implementation
of the Graph of Thoughts framework, which demonstrably builds the graph structures the
paper describes, reproduces its theoretical latency–volume result empirically, and runs
identically on a laptop and an HPC cluster with open-source models.

**What remains:** running it against a real open-weights model at scale to obtain
quality numbers comparable to the paper's. Everything needed for that is in place —
`scripts/slurm/`, the vLLM backend, and the datasets.

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
