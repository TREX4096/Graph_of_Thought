# Project plan — status and next steps

**Causality-Preserving Graph Coarsening for Efficient LLM Reasoning**
Prasoon Raj (2023EE10708) · Nikhil Bansal (2023EE10787)
Guided by Subhanu Halder · Supervisor Prof. Sandeep Kumar

Last updated: 17 Oct 2026

---

## 1. Where we are

The project has three phases. Two are complete; the third started today.

| Phase | Status |
|---|---|
| **1. Replicate** — rebuild GoT and all four baselines on one engine | ✅ complete |
| **2. Measure** — establish what is true, locate the cost | 🟡 partial |
| **3. Compress** — shrink the graph without breaking causality | 🔵 started |

### What Phase 1 established

A working framework: five schemes (IO, CoT, CoT-SC, ToT, GoT) sharing one
controller, backend, prompt set and scorer, so any measured difference is
attributable to graph structure alone. Four interchangeable backends, 60
automated tests, and runs on the authors' own published datasets.

### What Phase 2 established

Run R1 — 100 instances, 64-element sorting, Qwen2.5-7B, zero parse failures:

| Scheme | Error | Tokens | Volume | Latency |
|---|---|---|---|---|
| IO | 9.83 | 520 | 1.0 | 1.0 |
| CoT | 10.34 | 1,169 | 2.0 | 2.0 |
| CoT-SC | 8.17 | 887 | 2.0 | 2.0 |
| **ToT** | **7.45** | 2,185 | 5.3 | 5.3 |
| GoT | 8.28 | 7,910 | 20.0 | 9.0 |

Four findings:

1. **Structural claims reproduce exactly** — 22/22 checks. GoT reaches volume
   20 at latency 9 and is the only scheme that aggregates (15 per instance);
   every other scheme reports exactly 0.
2. **The quality claim does not reproduce at 7B.** ToT beats GoT by 11% at
   3.6× fewer tokens. GoT still beats direct prompting (+15.8%), so structure
   helps — it just loses to a cheaper structure.
3. **Cost is concentrated**: aggregation consumes ~65% of the token budget.
4. **Naive compression fails**: cutting cost 9× raises error to 10.50, worse
   than using no reasoning structure at all (IO is 9.83).

Finding 2 only emerged after we found our own ToT baseline could not reject a
refinement that worsened the answer — a defect that had inflated GoT's margin
by ~18 points in our favour.

### What Phase 2 still owes

**R2–R5 never completed.** The matrix died after R1 on 25 Sep. Outstanding:

- R2 — difficulty sweep (32, 128 elements)
- R3 — set intersection, all five schemes
- R4 — aggregation-attempt sweep (k = 3, 5, 10)
- R5 — chunk-count sweep (m = 2, 4, 8) and the refinement ablation

R4 and R5 matter most: they show *where inside the graph* cost converts to
quality, which is what the coarsening operator needs.

---

## 2. What changed today

Li et al., **"Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via
Step Entropy"** (ICLR 2026, [arXiv:2508.03346](https://arxiv.org/abs/2508.03346))
reports that ~80% of *low-entropy* reasoning steps can be pruned with minor
accuracy loss, while pruning random or high-entropy steps hurts badly.

**Why this is on our critical path.** Our open question has been: *which
vertices can be dropped?* Marginal contribution answers it exactly but needs an
ablation run per vertex — far too expensive to use as an operator. Step entropy
is a candidate **cheap surrogate**: it falls out of the same forward pass that
generated the thought, at no extra cost.

$$H(S_i) = \frac{1}{M_i}\sum_{j} H(t_{ij} \mid c_{ij}), \qquad
H(t\mid c) = -\sum_{w \in V} p(w\mid c)\log_2 p(w\mid c)$$

The paper measures this on **chains**. Whether it transfers to **graph
vertices** is untested — and that gap is where our contribution sits.

### Built today

- `got/entropy.py` — segmentation, length-normalised step entropy, selection
  (low / high / random) and `[SKIP]` pruning.
- `got/tasks/math_reasoning/` — GSM8K and MATH-500 prompts plus an answer
  scorer that handles LaTeX equivalence (`1/2`, `\frac{1}{2}`, `0.5`).
- `scripts/download_math_data.py` — GSM8K (1,319) and MATH-500 (500), fetched.
- `scripts/run_entropy_pruning.py` — the three-stage protocol.
- `generate_with_entropy()` on the backends; vLLM returns top-k logprobs,
  MockLM synthesises uneven entropy so the path is testable without a GPU.

**Caveat to carry forward:** entropy is computed over top-k logprobs
renormalised, because a serving engine does not expose the full vocabulary.
That is a *lower bound* on true entropy — fine for ranking steps, not for
quoting absolute values.

---

## 3. Next steps

### Step 1 — Validate the criterion on chains  🔵 running now

GSM8K and MATH-500, 200 problems each, ratios 0.5 and 0.8, all three
strategies.

**Success looks like:** low-entropy pruning retains accuracy while high and
random degrade. The controls are what make it a result — "removing 80% of steps
barely hurts" means nothing if removing a *different* 80% also barely hurts.

> ⚠️ **Known risk.** The paper used *reasoning* models (DeepSeek-R1-7B/14B,
> Qwen3-8B) that emit long `<think>` chains with dozens of steps. We are
> running **Qwen2.5-7B-Instruct**, an instruct model whose chains are short.
> If mean steps per chain is below ~5 there is little to prune and the
> experiment cannot show anything either way. The runner warns when this
> happens.
>
> **Mitigation, already available:** `DeepSeek-R1-Distill-Qwen-7B` is
> **already cached on the server** — the same family the paper used. If the
> Qwen2.5 run shows too few steps, rerun against R1-Distill. This is the
> single most likely correction needed and it costs nothing but GPU time.

### Step 2 — Lift entropy to graph vertices  ← the novel contribution

The experiment nobody has run:

1. Instrument a GoT run on sorting so every vertex records its generation
   entropy (the plumbing now exists).
2. Independently measure each vertex's **marginal contribution** by ablation:
   remove it, re-run, measure the change in error scope.
3. **Correlate the two.** Does low entropy predict low contribution?

| Outcome | Meaning |
|---|---|
| Strong correlation | Entropy is a cheap, principled pruning criterion for graphs. This is the result. |
| Weak correlation | Chain intuitions do not transfer. Also publishable, and it tells us a graph-native criterion is required. |

Either way this answers the open question on the Future Plan slide, with
evidence rather than a hope.

### Step 3 — Build the coarsening operator

Specification already fixed (`explanation.md`, deck slide 14):

- **C1 — Reachability.** Every surviving vertex must retain a directed path to
  the final thought.
- **C2 — No spurious causality.** Contract `a` and `b` only if the merged
  content is derivable from `parents(a) ∪ parents(b)`.

Three admissible moves, in order of how soon they can be tested:

1. **Level contraction** — replace the two pairwise merge levels with a single
   4-way `Aggregate`. Provably satisfies C1 and C2, removes a level, and is
   implementable with the existing `Aggregate` class. *Build this first.*
2. **Entropy-guided vertex pruning** — drop vertices below an entropy
   threshold, subject to C1. Depends on Step 2.
3. **Adaptive branch width** — allocate *k* per vertex by measured difficulty
   rather than uniformly.

> ⚠️ Moves 2 and 3 must be distinguished clearly from the naive pruning we
> showed fails. Naive is **uniform and blind**; coarsening is **per-vertex and
> measured**. If that distinction is not crisp, it reads as a rerun of a failed
> experiment.

### Step 4 — Finish Phase 2

Rerun R2–R5. Cheap, already scripted (`scripts/run_tmux.sh`), and R4/R5 feed
directly into Step 3.

---

## 4. Success criterion

Unchanged, and measured rather than aspirational:

> **Reach error 7.45 at 2,185 tokens** — the operating point Tree of Thoughts
> occupies today. Graph of Thoughts currently sits at **8.28 error and 7,910
> tokens**.

Cost factorises as `|V| × tokens-per-vertex`. Chain compression attacks the
second factor; coarsening attacks the first. If they are orthogonal the savings
multiply — and testing whether they actually are is itself an open question.

---

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Instruct model produces too few steps to prune | **high** | R1-Distill already cached; rerun against it |
| Entropy does not transfer to graph vertices | medium | Still a reportable finding; falls back to ablation-measured contribution |
| The two compression axes interact | medium | Test explicitly rather than assuming |
| 7B result is a scale artefact | medium | Repeat at larger scale if GPU budget allows |
| Top-k entropy approximation too coarse | low | Raise `--logprobs`; only ranking is needed |
| Shared GPU / proxy-gated network | low | Runs detached; offline mode; ship code by git bundle when the proxy is down |
