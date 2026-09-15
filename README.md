# Graph of Thoughts — B.Tech Project Replication

A from-scratch implementation of **Graph of Thoughts** (Besta et al., AAAI 2024),
built to run on **open-source LLMs only — no API keys, no paid services** — both
locally on a laptop and at scale on HPC GPU nodes.

> **Paper:** [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)
> **Reference implementation:** [spcl/graph-of-thoughts](https://github.com/spcl/graph-of-thoughts)

For the full conceptual deep-dive across all four papers — every term explained,
diagrams, and the reasoning behind each design decision — see **[explanation.md](explanation.md)**.

**New to LLMs?** Start with these, in order:

| Section | What it gives you |
|---|---|
| [§0 — LLMs in fifteen minutes](explanation.md#0-start-here--llms-in-fifteen-minutes) | tokens, prompts, temperature, batching — no prior knowledge assumed |
| [§17 — The GoT paper, section by section](explanation.md#17-the-got-paper-read-section-by-section) | a plain-English reading companion for the PDF |
| [§18 — Datasets](explanation.md#18-datasets--which-ones-and-where-they-come-from) | which data, and why there is nothing to download |
| [§19 — Models and vLLM](explanation.md#19-models-and-vllm) | what the paper ran, what we run, what vLLM is |
| [§20 — GPU replication runbook](explanation.md#20-gpu-replication-runbook) | clone → results, for a plain GPU server **or** a SLURM cluster |

---

## What this project is about

A language model writes text one token at a time, left to right. It cannot backtrack,
try two approaches at once, or combine two partial solutions. For hard problems, that's
a real limitation.

Three papers attack this by imposing structure on the model's intermediate reasoning:

| Scheme | Reasoning shape | New capability |
|---|---|---|
| **CoT** | a chain | show intermediate steps |
| **ToT** | a tree | branch, evaluate, backtrack |
| **GoT** | a **graph** | **aggregate** separate lines of reasoning |

Aggregation — merging *k* thoughts into one — requires a vertex with **in-degree > 1**,
which a tree cannot express by definition. That single structural fact is GoT's entire
contribution, and this repository implements it.

**Concretely:** to sort 64 numbers, GoT splits them into chunks, sorts each chunk
several different ways, keeps the best of each, then **merges them back together**
pairwise. The merging step is the part CoT and ToT cannot do.

<p align="center">
  <img src="docs/figures/graph_got.png" alt="GoT reasoning graph" width="720">
</p>

*An actual reasoning graph from this implementation: 39 thoughts, 15 aggregations (red).
The diamond shape — fan out, then fan back in — is what buys GoT high volume at low
latency. Compare `docs/figures/graph_tot.png`, which is a plain tree.*

---

## Quick start

```bash
# 0. Activate the environment (WSL / Linux)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTP

# 1. Install the package
pip install -e .

# 2. Generate the benchmark datasets
python scripts/generate_data.py --out data --seed 42 --small

# 3. Run all five schemes with the free mock backend (instant, no model needed)
python scripts/run_benchmark.py \
    --task sorting --data data/sorting/sorting_32.csv --limit 30 \
    --backend mock --schemes io cot cot_sc tot got --out results/demo

# 4. Draw the reasoning graphs
python scripts/visualize_graph.py --out docs/figures

# 5. Run the test suite
pytest -q
```

Expected output from step 3:

```
scheme       acc      err     tokens   calls    vol    lat
----------------------------------------------------------
io         3.33%     4.97        125     1.0    1.0    1.0
cot        0.00%     2.97        300     2.0    2.0    2.0
cot_sc     0.00%     3.67        377     1.0    2.0    2.0
tot        3.33%     2.93        724     3.0    6.0    6.0
got       10.00%     1.90       4995     7.0   18.0    7.0
```

`err` is the paper's error-scope (lower is better); `vol`/`lat` are volume and latency
from Section 6. **GoT has the lowest error and 3× the volume of ToT** — the paper's
predicted result.

---

## The three backends

Everything runs through one interface (`AbstractLanguageModel`), so the same task code
works everywhere. Pick a backend with `--backend`:

| Backend | Runs on | Model | Use it for |
|---|---|---|---|
| `mock` | anything | none | Validating graph logic. Free, instant, deterministic. |
| `llamacpp` | laptop CPU | Qwen2.5-1.5B Q4 (~1 GB) | Real-but-small end-to-end check. |
| `hf` | HPC GPU | Llama-3.1-8B, Qwen2.5-7B | Single-GPU runs; supports 4-bit. |
| `vllm` | HPC GPU | up to Llama-3.1-70B | **Fast** multi-GPU runs. The one for real results. |

> **Why a mock backend?** This laptop has 3.5 GB of RAM in WSL and an AMD integrated
> GPU (no CUDA), so no real model of useful size fits. But what we're replicating is the
> *control structure*, not a language model — and that can be tested exactly without
> one, provided the fake model fails the way a real one does. `MockLM` drops, duplicates
> and misorders elements, and **degrades with input length**, reproducing the paper's
> motivating failure. See [explanation.md §14.2](explanation.md#142-the-mock-must-be-fallible-not-an-oracle).
>
> **Mock numbers are diagnostics, not results.** Paper-comparable quality figures
> require a real model on the HPC.

### Running locally with a real model

```bash
bash scripts/download_local_model.sh          # ~1 GB GGUF, CPU-friendly

python scripts/run_benchmark.py \
    --task sorting --data data/smoke/sorting_32_small.csv --limit 2 \
    --backend llamacpp \
    --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --schemes io got --out results/local
```

Expect this to be **slow** (CPU inference) and **weak** — a 1.5B model is below the
scale at which chain-of-thought reliably emerges. That's a documented property of the
method, not a bug in the code.

---

## Running on a GPU machine

**First, find out which kind of machine you have** — the workflows are different:

```bash
for c in sbatch srun sinfo; do printf "%-8s %s\n" "$c" "$(command -v $c || echo no)"; done
nvidia-smi
```

| Result | You have | Use |
|---|---|---|
| all `no`, `nvidia-smi` works | a **plain GPU server** | `scripts/run_direct.sh` |
| `sbatch` resolves | a **SLURM cluster** | `scripts/slurm/run_got.sbatch` |

> `sinfo: command not found` is an answer, not a problem — it means there is no scheduler,
> so there is nothing to install. **Nothing in this project needs root**: conda, pip, model
> weights and the benchmark all live under directories you own.

### Plain GPU server (no scheduler)

```bash
# Once:
export CONDA_BASE=$(dirname $(dirname $(which conda)))   # wherever your conda lives
bash scripts/slurm/setup_hpc_env.sh                      # auto-detects; skips `module`

# Check which GPU is free, then run detached (survives SSH disconnect):
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash scripts/run_direct.sh --bg

tail -f logs/got_sorting_64_*.log
```

Everything is configured by environment variable — no file editing:

```bash
BACKEND=mock LIMIT=5 bash scripts/run_direct.sh                   # free dry run
TASK=set_intersection LENGTH=32 bash scripts/run_direct.sh --bg   # paper §5.2
MODEL_ID=Qwen/Qwen2.5-7B-Instruct AGG_K=5 bash scripts/run_direct.sh --bg
```

### SLURM cluster

```bash
bash scripts/slurm/setup_hpc_env.sh
sbatch scripts/slurm/run_got.sbatch

sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,LENGTH=64,LIMIT=100 \
       scripts/slurm/run_got.sbatch
sbatch --export=ALL,TASK=set_intersection,LENGTH=32 scripts/slurm/run_got.sbatch
```

**Before your first submission**, edit `scripts/slurm/run_got.sbatch` to match your
cluster — partition name, account code, CUDA module version. Check with `sinfo` and
`module avail cuda`.

> **Full step-by-step runbook for both paths** — machine discovery, model pre-download, a
> trial run, and a table of failure modes — is
> [explanation.md §20](explanation.md#20-gpu-replication-runbook).

Model choice by GPU memory:

| GPU RAM | Model | Flags |
|---|---|---|
| 16 GB | `Qwen/Qwen2.5-7B-Instruct` | `--backend hf --load-in-4bit` |
| 24 GB | `meta-llama/Llama-3.1-8B-Instruct` | `--backend vllm` |
| 40 GB | `Qwen/Qwen2.5-32B-Instruct` | `--backend vllm` |
| 80 GB | `meta-llama/Llama-3.1-70B-Instruct` | `--backend vllm --tensor-parallel-size 2` |

> Llama models are **gated** on HuggingFace — accept the licence, then
> `huggingface-cli login`. **Qwen models are ungated** and need no token; prefer them
> if you hit access problems.

---

## Estimating HPC cost before you submit

GPU hours are the budget. Price a configuration **before** queueing it:

```bash
python scripts/estimate_cost.py --data data/sorting/sorting_64.csv \
    --limit 100 --model-size 8b --aggregation-attempts 10 5 3
```

```
scheme     agg_k  batches     seqs   decode tok    GPU time
-------------------------------------------------------------
io             -      100      100        7,200     0.1 min
cot            -      200      200       14,400     0.1 min
cot_sc         -      100      300       21,600     0.2 min
tot            -      300      500       36,000     0.3 min
got           10      300    4,200      192,960     1.5 min
got            5      300    2,700      113,760     0.9 min
got            3      300    2,100       82,080     0.6 min
```

It runs the **real Graph of Operations on the free mock backend**, so prompt and batch
counts are *exact*; only throughput is modelled. Add `--gpu-hour-rate` for a money column.

Cost knobs, in order of impact: `--aggregation-attempts` (dominant, it is the
`k_a log2 m` term), then `--limit`, then `--branching-factor`, then input length.

> A full 100-instance, 5-scheme run on 64-element sorting is **single-digit minutes of
> GPU compute**. On a cluster the allocation overhead (model load, queue, node hold) will
> dominate the bill, so batch many configurations into one job rather than submitting
> many short ones.

---

## Sample data

There are no fixed benchmark files to download — the GoT paper **generates** its
sorting, set-intersection and keyword-counting data, because the tasks are synthetic by
construction. `scripts/generate_data.py` does the same, reproducibly:

```bash
python scripts/generate_data.py --out data --seed 42 --n-samples 100 --small
```

| Dataset | Sizes | Matches paper |
|---|---|---|
| `data/sorting/` | 32, 64, 128 | digits 0–9 with duplicates (§5.1) |
| `data/set_intersection/` | 32, 64, 128 | 25–75% overlap (§5.2) |
| `data/keyword_counting/` | 4, 8, 16 sentences | country mentions (§5.3) |
| `data/document_merging/` | 4 documents | overlapping NDAs (§5.4) |
| `data/smoke/` | 5 instances each | fast local validation |

Every file ships with the known-correct answer, so ground-truth accuracy is computable
without an oracle. A fixed `--seed` gives byte-identical files, which matters when
comparing laptop and HPC runs.

**To validate the code locally**, start with `data/smoke/` and `--backend mock`: it runs
in under a second and exercises the whole pipeline.

---

## Project layout

```
├── explanation.md              ← deep dive on all 4 papers (start here)
├── README.md
│
├── got/                        ← the framework
│   ├── thought.py              Thought = a vertex (§3.1)
│   ├── operations.py           Generate / Aggregate / Improve / Score / KeepBest (§3.2–3.3)
│   ├── controller.py           executes a GoO in topological order (§4.4)
│   ├── prompter.py             Prompter + Parser base classes (§4.1–4.2)
│   ├── metrics.py              volume & latency (§6)
│   ├── backends/
│   │   ├── base.py             AbstractLanguageModel + token accounting
│   │   ├── mock.py             free, deterministic, fallible fake LLM
│   │   └── local_models.py     LlamaCppLM / HFLM / VLLMLM
│   └── tasks/
│       ├── sorting/            §5.1 — fully implemented, 5 schemes
│       └── set_intersection/   §5.2 — fully implemented, 5 schemes
│
├── scripts/
│   ├── generate_data.py        build the datasets
│   ├── run_direct.sh           run on a plain GPU server (no scheduler, no root)
│   ├── run_benchmark.py        main experiment runner
│   ├── visualize_graph.py      draw reasoning graphs + latency/volume plot
│   ├── download_local_model.sh fetch a small GGUF model
│   └── slurm/                  HPC setup + batch scripts
│
├── data/                       generated datasets
├── tests/                      46 tests
└── docs/figures/               generated plots
```

---

## How to add a new task

The framework is task-agnostic. Adding a task means writing **two classes and one
graph builder** — you never touch the core:

1. **Prompter** — `build(name, states)` returns the prompt text for each step.
   Return `None` to mark a step as local (no LLM call).
2. **Parser** — `parse(name, states, raw)` turns raw model output into a thought state.
   Be defensive: set `valid=False` rather than raising.
3. **Graph builder** — wire `Operation` objects with `add_predecessor()` to form the GoO.

```python
from got import Controller, InputOp, Generate, Aggregate, Score, KeepBest
from got.backends import get_backend

root = InputOp({"current": data})
gen  = Generate("solve", branching_factor=3); gen.add_predecessor(root)
sc   = Score(scoring_fn=my_scorer);           sc.add_predecessor(gen)
keep = KeepBest(n=1);                         keep.add_predecessor(sc)

lm = get_backend("mock")
ctrl = Controller(lm, MyPrompter(), MyParser(), [keep])
ctrl.run()
print(ctrl.best_thought().state)
```

See `got/tasks/sorting/` as the worked reference.

---

## Results so far

Measured structure on a 32-element sorting problem (mock backend):

| Scheme | thoughts | aggregations | volume | latency | mean error |
|---|---|---|---|---|---|
| IO | 2 | 0 | 1 | 1 | 4.97 |
| CoT | 3 | 0 | 2 | 2 | 2.97 |
| CoT-SC | 7 | 0 | 2 | 2 | 3.67 |
| ToT | 9 | 0 | 6 | 6 | 2.93 |
| **GoT** | **39** | **15** | **18** | **7** | **1.90** |

Set intersection (32 elements, 10 instances): GoT reduces error **3.90 → 1.10 (−72%)**
versus the IO baseline, with accuracy **0% → 40%**.

This reproduces the paper's **qualitative** claims: GoT is the only scheme that
aggregates, it achieves the highest volume per unit latency, and it produces the lowest
error.

**⚠️ What is *not* yet established:** absolute quality numbers comparable to the paper
(the "62% over ToT" claim) require a real LLM. The mock backend validates control flow,
not linguistic ability. Our GoT also uses *more* tokens than our ToT baseline, whereas
the paper reports a cost *reduction* — because our ToT baseline is deliberately lean.
A fair cost comparison needs the paper's wider ToT configuration on HPC.

See [explanation.md §16](explanation.md#16-what-i-verified-and-what-i-did-not) for the
full honest accounting.

---

## Status

| Component | State |
|---|---|
| Core framework (thoughts, operations, controller, metrics) | ✅ complete, tested |
| Backends (mock / llama.cpp / HF / vLLM) | ✅ complete |
| Sorting task, all 5 schemes | ✅ complete |
| Set intersection task, all 5 schemes | ✅ complete |
| Datasets (all 4 tasks) | ✅ generated |
| Visualisation | ✅ complete |
| Plain-GPU-server runner (`run_direct.sh`) | ✅ complete — no scheduler or root needed |
| HPC / SLURM scripts | ✅ written — **needs cluster-specific edits** |
| Keyword counting GoO | ⬜ data + mock ready, builder not written |
| Document merging GoO | ⬜ data ready, needs LLM-based scoring |
| Real-model HPC runs | ⬜ **the main remaining work** |

---

## Environment

Built and tested on WSL2 Ubuntu, `conda activate BTP`, Python 3.11.16.

| Package | Version |
|---|---|
| torch | 2.14.0+cpu |
| transformers | 5.17.0 |
| llama-cpp-python | 0.3.35 |
| numpy / pandas / matplotlib / networkx | latest |

```bash
pip install -e .            # core
pip install -e ".[local]"   # + llama.cpp for local GGUF
pip install -e ".[hpc]"     # + vLLM, CUDA torch, bitsandbytes
```

---

## References

1. Besta, M. et al. **Graph of Thoughts.** AAAI 2024. [arXiv:2308.09687](https://arxiv.org/abs/2308.09687)
2. Yao, S. et al. **Tree of Thoughts.** NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
3. Wei, J. et al. **Chain-of-Thought Prompting.** NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
4. Zhang, Z. et al. **Multimodal Chain-of-Thought Reasoning.** TMLR 2024. [arXiv:2302.00923](https://arxiv.org/abs/2302.00923)
