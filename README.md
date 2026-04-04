# Fintech Python LLM Coding Benchmark

**Course:** CSE594A — Agentic and RAG-based Systems  
**Assignment:** Project 2 — Domain-Specific LLM Coding Benchmark + Multi-Agent System

---

## Overview

This project constructs a **domain-specific coding benchmark** for **Python in the Fintech domain**, evaluates three LLMs on it using zero-shot prompting, and then builds a **Multi-Agent System (MAS)** with a ReAct loop to improve the weakest model's performance.

---

## Domain & Language

| Attribute | Selection |
|---|---|
| **Language** | Python |
| **Domain** | Financial Technology (Fintech) |
| **Sub-domains** | Quantitative Finance, Technical Analysis, Portfolio Management, Derivatives Pricing |

---

## Benchmark Construction

### Source Repositories

Tasks were collected from 9 well-known open-source Python financial libraries on GitHub:

| Repository | Domain |
|---|---|
| `twopirllc/pandas-ta` | Technical Indicators |
| `lballabio/quantlib` | Derivatives & Fixed Income |
| `jpmorganchase/QuantiPy` | Portfolio Analytics |
| `pandas-dev/pandas` | Financial Data Processing |
| `scipy/scipy` | Statistical Finance |
| `numpy/numpy-financial` | Financial Functions |
| `matplotlib/mplfinance` | Trading Signals |
| `statsmodels/statsmodels` | Econometrics |
| `quantopian/zipline` | Backtesting |

### Tasks (25 Total)

Each benchmark task includes:
- A unique **task ID** (FT_01 – FT_25)
- A **natural-language prompt** explaining the function's original reason for existence
- **Reference code** drawn from the actual repository implementation

| ID | Task | Sub-domain |
|---|---|---|
| FT_01 | Relative Strength Index (RSI) | Technical Analysis |
| FT_02 | Black-Scholes Call Price | Derivatives |
| FT_03 | Option Gamma | Derivatives (Greeks) |
| FT_04 | Sharpe Ratio | Portfolio Management |
| FT_05 | Log Returns | Time Series |
| FT_06 | Bollinger Bands | Technical Analysis |
| FT_07 | Value at Risk (VaR) | Risk Management |
| FT_08 | Moving Average Crossover (Golden Cross) | Trading Signals |
| FT_09 | Internal Rate of Return (IRR) | Corporate Finance |
| FT_10 | Maximum Drawdown (MDD) | Risk Management |
| FT_11 | Implied Volatility (Newton-Raphson) | Derivatives |
| FT_12 | Beta Calculation | Portfolio Analysis |
| FT_13 | GARCH(1,1) Volatility Simulation | Econometrics |
| FT_14 | Sortino Ratio | Portfolio Management |
| FT_15 | Adjusted Close for Stock Splits | Data Adjustment |
| FT_16 | PCA for Yield Curve | Fixed Income |
| FT_17 | Macaulay Bond Duration | Fixed Income |
| FT_18 | Standardized Returns | Statistics |
| FT_19 | Spearman Correlation Matrix | Portfolio Analytics |
| FT_20 | Information Ratio | Portfolio Management |
| FT_21 | Average True Range (ATR) | Technical Analysis |
| FT_22 | Rolling Correlation | Statistical Analysis |
| FT_23 | Portfolio Variance (2-asset) | Portfolio Theory |
| FT_24 | Option Vega | Derivatives (Greeks) |
| FT_25 | Minimum Variance Weights | Portfolio Optimization |

---

## Extended CodeBLEU Score: Domain-Weighted Similarity (DWS)

### Motivation

Standard CodeBLEU evaluates general code quality. For Fintech Python, certain libraries and functions carry disproportionate semantic importance. We extend the metric with **Domain-Weighted Similarity (DWS)**, which augments structural matching with a domain-keyword accuracy component.

### DWS Formula

```
DWS = 0.60 × structural_similarity + 0.40 × keyword_accuracy
```

**Components:**

1. **Structural Similarity (60%):** Gestalt Pattern Matching (`difflib.SequenceMatcher`) between the reference and generated code, capturing token-level structural alignment.

2. **Keyword Accuracy (40%):** Precision score measuring whether all domain-critical keywords present in the reference also appear in the generated code.

**Fintech Domain Keywords:**

```python
["numpy", "pandas", "norm", "cdf", "rolling", "std", "mean", "np", "pd", "scipy"]
```

These keywords correspond to the core numerical and statistical libraries that most correct Fintech Python solution use. Omitting them typically indicates an incorrect or incomplete implementation.

### Rationale for Weights

- The **60/40 split** prioritises code structure (algorithm correctness) while giving significant weight to domain vocabulary (library correctness). A syntactically valid function implementing the wrong formula scores poorly on keyword match, which appropriately penalises it.

---

## Evaluation Methodology

### Models Evaluated

| Role | Model | Provider |
|---|---|---|
| Baseline A | `gemini-2.0-flash` | Google (direct API) |
| Baseline B | `meta/llama-3.1-70b-instruct` | NVIDIA NIM (Free Tier) |
| Baseline C | `qwen/qwen2.5-coder-32b-instruct` | NVIDIA NIM (Free Tier) |
| MAS Orchestrator | `gemini-2.0-flash` | Google |
| MAS Programmer | `meta/llama-3.1-70b-instruct` | NVIDIA NIM |
| MAS Critic/Judge | `qwen/qwen2.5-coder-32b-instruct` | NVIDIA NIM |

### Pass@1 Computation

Each task was prompted **10 times** with `temperature=0.7` (variance in responses). A generation is counted as a **pass** if it executes without error in a sandboxed Python environment pre-loaded with `numpy`, `pandas`, `scipy.stats`, `numpy-financial`, `scikit-learn`, `statsmodels`, and `yfinance`.

```
Pass@1 (per task) = (successful executions / 10) × 100%
```

---

## Zero-Shot Benchmark Results

### Per-Model Aggregate Scores

| Model | Avg Pass@1 | DWS (Extended CodeBLEU) |
|---|---|---|
| `gemini-2.0-flash` | **86.40%** | **0.1884** |
| `meta/llama-3.1-70b-instruct` | **81.60%** | **0.1987** |
| `qwen/qwen2.5-coder-32b-instruct` | **96.00%** | **0.2659** |

### Notable Observations

- **Qwen** was the strongest zero-shot model at 96.00% Pass@1 and the highest DWS score of **0.2659**, consistently generating concise, library-idiomatic code closest to the reference implementations.
- **Llama** was identified as the **weakest model** (81.60% Pass@1, DWS 0.1987) and was selected as the programmer agent in the MAS to demonstrate improvement.

---

## Multi-Agent System (MAS) with ReAct

### Architecture

The MAS follows the **Agent-as-a-Judge** framework described in [arXiv:2408.08927](https://arxiv.org/abs/2408.08927), implemented as a three-agent pipeline:

```
[Orchestrator: Gemini Flash]
        │
        ▼  (task dispatch)
[Programmer Agent: Llama-3.1-70B] ──► (generates code)
        │
        ▼  (execute locally)
[Executor: Python subprocess]
        │
        ├──► (SUCCESS) ──► return code
        │
        └──► (FAILURE) ──► error logs
                  │
                  ▼
        [Critic/Judge Agent: Qwen-2.5-Coder-32B]
                  │
                  ▼  (1-sentence correction feedback)
        [Programmer Agent: Llama-3.1-70B] ──► (re-generates)
             (max 3 correction loops)
```

### Agent Roles

| Agent | Model | Role |
|---|---|---|
| **Orchestrator** | `gemini-2.0-flash` | Task dispatcher; drives the benchmark loop |
| **Programmer** | `meta/llama-3.1-70b-instruct` | Generates and corrects Python code |
| **Executor** | Python `subprocess` | Acts/Observes — runs code, captures errors |
| **Critic (Judge)** | `qwen/qwen2.5-coder-32b-instruct` | Reasons — identifies the mathematical/logical mismatch and provides correction feedback |

### ReAct Loop (per task, repeated 10× for Pass@1)

1. **ACT:** Llama generates code for the task (with prior Judge feedback if available)
2. **OBSERVE:** Executor runs the code; captures `stdout`/`stderr`
3. **REASON:** If failed → Qwen (Judge) analyses the error vs. original requirement and issues a 1-sentence correction directive
4. **RE-ACT:** Llama regenerates incorporating the feedback
5. Repeat up to **3 correction attempts** before recording failure

### MAS Results

| Metric | Llama Zero-Shot | MAS (Llama + Judge) | Improvement |
|---|---|---|---|
| **Avg Pass@1** | 81.60% | **87.20%** | **+5.6 pp** |
| **DWS Score** | 0.1987 | **0.1980** | −0.0007 (negligible) |

The MAS successfully improved the weakest model's Pass@1 benchmark score by **5.6 percentage points** (81.60% → 87.20%), validating that iterative Judge feedback corrects failure modes present in zero-shot generation. The DWS score remained essentially unchanged (0.1987 → 0.1980), indicating the MAS preserved code structure quality while recovering from runtime failures.

**MAS per-task comparison highlights:**

| Task ID | Llama Zero-Shot | MAS | Δ |
|---|---|---|---|
| FT_06 (Bollinger Bands) | 50% | 100% | +50% |
| FT_08 (Golden Cross) | 40% | 70% | +30% |
| FT_19 (Spearman Corr) | 20% | 40% | +20% |
| FT_13 (GARCH) | 90% | 10% | −80% (regression: Judge injected matplotlib) |
| FT_21 (ATR) | 100% | 100% | maintained |

> Note: FT_13 shows a regression in MAS results due to the Judge's suggestion introducing a `matplotlib` import, which caused subprocess execution to fail silently.

---

## File Structure

```
final/
├── fintech_benchmark.json   # 25-task benchmark (prompts + reference code)
├── gemini_zero_shot.py               # Script: Gemini 2.0 Flash zero-shot evaluation
├── nim_models_zero_shot.py           # Script: Llama & Qwen zero-shot evaluation via NIM
├── mas_react.py                      # Script: Multi-Agent ReAct system
├── final_calc.py                     # Script: DWS scoring & report generation
├── gemini_zero_shot_results.json     # Raw results: Gemini zero-shot
├── llama_zero_shot_results.json      # Raw results: Llama zero-shot
├── qwen_zero_shot_results.json       # Raw results: Qwen zero-shot
└── agentic_mas_results.json          # Raw results: MAS (Llama + Judge)
```

---

## Setup & Reproduction

### Prerequisites

```bash
pip install google-generativeai openai numpy pandas scipy scikit-learn numpy-financial statsmodels yfinance matplotlib
```

### Environment Variables

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export NVIDIA_API_KEY="your-nim-api-key"
```

### Running Zero-Shot Evaluations

```bash
# Gemini baseline (25 tasks × 10 prompts = 250 API calls)
python3 gemini_zero_shot.py

# Llama & Qwen baselines via NIM
python3 nim_models_zero_shot.py
```

### Running the MAS

```bash
# Multi-agent ReAct system (Llama programmer + Qwen judge)
python3 mas_react.py
```

### Computing DWS Scores

```bash
python3 final_calc.py
# Outputs: final_benchmark_comparison.json
```
