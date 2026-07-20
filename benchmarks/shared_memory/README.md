# Shared Memory Benchmark — Personalization Baselines

## Overview

Compares four personalization methods using the same synthetic trial data,
HybridJudge scoring, and qwen2.5:7b as the assistant model with GPT-5.4 as
the judge.

## Methods

| # | Method | Description | Kernel Config |
|---|--------|-------------|---------------|
| 1 | naive_concat | Full context concatenation into prompt | auto_inject: false, auto_extract: false |
| 2 | vanilla_rag | TF-IDF retrieval over context chunks | auto_inject: false, auto_extract: false |
| 3 | mem0_default | Mem0 client add/search | auto_inject: false, auto_extract: false |
| 4 | kernel_shared | AIOS kernel shared memory (auto_inject) | auto_inject: true, auto_extract: true |

## Run Commands

Run sequentially in this order. Restart the kernel between each method to
clear the memory store.

```bash
# ============================================================
# 1. Naive Concat
# ============================================================
# Kernel config: auto_inject: false, auto_extract: false
# Restart kernel, then run:
python benchmarks/shared_memory/run_evaluation.py --trials 150 --output results/naive_concat/ --method naive_concat --assistant-model "qwen2.5:7b:ollama" --judge-model "gpt-5.4:openai" --csv

# ============================================================
# 2. Vanilla RAG
# ============================================================
# Kernel config: auto_inject: false, auto_extract: false
# Restart kernel, then run:
python benchmarks/shared_memory/run_evaluation.py --trials 150 --output results/vanilla_rag/ --method vanilla_rag --assistant-model "qwen2.5:7b:ollama" --judge-model "gpt-5.4:openai" --csv

# ============================================================
# 3. Mem0 Default
# ============================================================
# Kernel config: auto_inject: false, auto_extract: false
# Restart kernel, then run:
python benchmarks/shared_memory/run_evaluation.py --trials 150 --output results/mem0_default/ --method mem0_default --assistant-model "qwen2.5:7b:ollama" --judge-model "gpt-5.4:openai" --csv

# ============================================================
# 4. Kernel Shared (AIOS shared memory system)
# ============================================================
# Kernel config: auto_inject: true, auto_extract: true
# Restart kernel, then run:
python benchmarks/shared_memory/run_evaluation.py --trials 150 --output results/kernel_shared/ --method kernel_shared --assistant-model "qwen2.5:7b:ollama" --judge-model "gpt-5.4:openai" --csv
```

## Kernel Config Changes

In the AIOS kernel's `config.yaml`, toggle these fields under `memory:`:

```yaml
# For methods 1-3 (naive_concat, vanilla_rag, mem0_default):
auto_extract: false
auto_inject: false

# For method 4 (kernel_shared):
auto_extract: true
auto_inject: true
```

All other config fields stay the same across all runs.

## Notes

- Assistant model: qwen2.5:7b via Ollama (the model being evaluated)
- Judge model: GPT-5.4 via Azure OpenAI (scores responses on 1-5 rubric)
- Restart the kernel between each method to clear the memory store
- Results are written to `results/<method_name>/`
- Each run takes approximately 80-90 minutes (150 trials × ~35s/trial)
