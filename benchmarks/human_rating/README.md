# Human Rating Evaluation

A standalone utility for blinded human evaluation of GPT-4o personalization quality across four memory methods. Validates automated judge scores through independent human assessment.

## Purpose

This package independently validates the GPT-5.4 automated integration scores by collecting blinded human ratings on the same response quality dimension. It produces human-judge agreement metrics suitable for research reporting.

## Frozen Protocol: `mixed_personalization_source_grounded_v1`

| Parameter | Value |
|-----------|-------|
| Methods | naive_concat, vanilla_rag, mem0_default, kernel_shared |
| Unique trials per method | 6 |
| Total unique trials | 24 |
| Duplicate appearances | 6 (for intra-rater consistency) |
| Total rating items | 30 |
| Scoring dimension | Integration (1–5 rubric) |
| Human raters | 1 |
| Question stratification | None (mixed prioritization queries) |
| Displayed context | Synthetic source profile + task (not exact model-visible) |
| Comparison score | integration_score from GPT-5.4 judge |

## Directory Structure

```
benchmarks/human_rating/
├── config/
│   └── evaluation_manifest.json    # Frozen protocol parameters and seeds
├── tests/
│   ├── run_all.py                  # Single command to run all tests
│   ├── test_schemas.py
│   ├── test_context_formatter.py
│   ├── test_trial_loader.py
│   ├── test_sampling.py
│   ├── test_blinding.py
│   ├── test_artifact_writer.py
│   ├── test_rate.py
│   ├── test_compile_ratings.py
│   ├── test_artifact_audit.py
│   └── test_end_to_end.py
├── runs/                           # Generated runtime artifacts (gitignored)
├── __init__.py
├── artifact_audit.py               # Compatibility audit for existing results
├── artifact_writer.py              # Queue + answer key generation
├── blinding.py                     # Duplicate selection and ID assignment
├── compile_ratings.py              # Private compilation and analysis
├── context_formatter.py            # Reference context formatting
├── paths.py                        # Directory layout helpers
├── rate.py                         # Interactive rating CLI
├── rubric.py                       # Frozen 1–5 rubric
├── sample_and_blind.py             # Sampling and generation CLI
├── sampling.py                     # Deterministic stratified sampling
├── schemas.py                      # Data contracts
├── trial_loader.py                 # Result file loading
├── validation.py                   # Structural validators
├── PROTOCOL.md                     # Protocol design documentation
├── RUNBOOK.md                      # Operational execution guide
├── PAPER_REPORTING.md              # Research reporting guidance
└── README.md                       # This file
```

## Prerequisites

- Python 3.10 or 3.11
- Repository working directory with `benchmarks/` and `results/` accessible
- Finalized GPT-4o result files at:
  - `results/gpt4o_naive_concat/results_naive_concat.json`
  - `results/gpt4o_vanilla_rag/results_vanilla_rag.json`
  - `results/gpt4o_mem0_default/results_mem0_default.json`
  - `results/gpt4o_kernel_shared/results_kernel_shared.json`
- No need to rerun the LLM benchmark or start the AIOS kernel

## Step 1: Run Tests

```bash
python benchmarks/human_rating/tests/run_all.py
```

All 10 test suites must pass before proceeding.

## Step 2: Generate Artifacts

```bash
python -m benchmarks.human_rating.sample_and_blind \
    --generate \
    --run-id human-rating-v1
```

Produces:
- `benchmarks/human_rating/runs/human-rating-v1/rater/rating_queue.json` (30 blinded items)
- `benchmarks/human_rating/runs/human-rating-v1/private/answer_key.json` (private)

Preview without writing:
```bash
python -m benchmarks.human_rating.sample_and_blind --preview-selection
python -m benchmarks.human_rating.sample_and_blind --preview-blinded-plan
```

## Step 3: Rate Responses

```bash
python -m benchmarks.human_rating.rate \
    --queue benchmarks/human_rating/runs/human-rating-v1/rater/rating_queue.json \
    --rater-id rater-01
```

Interactive commands:
- `1`–`5`: Submit rating
- `h`: Show rubric
- `n`: Add/clear note
- `f`: Toggle flag
- `q`: Quit (progress saved)

The session is crash-safe and resumable. Run the same command to continue.

## Step 4: Compile Ratings

```bash
python -m benchmarks.human_rating.compile_ratings \
    --queue benchmarks/human_rating/runs/human-rating-v1/rater/rating_queue.json \
    --ratings benchmarks/human_rating/runs/human-rating-v1/rater/ratings.jsonl \
    --session benchmarks/human_rating/runs/human-rating-v1/rater/rating_session.json \
    --answer-key benchmarks/human_rating/runs/human-rating-v1/private/answer_key.json \
    --output-dir benchmarks/human_rating/runs/human-rating-v1/private/compiled
```

## Step 5: Inspect Outputs

| File | Content |
|------|---------|
| `summary.json` | Overall metrics, method summaries, duplicate consistency, limitations |
| `primary_items.csv` | 24 unblinded items with human-judge agreement |
| `all_appearances.csv` | 30 items including duplicates |
| `method_summary.csv` | Per-method aggregates (4 rows) |
| `duplicate_consistency.csv` | Intra-rater reliability (6 pairs) |
| `confusion_matrix.csv` | 5×5 human vs judge counts |

## Reproducibility

Record before starting:
- Git commit hash
- Protocol name and version
- Sampling seed (20260715) and blinding seed (20260716)
- Rater ID
- Rating start/completion timestamps

## Additional Documentation

- **PROTOCOL.md** — Full protocol design and limitations
- **RUNBOOK.md** — Safe execution and recovery procedures
- **PAPER_REPORTING.md** — Research reporting guidance and required limitations
