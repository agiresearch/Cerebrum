# Human Rating Protocol: mixed_personalization_source_grounded_v1

## Design Overview

- **4 methods** × **6 unique trials** = **24 unique items**
- 24 unique items + **6 duplicates** = **30 total rating items**
- Single 1–5 human score per item using the integration rubric
- Primary automated comparison: `integration_score`

## Methods

| Method | Description |
|--------|-------------|
| naive_concat | Full profile + task context concatenated into the prompt |
| vanilla_rag | TF-IDF chunk retrieval from profile/task data |
| mem0_default | 3-agent pipeline with private memories (no cross-agent injection) |
| kernel_shared | 3-agent pipeline with shared memories (kernel auto-injects cross-agent) |

## No Profile/Task Stratification

All existing benchmark trials use a single vague prioritization query that jointly requires both profile and task information to answer well. There is no authoritative profile-only or task-only question category in the finalized results.

Sampling is therefore stratified by **method only** (6 trials per method), not by question type.

## Judge Score: integration_score

The primary automated judge score for human-versus-judge comparison is `integration_score`, which evaluates:

> How well the response seamlessly combines profile preferences and task context into a single grounded recommendation.

This is the most appropriate dimension because the benchmark questions are mixed profile/task prioritization questions. The other judge dimensions (`profile_usage_score`, `task_usage_score`) are retained in the answer key for secondary analysis.

## Displayed Context: Source-Grounded Reference Context

The rating interface displays a **Reference User and Task Context** constructed from the stored `synthetic_profile` and `synthetic_task_context` fields.

**This is NOT the exact context visible to the model at inference time.**

- For `naive_concat`: the model received a similarly-formatted version of this data.
- For `vanilla_rag`: the model received only a TF-IDF-selected subset.
- For `mem0_default`: the model received only its system prompt and user query (no shared context injected).
- For `kernel_shared`: the model received kernel-injected memories derived from this data.

The exact prompt text supplied to GPT-4o is not recorded in the finalized result files.

### Implications for Rating Interpretation

The human rater evaluates whether the response **appropriately uses the source profile and task information**, not whether it uses only the information demonstrably visible to the model. This means:

- A `kernel_shared` response that references profile details is rated positively even though the rater cannot confirm the model had access to those specific details.
- A `mem0_default` response cannot be penalized for failing to reference context that was not actually injected.

This limitation must be acknowledged in any paper reporting these results.

## Blinding

- The rater sees: reference context, question, response.
- The rater does NOT see: method, trial ID, judge scores, duplicate status.
- 6 duplicate items (same question+context+response shown twice under different IDs) measure intra-rater consistency.

## Answer Key Isolation

- The answer key lives in `private/` and is never accessible to the rating CLI.
- The rating CLI operates only on files in `rater/`.
- The rating CLI has no `--answer-key` argument.

## Limitations for Paper Reporting

1. **No exact model-visible context.** The displayed reference context is the source synthetic data, not the exact prompt text supplied to GPT-4o.
2. **No profile/task stratification.** All questions are mixed prioritization queries. Analysis cannot separate profile-specific from task-specific personalization quality.
3. **Independent trial generation.** The four methods were run with independently generated synthetic data (not `--method all`). Cross-method comparisons are aggregate, not per-scenario.
4. **Single evaluation dimension.** The human rates on integration quality only. Profile-usage and task-usage agreement with the automated judge is available as secondary analysis from the answer key but was not directly rated by the human.
