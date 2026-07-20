# Paper Reporting Guidance

Copy-ready factual guidance for reporting the human evaluation in a research paper.

## Study Design

- **24 unique trials**: 6 per method (naive_concat, vanilla_rag, mem0_default, kernel_shared)
- **6 duplicated appearances**: For intra-rater consistency measurement
- **30 total rating items**: Presented in blinded random order
- **1 human rater**: Single-rater design
- **Scoring**: Single 1–5 integration score per item
- **Comparison**: GPT-5.4 automated judge (integration dimension)
- **Model**: GPT-4o responses only

## Primary Analysis (24 items)

Report using the 24 original appearances only. Duplicates are excluded to avoid double-weighting.

Metrics to report:
- Exact agreement rate (human rating == judge score)
- Within-one agreement rate (|human - judge| ≤ 1)
- Mean absolute error
- Mean signed difference (positive = human rated higher)
- Per-method breakdown (6 items each)

## Reliability Analysis (6 pairs)

Report using the 6 duplicate pairs:
- Exact repeat consistency rate
- Within-one repeat consistency rate
- Mean absolute repeat difference
- Maximum absolute repeat difference

This measures **intra-rater repeatability**, not inter-rater agreement.

## Sensitivity Analysis (30 items)

Optionally report appearance-weighted metrics over all 30 items. Explicitly label as "appearance-weighted" and note that 6 trials are intentionally repeated.

## Required Limitations

The following must be stated in any paper using these results:

1. **Context is not exact model-visible.** The reference context shown to the rater is the deterministic synthetic source profile and task data. It is not necessarily identical to what GPT-4o received at inference time (which varies by method — naive_concat prepends all context, vanilla_rag selects chunks, kernel_shared injects memories, mem0_default injects nothing).

2. **No profile/task stratification.** The finalized benchmark results do not contain authoritative profile-only or task-only question categories. All trials use mixed prioritization queries. The evaluation therefore cannot report separate profile-usage or task-usage agreement.

3. **Single rater.** With one human rater, inter-rater reliability (Cohen's kappa, Krippendorff's alpha) cannot be computed. Duplicate consistency measures repeatability of the same rater, not agreement between raters.

4. **Integration dimension only.** The human rates on integration quality (how well the response combines profile and task context). Profile-usage and task-usage judge scores are available in the answer key for secondary analysis but were not directly rated by the human.

5. **Independent trial generation.** The four methods were run with independently generated synthetic scenarios (not `--method all`). Cross-method comparisons are aggregate, not per-scenario paired.

## Do Not Claim

- Inter-rater reliability or agreement (only one rater)
- Exact model-visible context was shown to the rater
- Profile-type or task-type questions were separately evaluated
- Paired per-scenario method comparisons
