"""Human Rating Evaluation Harness.

Provides a three-phase workflow for human evaluation of benchmark trials:

1. **sample_and_blind** — Samples trials from automated evaluation results,
   blinds them for unbiased rating, and produces a rating queue + answer key.
2. **rate** — Interactive CLI for rating blinded items on a 1–5 scale.
3. **compile_ratings** — Joins ratings with the answer key and computes
   agreement and consistency metrics.

Run each phase independently:
    python -m benchmarks.human_rating.sample_and_blind ...
    python -m benchmarks.human_rating.rate ...
    python -m benchmarks.human_rating.compile_ratings ...

The rating command never imports or receives the answer-key path.
"""
