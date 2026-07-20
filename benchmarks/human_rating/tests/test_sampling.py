"""Sampling tests for the human-rating evaluation workflow.

Verifies:
- Exactly six selections per method
- Exactly 24 total selections
- Same seed produces identical selections
- Shuffled input produces identical selection
- Different seeds normally produce different selections
- Changes to one method's pool do not alter other methods' selections
- Insufficient method pool raises a clear error
- Missing method raises a clear error
- Duplicate composite trial identity is rejected
- Selected objects preserve all source fields
- Global random state is not modified
- Real finalized results produce a valid 24-item selection

Run:
    python benchmarks/human_rating/tests/test_sampling.py
"""

import os
import random
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.sampling import (
    sample_unique_trials,
    derive_method_seed,
    stable_trial_key,
    SamplingSummary,
)
from benchmarks.human_rating.schemas import SourceTrial, VALID_SOURCE_METHODS


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------

def _make_trial(method: str, trial_id: str, score: int = 3) -> SourceTrial:
    return SourceTrial(
        source_method=method, original_trial_id=trial_id, model="gpt-4o",
        evaluation_dimension="integration",
        reference_context=f"Context for {method}/{trial_id}",
        reference_context_provenance="synthetic_source_context",
        is_exact_model_visible_context=False,
        question=f"Q {trial_id}", response=f"R {trial_id}",
        judge_score=score, profile_usage_score=score,
        task_usage_score=score, integration_score=score,
    )


def _make_pool(n_per_method: int = 20) -> list[SourceTrial]:
    """Create a pool with n_per_method trials per method."""
    pool = []
    for method in sorted(VALID_SOURCE_METHODS):
        for i in range(n_per_method):
            pool.append(_make_trial(method, str(i)))
    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exactly_six_per_method():
    """Each method gets exactly 6 selections."""
    pool = _make_pool(20)
    selected, summary = sample_unique_trials(pool, seed=42)
    for method in VALID_SOURCE_METHODS:
        assert summary.selected_count_by_method[method] == 6
    print("  PASS: test_exactly_six_per_method")


def test_exactly_24_total():
    """Total selection is exactly 24."""
    pool = _make_pool(20)
    selected, _ = sample_unique_trials(pool, seed=42)
    assert len(selected) == 24
    print("  PASS: test_exactly_24_total")


def test_same_seed_identical():
    """Same seed produces identical selections."""
    pool = _make_pool(20)
    s1, _ = sample_unique_trials(pool, seed=12345)
    s2, _ = sample_unique_trials(pool, seed=12345)
    ids1 = [(t.source_method, t.original_trial_id) for t in s1]
    ids2 = [(t.source_method, t.original_trial_id) for t in s2]
    assert ids1 == ids2
    print("  PASS: test_same_seed_identical")


def test_shuffled_input_same_result():
    """Shuffled input produces identical selection."""
    pool = _make_pool(20)
    shuffled = list(pool)
    random.Random(999).shuffle(shuffled)
    s1, _ = sample_unique_trials(pool, seed=42)
    s2, _ = sample_unique_trials(shuffled, seed=42)
    ids1 = [(t.source_method, t.original_trial_id) for t in s1]
    ids2 = [(t.source_method, t.original_trial_id) for t in s2]
    assert ids1 == ids2
    print("  PASS: test_shuffled_input_same_result")


def test_different_seeds_different_results():
    """Different seeds normally produce different selections."""
    pool = _make_pool(50)
    s1, _ = sample_unique_trials(pool, seed=1)
    s2, _ = sample_unique_trials(pool, seed=2)
    ids1 = {(t.source_method, t.original_trial_id) for t in s1}
    ids2 = {(t.source_method, t.original_trial_id) for t in s2}
    assert ids1 != ids2
    print("  PASS: test_different_seeds_different_results")


def test_method_isolation():
    """Changing one method's pool doesn't affect other methods' selections."""
    pool_a = _make_pool(20)
    # Add extra trials only to naive_concat
    pool_b = list(pool_a) + [_make_trial("naive_concat", str(i)) for i in range(20, 30)]

    s_a, _ = sample_unique_trials(pool_a, seed=42)
    s_b, _ = sample_unique_trials(pool_b, seed=42)

    # Non-naive_concat methods should have identical selections
    for method in ["vanilla_rag", "mem0_default", "kernel_shared"]:
        ids_a = [t.original_trial_id for t in s_a if t.source_method == method]
        ids_b = [t.original_trial_id for t in s_b if t.source_method == method]
        assert ids_a == ids_b, f"{method} selections changed when naive_concat pool changed"
    print("  PASS: test_method_isolation")


def test_insufficient_pool_raises():
    """Fewer than 6 trials for a method raises ValueError."""
    pool = []
    for method in sorted(VALID_SOURCE_METHODS):
        n = 3 if method == "kernel_shared" else 20
        for i in range(n):
            pool.append(_make_trial(method, str(i)))
    try:
        sample_unique_trials(pool, seed=42)
        assert False, "Should have raised"
    except ValueError as e:
        assert "kernel_shared" in str(e)
        assert "Insufficient" in str(e)
    print("  PASS: test_insufficient_pool_raises")


def test_missing_method_raises():
    """Missing method entirely raises ValueError."""
    pool = []
    for method in ["naive_concat", "vanilla_rag", "mem0_default"]:
        for i in range(20):
            pool.append(_make_trial(method, str(i)))
    try:
        sample_unique_trials(pool, seed=42)
        assert False, "Should have raised"
    except ValueError as e:
        assert "kernel_shared" in str(e)
    print("  PASS: test_missing_method_raises")


def test_duplicate_identity_rejected():
    """Duplicate (method, trial_id) raises ValueError."""
    pool = _make_pool(20)
    pool.append(_make_trial("naive_concat", "5"))  # duplicate
    try:
        sample_unique_trials(pool, seed=42)
        assert False, "Should have raised"
    except ValueError as e:
        assert "Duplicate" in str(e)
    print("  PASS: test_duplicate_identity_rejected")


def test_selected_preserve_fields():
    """Selected trials preserve all source fields exactly."""
    pool = _make_pool(20)
    selected, _ = sample_unique_trials(pool, seed=42)
    for trial in selected:
        # Find the original in the pool
        original = next(
            t for t in pool
            if t.source_method == trial.source_method
            and t.original_trial_id == trial.original_trial_id
        )
        assert trial.reference_context == original.reference_context
        assert trial.question == original.question
        assert trial.response == original.response
        assert trial.judge_score == original.judge_score
        assert trial.is_exact_model_visible_context is False
    print("  PASS: test_selected_preserve_fields")


def test_global_random_state_unchanged():
    """Sampling does not modify the global random state."""
    random.seed(777)
    before = random.random()
    random.seed(777)

    pool = _make_pool(20)
    sample_unique_trials(pool, seed=42)

    after = random.random()
    assert before == after
    print("  PASS: test_global_random_state_unchanged")


def test_real_results():
    """Real finalized results produce a valid 24-item selection."""
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(os.path.join(results_dir, "gpt4o_naive_concat")):
        print("  SKIP: test_real_results (result files not present)")
        return

    from benchmarks.human_rating.trial_loader import load_eligible_trials
    trials, _ = load_eligible_trials(results_root=results_dir, min_per_method=6)

    selected, summary = sample_unique_trials(trials, seed=20260715)
    assert len(selected) == 24
    for method in VALID_SOURCE_METHODS:
        assert summary.selected_count_by_method[method] == 6
    # All have valid scores
    for t in selected:
        assert t.judge_score == t.integration_score
        assert 1 <= t.judge_score <= 5
    print(f"  PASS: test_real_results (24 selected from {len(trials)} eligible)")


def test_derive_method_seed_deterministic():
    """derive_method_seed is deterministic."""
    s1 = derive_method_seed(42, "naive_concat")
    s2 = derive_method_seed(42, "naive_concat")
    assert s1 == s2
    # Different methods get different seeds
    s3 = derive_method_seed(42, "kernel_shared")
    assert s1 != s3
    print("  PASS: test_derive_method_seed_deterministic")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def main():
    print("=== Sampling Tests ===\n")
    test_exactly_six_per_method()
    test_exactly_24_total()
    test_same_seed_identical()
    test_shuffled_input_same_result()
    test_different_seeds_different_results()
    test_method_isolation()
    test_insufficient_pool_raises()
    test_missing_method_raises()
    test_duplicate_identity_rejected()
    test_selected_preserve_fields()
    test_global_random_state_unchanged()
    test_real_results()
    test_derive_method_seed_deterministic()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
