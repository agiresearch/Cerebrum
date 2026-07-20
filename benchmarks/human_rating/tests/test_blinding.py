"""Blinding and duplicate selection tests.

Run:
    python benchmarks/human_rating/tests/test_blinding.py
"""

import os
import random
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.blinding import (
    build_blinded_plan, validate_blinded_plan, PlannedRatingItem, BlindedPlan,
)
from benchmarks.human_rating.schemas import SourceTrial, VALID_SOURCE_METHODS


def _make_trial(method, trial_id, score=3):
    return SourceTrial(
        source_method=method, original_trial_id=str(trial_id), model="gpt-4o",
        evaluation_dimension="integration",
        reference_context=f"Ctx {method}/{trial_id}",
        reference_context_provenance="synthetic_source_context",
        is_exact_model_visible_context=False,
        question=f"Q{trial_id}", response=f"R{trial_id}",
        judge_score=score, profile_usage_score=score,
        task_usage_score=score, integration_score=score,
    )


def _make_24():
    """Create a valid 24-item selected pool (6 per method)."""
    trials = []
    for method in sorted(VALID_SOURCE_METHODS):
        for i in range(6):
            trials.append(_make_trial(method, i))
    return trials


def test_exactly_30_items():
    plan = build_blinded_plan(_make_24(), seed=42)
    assert len(plan.items) == 30
    print("  PASS: test_exactly_30_items")


def test_24_unique_identities():
    plan = build_blinded_plan(_make_24(), seed=42)
    identities = {(i.source_trial.source_method, i.source_trial.original_trial_id) for i in plan.items}
    assert len(identities) == 24
    print("  PASS: test_24_unique_identities")


def test_exactly_6_duplicates():
    plan = build_blinded_plan(_make_24(), seed=42)
    dups = [i for i in plan.items if i.duplicate_of_blinded_id is not None]
    assert len(dups) == 6
    print("  PASS: test_exactly_6_duplicates")


def test_duplicate_sources_distinct():
    plan = build_blinded_plan(_make_24(), seed=42)
    assert len(set(plan.duplicated_source_identities)) == 6
    print("  PASS: test_duplicate_sources_distinct")


def test_at_least_one_dup_per_method():
    plan = build_blinded_plan(_make_24(), seed=42)
    methods_with_dups = {m for m, _ in plan.duplicated_source_identities}
    assert methods_with_dups == VALID_SOURCE_METHODS
    print("  PASS: test_at_least_one_dup_per_method")


def test_no_method_more_than_two_dups():
    plan = build_blinded_plan(_make_24(), seed=42)
    counts: dict[str, int] = {}
    for m, _ in plan.duplicated_source_identities:
        counts[m] = counts.get(m, 0) + 1
    assert all(c <= 2 for c in counts.values())
    print("  PASS: test_no_method_more_than_two_dups")


def test_all_blinded_ids_unique():
    plan = build_blinded_plan(_make_24(), seed=42)
    ids = [i.blinded_id for i in plan.items]
    assert len(set(ids)) == 30
    print("  PASS: test_all_blinded_ids_unique")


def test_blinded_ids_opaque():
    plan = build_blinded_plan(_make_24(), seed=42)
    for item in plan.items:
        bid = item.blinded_id.lower()
        assert item.source_trial.source_method not in bid
        assert "naive" not in bid and "kernel" not in bid and "vanilla" not in bid
    print("  PASS: test_blinded_ids_opaque")


def test_original_and_duplicate_ids_differ():
    plan = build_blinded_plan(_make_24(), seed=42)
    dups = [i for i in plan.items if i.duplicate_of_blinded_id is not None]
    for dup in dups:
        assert dup.blinded_id != dup.duplicate_of_blinded_id
    print("  PASS: test_original_and_duplicate_ids_differ")


def test_duplicate_refs_point_to_originals():
    plan = build_blinded_plan(_make_24(), seed=42)
    originals = {i.blinded_id for i in plan.items if i.duplicate_of_blinded_id is None}
    dups = [i for i in plan.items if i.duplicate_of_blinded_id is not None]
    for dup in dups:
        assert dup.duplicate_of_blinded_id in originals
    print("  PASS: test_duplicate_refs_point_to_originals")


def test_no_dup_points_to_another_dup():
    plan = build_blinded_plan(_make_24(), seed=42)
    dup_ids = {i.blinded_id for i in plan.items if i.duplicate_of_blinded_id is not None}
    for item in plan.items:
        if item.duplicate_of_blinded_id is not None:
            assert item.duplicate_of_blinded_id not in dup_ids
    print("  PASS: test_no_dup_points_to_another_dup")


def test_original_and_duplicate_content_identical():
    plan = build_blinded_plan(_make_24(), seed=42)
    id_map = {i.blinded_id: i for i in plan.items}
    dups = [i for i in plan.items if i.duplicate_of_blinded_id is not None]
    for dup in dups:
        orig = id_map[dup.duplicate_of_blinded_id]
        assert dup.source_trial == orig.source_trial
    print("  PASS: test_original_and_duplicate_content_identical")


def test_minimum_duplicate_separation():
    plan = build_blinded_plan(_make_24(), seed=42, min_duplicate_separation=3)
    id_map = {i.blinded_id: i for i in plan.items}
    dups = [i for i in plan.items if i.duplicate_of_blinded_id is not None]
    for dup in dups:
        orig = id_map[dup.duplicate_of_blinded_id]
        dist = abs(dup.appearance_index - orig.appearance_index)
        assert dist >= 3, f"Distance {dist} < 3"
    print("  PASS: test_minimum_duplicate_separation")


def test_same_seed_same_plan():
    plan1 = build_blinded_plan(_make_24(), seed=99)
    plan2 = build_blinded_plan(_make_24(), seed=99)
    ids1 = [(i.blinded_id, i.appearance_index) for i in plan1.items]
    ids2 = [(i.blinded_id, i.appearance_index) for i in plan2.items]
    assert ids1 == ids2
    print("  PASS: test_same_seed_same_plan")


def test_shuffled_input_same_plan():
    trials = _make_24()
    shuffled = list(trials)
    random.Random(777).shuffle(shuffled)
    plan1 = build_blinded_plan(trials, seed=42)
    plan2 = build_blinded_plan(shuffled, seed=42)
    ids1 = [(i.blinded_id, i.appearance_index) for i in sorted(plan1.items, key=lambda x: x.appearance_index)]
    ids2 = [(i.blinded_id, i.appearance_index) for i in sorted(plan2.items, key=lambda x: x.appearance_index)]
    assert ids1 == ids2
    print("  PASS: test_shuffled_input_same_plan")


def test_different_seeds_different_plan():
    plan1 = build_blinded_plan(_make_24(), seed=1)
    plan2 = build_blinded_plan(_make_24(), seed=2)
    ids1 = [i.blinded_id for i in plan1.items]
    ids2 = [i.blinded_id for i in plan2.items]
    assert ids1 != ids2
    print("  PASS: test_different_seeds_different_plan")


def test_global_random_unchanged():
    random.seed(555)
    before = random.random()
    random.seed(555)
    build_blinded_plan(_make_24(), seed=42)
    after = random.random()
    assert before == after
    print("  PASS: test_global_random_unchanged")


def test_invalid_23_items_rejected():
    trials = _make_24()[:23]
    try:
        build_blinded_plan(trials, seed=42)
        assert False
    except ValueError as e:
        assert "24" in str(e)
    print("  PASS: test_invalid_23_items_rejected")


def test_invalid_25_items_rejected():
    """A 25-item input with all distinct identities fails for size."""
    trials = _make_24()
    # Add a genuinely new trial (distinct identity)
    trials.append(_make_trial("naive_concat", 99))
    try:
        build_blinded_plan(trials, seed=42)
        assert False
    except ValueError as e:
        assert "Expected exactly 24" in str(e)
        assert "got 25" in str(e)
    print("  PASS: test_invalid_25_items_rejected")


def test_duplicate_identity_in_input_rejected():
    """A 24-item input with a repeated composite identity fails for duplicate."""
    trials = list(_make_24())
    # Replace last trial with a copy of the first (same method + trial_id)
    # This keeps length at 24 but introduces a duplicate identity
    first = trials[0]
    duplicate = _make_trial(first.source_method, int(first.original_trial_id), score=2)
    trials[-1] = duplicate
    assert len(trials) == 24
    try:
        build_blinded_plan(trials, seed=42)
        assert False, "Should have raised ValueError for duplicate identity"
    except ValueError as e:
        assert "Duplicate" in str(e) and "identity" in str(e).lower()
        # Must NOT be a size error
        assert "Expected exactly" not in str(e)
    print("  PASS: test_duplicate_identity_in_input_rejected")


def test_validate_passes_for_valid_plan():
    plan = build_blinded_plan(_make_24(), seed=42)
    validate_blinded_plan(plan)  # Should not raise
    print("  PASS: test_validate_passes_for_valid_plan")


def test_real_results():
    """Real finalized results produce a valid 30-item plan."""
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(os.path.join(results_dir, "gpt4o_naive_concat")):
        print("  SKIP: test_real_results (result files not present)")
        return
    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials
    trials, _ = load_eligible_trials(results_root=results_dir, min_per_method=6)
    selected, _ = sample_unique_trials(trials, seed=20260715)
    plan = build_blinded_plan(selected, seed=20260716)
    validate_blinded_plan(plan)
    assert len(plan.items) == 30
    print("  PASS: test_real_results")


def main():
    print("=== Blinding Tests ===\n")
    test_exactly_30_items()
    test_24_unique_identities()
    test_exactly_6_duplicates()
    test_duplicate_sources_distinct()
    test_at_least_one_dup_per_method()
    test_no_method_more_than_two_dups()
    test_all_blinded_ids_unique()
    test_blinded_ids_opaque()
    test_original_and_duplicate_ids_differ()
    test_duplicate_refs_point_to_originals()
    test_no_dup_points_to_another_dup()
    test_original_and_duplicate_content_identical()
    test_minimum_duplicate_separation()
    test_same_seed_same_plan()
    test_shuffled_input_same_plan()
    test_different_seeds_different_plan()
    test_global_random_unchanged()
    test_invalid_23_items_rejected()
    test_invalid_25_items_rejected()
    test_duplicate_identity_in_input_rejected()
    test_validate_passes_for_valid_plan()
    test_real_results()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
