"""Schema tests for the compatible human-rating protocol.

Verifies:
- SourceTrial uses integration_score as judge_score
- SourceTrial requires is_exact_model_visible_context=False
- SourceTrial requires synthetic_source_context provenance
- RatingQueueItem cannot hold forbidden fields
- RatingRecord is single 1–5 score
- Answer key requires judge_score == integration_score
- Manifest validation

Run:
    python benchmarks/human_rating/tests/test_schemas.py
"""

import json
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.schemas import (
    SourceTrial, RatingQueueItem, RatingRecord, AnswerKeyItem, CompiledRecord,
    VALID_SOURCE_METHODS, EVALUATION_DIMENSION, UNIQUE_ITEMS_PER_METHOD,
)
from benchmarks.human_rating.validation import (
    validate_rating_queue, validate_answer_key, validate_rating_record,
    validate_manifest,
)


def _valid_source_trial(**overrides):
    defaults = dict(
        source_method="kernel_shared", original_trial_id="42", model="gpt-4o",
        evaluation_dimension="integration",
        reference_context="USER PROFILE\n  Name: Alice",
        reference_context_provenance="synthetic_source_context",
        is_exact_model_visible_context=False,
        question="What should I focus on?", response="I recommend...",
        judge_score=4, profile_usage_score=3, task_usage_score=5, integration_score=4,
    )
    defaults.update(overrides)
    return SourceTrial(**defaults)


# --- SourceTrial ---

def test_source_trial_valid():
    t = _valid_source_trial()
    assert t.judge_score == t.integration_score == 4
    print("  PASS: test_source_trial_valid")

def test_source_trial_judge_score_must_equal_integration():
    try:
        _valid_source_trial(judge_score=3, integration_score=4)
        assert False
    except ValueError as e:
        assert "must equal" in str(e)
    print("  PASS: test_source_trial_judge_score_must_equal_integration")

def test_source_trial_rejects_exact_context_true():
    try:
        _valid_source_trial(is_exact_model_visible_context=True)
        assert False
    except ValueError:
        pass
    print("  PASS: test_source_trial_rejects_exact_context_true")

def test_source_trial_rejects_wrong_provenance():
    try:
        _valid_source_trial(reference_context_provenance="stored")
        assert False
    except ValueError:
        pass
    print("  PASS: test_source_trial_rejects_wrong_provenance")

def test_source_trial_rejects_wrong_dimension():
    try:
        _valid_source_trial(evaluation_dimension="profile_usage")
        assert False
    except ValueError:
        pass
    print("  PASS: test_source_trial_rejects_wrong_dimension")

def test_source_trial_rejects_empty_context():
    try:
        _valid_source_trial(reference_context="")
        assert False
    except ValueError:
        pass
    print("  PASS: test_source_trial_rejects_empty_context")


# --- RatingQueueItem ---

def test_queue_item_uses_reference_context():
    item = RatingQueueItem(blinded_id="R01", reference_context="ctx", question="Q", response="R")
    assert item.reference_context == "ctx"
    print("  PASS: test_queue_item_uses_reference_context")

def test_queue_item_cannot_hold_method():
    try:
        RatingQueueItem(blinded_id="R01", reference_context="c", question="Q", response="R", source_method="x")
        assert False
    except TypeError:
        pass
    print("  PASS: test_queue_item_cannot_hold_method")


# --- RatingRecord ---

def test_rating_record_single_score():
    r = RatingRecord(blinded_id="R01", rating=4, note=None, flagged=False, rated_at="2026-07-14T10:00:00Z")
    assert r.rating == 4
    print("  PASS: test_rating_record_single_score")

def test_rating_record_invalid():
    try:
        RatingRecord(blinded_id="R01", rating=6, note=None, flagged=False, rated_at="t")
        assert False
    except ValueError:
        pass
    print("  PASS: test_rating_record_invalid")


# --- Manifest validation ---

def test_manifest_valid():
    with open(os.path.join(_project_root, "benchmarks/human_rating/config/evaluation_manifest.json")) as f:
        data = json.load(f)
    errors = validate_manifest(data)
    assert errors == [], f"Unexpected: {errors}"
    print("  PASS: test_manifest_valid")

def test_manifest_rejects_profile_task_stratification():
    data = {"schema_version": 1, "protocol": {
        "assistant_model": "gpt-4o", "judge_score_dimension": "integration",
        "unique_items_per_method": 6, "duplicate_count": 6, "total_rating_items": 30,
        "question_stratification": "profile/task", "is_exact_model_visible_context": False,
    }, "sources": {m: {"path": f"x/{m}.json"} for m in VALID_SOURCE_METHODS}}
    errors = validate_manifest(data)
    assert any("profile/task" in e for e in errors)
    print("  PASS: test_manifest_rejects_profile_task_stratification")

def test_manifest_rejects_exact_context_true():
    data = {"schema_version": 1, "protocol": {
        "assistant_model": "gpt-4o", "judge_score_dimension": "integration",
        "unique_items_per_method": 6, "duplicate_count": 6, "total_rating_items": 30,
        "question_stratification": None, "is_exact_model_visible_context": True,
    }, "sources": {m: {"path": f"x/{m}.json"} for m in VALID_SOURCE_METHODS}}
    errors = validate_manifest(data)
    assert any("exact" in e.lower() for e in errors)
    print("  PASS: test_manifest_rejects_exact_context_true")

def test_manifest_rejects_wrong_judge_dimension():
    data = {"schema_version": 1, "protocol": {
        "assistant_model": "gpt-4o", "judge_score_dimension": "profile_usage",
        "unique_items_per_method": 6, "duplicate_count": 6, "total_rating_items": 30,
        "question_stratification": None, "is_exact_model_visible_context": False,
    }, "sources": {m: {"path": f"x/{m}.json"} for m in VALID_SOURCE_METHODS}}
    errors = validate_manifest(data)
    assert any("judge_score_dimension" in e for e in errors)
    print("  PASS: test_manifest_rejects_wrong_judge_dimension")

def test_manifest_rejects_missing_method():
    data = {"schema_version": 1, "protocol": {
        "assistant_model": "gpt-4o", "judge_score_dimension": "integration",
        "unique_items_per_method": 6, "duplicate_count": 6, "total_rating_items": 30,
        "question_stratification": None, "is_exact_model_visible_context": False,
    }, "sources": {"naive_concat": {"path": "x.json"}}}
    errors = validate_manifest(data)
    assert any("Missing methods" in e for e in errors)
    print("  PASS: test_manifest_rejects_missing_method")


# --- Answer key validation ---

def _valid_answer_key():
    items = []
    methods = sorted(VALID_SOURCE_METHODS)
    for i in range(24):
        m = methods[i % 4]
        items.append({
            "blinded_id": f"R{i+1:02d}", "source_method": m, "model": "gpt-4o",
            "original_trial_id": str(i), "judge_score": 3, "judge_score_dimension": "integration",
            "profile_usage_score": 4, "task_usage_score": 2, "integration_score": 3,
            "duplicate_of": None,
        })
    for i in range(6):
        orig = items[i]
        items.append({
            "blinded_id": f"R{25+i:02d}", "source_method": orig["source_method"],
            "model": "gpt-4o", "original_trial_id": orig["original_trial_id"],
            "judge_score": orig["judge_score"], "judge_score_dimension": "integration",
            "profile_usage_score": orig["profile_usage_score"],
            "task_usage_score": orig["task_usage_score"],
            "integration_score": orig["integration_score"],
            "duplicate_of": orig["blinded_id"],
        })
    return {"schema_version": 1, "benchmark_name": "t", "sampling_seed": 1,
            "duplicate_seed": 2, "id_assignment_seed": 3, "presentation_seed": 4,
            "created_at": "t", "items": items}

def test_answer_key_valid():
    errors = validate_answer_key(_valid_answer_key())
    assert errors == [], f"Unexpected: {errors}"
    print("  PASS: test_answer_key_valid")

def test_answer_key_rejects_mismatched_judge_integration():
    data = _valid_answer_key()
    data["items"][0]["judge_score"] = 5  # != integration_score (3)
    errors = validate_answer_key(data)
    assert any("must equal" in e for e in errors)
    print("  PASS: test_answer_key_rejects_mismatched_judge_integration")


# --- Queue validation ---

def test_queue_rejects_inference_context_field():
    items = [{"blinded_id": "R01", "reference_context": "c", "question": "Q",
              "response": "R", "inference_context": "leak"}]
    data = {"schema_version": 1, "run_id": "t", "item_count": 1, "items": items}
    errors = validate_rating_queue(data)
    assert any("forbidden" in e or "unexpected" in e for e in errors)
    print("  PASS: test_queue_rejects_inference_context_field")


def main():
    print("=== Human Rating Schema Tests ===\n")
    test_source_trial_valid()
    test_source_trial_judge_score_must_equal_integration()
    test_source_trial_rejects_exact_context_true()
    test_source_trial_rejects_wrong_provenance()
    test_source_trial_rejects_wrong_dimension()
    test_source_trial_rejects_empty_context()
    print()
    test_queue_item_uses_reference_context()
    test_queue_item_cannot_hold_method()
    print()
    test_rating_record_single_score()
    test_rating_record_invalid()
    print()
    test_manifest_valid()
    test_manifest_rejects_profile_task_stratification()
    test_manifest_rejects_exact_context_true()
    test_manifest_rejects_wrong_judge_dimension()
    test_manifest_rejects_missing_method()
    print()
    test_answer_key_valid()
    test_answer_key_rejects_mismatched_judge_integration()
    print()
    test_queue_rejects_inference_context_field()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
