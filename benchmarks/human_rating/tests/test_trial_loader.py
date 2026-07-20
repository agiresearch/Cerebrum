"""Trial loader tests for the compatible human-rating protocol.

Verifies:
- integration_score used as judge_score
- Reference context formatted from synthetic data
- Missing synthetic data excluded
- Method-independent trial IDs
- Eligibility per method (min 6)
- No profile/task stratification required

Run:
    python benchmarks/human_rating/tests/test_trial_loader.py
"""

import json
import os
import sys
import tempfile

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.trial_loader import (
    ALLOWED_METHODS, ExclusionReport, load_result_file, load_eligible_trials,
    summarize_trial_pool, discover_result_files,
)
from benchmarks.human_rating.schemas import UNIQUE_ITEMS_PER_METHOD


def _make_trial(i, method, **overrides):
    defaults = {
        "trial_index": i, "condition": method, "method": method,
        "profile_usage_score": 3, "task_usage_score": 4, "integration_score": 4,
        "memory_counts": {"total": 0, "shared": 0, "private": 0},
        "retrieved_context_count": 0, "latency_seconds": 5.0,
        "follow_up_query": f"Question {i}", "assistant_response": f"Response {i}",
        "synthetic_profile": {"user_name": f"User {i}", "preferred_tools": ["Git"],
                              "preferred_language": "Python", "response_style": "concise"},
        "synthetic_task_context": {"current_project": f"Proj {i}", "active_experiment": f"Exp {i}",
                                   "goals": [f"Goal {i}"], "blockers": [], "next_steps": [f"Step {i}"]},
        "failed": False, "error_message": None,
    }
    defaults.update(overrides)
    return defaults


def _write(tmpdir, method, trials):
    d = os.path.join(tmpdir, f"gpt4o_{method}")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"results_{method}.json")
    data = {"experiment_metadata": {"trials_per_condition": len(trials), "timestamp": "t",
            "kernel_url": "http://localhost:8000", "conditions_run": [method], "methods_run": [method]},
            "conditions": [{"condition": method, "trials": trials, "summary": {}}]}
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def _write_all(tmpdir, n=10):
    return {m: _write(tmpdir, m, [_make_trial(i, m) for i in range(n)]) for m in ALLOWED_METHODS}


# --- Tests ---

def test_integration_score_as_judge():
    """judge_score equals integration_score for every loaded trial."""
    with tempfile.TemporaryDirectory() as d:
        _write_all(d)
        trials, _ = load_eligible_trials(results_root=d, min_per_method=6)
        for t in trials:
            assert t.judge_score == t.integration_score == 4
    print("  PASS: test_integration_score_as_judge")


def test_reference_context_from_synthetic():
    """Reference context is formatted from stored synthetic data."""
    with tempfile.TemporaryDirectory() as d:
        _write_all(d, n=7)
        trials, _ = load_eligible_trials(results_root=d, min_per_method=6)
        for t in trials:
            assert "USER PROFILE" in t.reference_context
            assert "TASK CONTEXT" in t.reference_context
            assert t.reference_context_provenance == "synthetic_source_context"
            assert t.is_exact_model_visible_context is False
    print("  PASS: test_reference_context_from_synthetic")


def test_missing_synthetic_excluded():
    """Trial without synthetic_profile/task is excluded."""
    with tempfile.TemporaryDirectory() as d:
        trials = [_make_trial(i, "naive_concat") for i in range(8)]
        trials[2]["synthetic_profile"] = None
        trials[2]["synthetic_task_context"] = None
        _write(d, "naive_concat", trials)
        report = ExclusionReport()
        loaded = load_result_file(
            os.path.join(d, "gpt4o_naive_concat", "results_naive_concat.json"),
            "naive_concat", report)
        assert len(loaded) == 7
        assert any(e.reason == "missing_synthetic_context" for e in report.entries)
    print("  PASS: test_missing_synthetic_excluded")


def test_method_independent_ids():
    """original_trial_id is just the index string."""
    with tempfile.TemporaryDirectory() as d:
        _write(d, "vanilla_rag", [_make_trial(i, "vanilla_rag") for i in range(3)])
        trials = load_result_file(
            os.path.join(d, "gpt4o_vanilla_rag", "results_vanilla_rag.json"), "vanilla_rag")
        assert [t.original_trial_id for t in trials] == ["0", "1", "2"]
    print("  PASS: test_method_independent_ids")


def test_eligibility_min_per_method():
    """Fewer than 6 trials per method raises ValueError."""
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for m in ALLOWED_METHODS:
            n = 3 if m == "kernel_shared" else 10
            paths[m] = _write(d, m, [_make_trial(i, m) for i in range(n)])
        try:
            load_eligible_trials(method_paths=paths, min_per_method=6)
            assert False
        except ValueError as e:
            assert "kernel_shared" in str(e)
    print("  PASS: test_eligibility_min_per_method")


def test_empty_response_excluded():
    """Empty assistant_response leads to exclusion."""
    with tempfile.TemporaryDirectory() as d:
        trials = [_make_trial(i, "mem0_default") for i in range(8)]
        trials[4]["assistant_response"] = ""
        _write(d, "mem0_default", trials)
        report = ExclusionReport()
        loaded = load_result_file(
            os.path.join(d, "gpt4o_mem0_default", "results_mem0_default.json"),
            "mem0_default", report)
        assert len(loaded) == 7
        assert report.entries[0].reason == "empty_response"
    print("  PASS: test_empty_response_excluded")


def test_pool_summary():
    """Summary shows per-method counts."""
    with tempfile.TemporaryDirectory() as d:
        _write_all(d, n=10)
        trials, _ = load_eligible_trials(results_root=d, min_per_method=6)
        summary = summarize_trial_pool(trials)
        assert summary.total_trials == 40
        for m in ALLOWED_METHODS:
            assert summary.method_counts[m] == 10
    print("  PASS: test_pool_summary")


def test_real_results():
    """Load the actual finalized results (integration test)."""
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(os.path.join(results_dir, "gpt4o_naive_concat")):
        print("  SKIP: test_real_results (result files not present)")
        return
    trials, report = load_eligible_trials(results_root=results_dir, min_per_method=6)
    assert len(trials) > 500
    for t in trials:
        assert t.judge_score == t.integration_score
        assert t.is_exact_model_visible_context is False
        assert t.evaluation_dimension == "integration"
    print(f"  PASS: test_real_results ({len(trials)} trials, {report.total_excluded} excluded)")


def main():
    print("=== Trial Loader Tests ===\n")
    test_integration_score_as_judge()
    test_reference_context_from_synthetic()
    test_missing_synthetic_excluded()
    test_method_independent_ids()
    test_eligibility_min_per_method()
    test_empty_response_excluded()
    test_pool_summary()
    test_real_results()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
