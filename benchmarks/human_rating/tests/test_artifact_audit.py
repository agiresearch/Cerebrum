"""Tests for the artifact audit module.

Verifies:
- Complete supported synthetic artifact
- Missing question type detection
- Missing exact context detection
- Multiple judge dimensions detected
- Failed trials counted correctly
- Missing model metadata
- Deterministic report output
- No writes to source files
- Correct overall compatibility decision
- Clear blocking-issue messages
- Manifest contract structure

Run:
    python benchmarks/human_rating/tests/test_artifact_audit.py
"""

import json
import os
import sys
import tempfile

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.artifact_audit import (
    ArtifactAuditReport,
    ArtifactCapabilities,
    audit_result_files,
    format_summary_table,
    _serialize_report,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_trial(trial_index, method, **overrides):
    """Create a trial with all standard fields."""
    defaults = {
        "trial_index": trial_index,
        "condition": method,
        "method": method,
        "profile_usage_score": 3,
        "task_usage_score": 4,
        "integration_score": 3,
        "memory_counts": {"total": 0, "shared": 0, "private": 0},
        "retrieved_context_count": 0,
        "latency_seconds": 5.0,
        "follow_up_query": f"Question {trial_index}",
        "assistant_response": f"Response {trial_index}",
        "synthetic_profile": {"user_name": f"User {trial_index}", "preferred_tools": ["Git"],
                              "preferred_language": "Python", "response_style": "concise"},
        "synthetic_task_context": {"current_project": f"Project {trial_index}",
                                   "active_experiment": f"Exp {trial_index}",
                                   "goals": ["G"], "blockers": [], "next_steps": ["S"]},
        "retrieval_log": None,
        "injection_diagnostics": None,
        "written_memories": [],
        "failed": False,
        "error_message": None,
    }
    defaults.update(overrides)
    return defaults


def _make_compatible_trial(trial_index, method, **overrides):
    """Create a trial with ALL human-rating-required fields."""
    t = _make_trial(trial_index, method, **overrides)
    t["question_type"] = "profile" if trial_index % 2 == 0 else "task"
    t["inference_context_text"] = f"Context for {trial_index}"
    t.update(overrides)
    return t


def _make_result_file(method, trials):
    return {
        "experiment_metadata": {"trials_per_condition": len(trials),
                                "timestamp": "2026-07-14T10:00:00",
                                "kernel_url": "http://localhost:8000",
                                "conditions_run": [method], "methods_run": [method]},
        "conditions": [{"condition": method, "trials": trials, "summary": {}}],
    }


def _write_methods(tmpdir, method_trials):
    """Write result files for given methods. method_trials: dict method->list of trial dicts."""
    for method, trials in method_trials.items():
        dir_path = os.path.join(tmpdir, f"gpt4o_{method}")
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"results_{method}.json")
        with open(path, "w") as f:
            json.dump(_make_result_file(method, trials), f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_complete_supported_artifact():
    """Result files with all required fields report as compatible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_compatible_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        assert report.supports_requested_design is True
        assert len(report.blocking_issues) == 0
        for cap in report.methods:
            assert cap.has_question_type is True
            assert cap.has_exact_inference_context is True
    print("  PASS: test_complete_supported_artifact")


def test_missing_question_type():
    """Artifacts without question_type report as incompatible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        # Standard trials WITHOUT question_type (like real results)
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        assert report.supports_requested_design is False
        for cap in report.methods:
            assert cap.has_question_type is False
        assert any("question type" in issue.lower() for issue in report.blocking_issues)
    print("  PASS: test_missing_question_type")


def test_missing_exact_context():
    """Artifacts without inference_context_text report context as unavailable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        for cap in report.methods:
            assert cap.has_exact_inference_context is False
            ctx_status = next(
                (fs for fs in cap.field_statuses if fs.field_name == "inference_context_text"), None
            )
            assert ctx_status is not None
            assert ctx_status.status == "unavailable"
    print("  PASS: test_missing_exact_context")


def test_judge_dimensions_detected():
    """All three judge dimensions are detected as stored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        for cap in report.methods:
            assert cap.has_profile_usage_score is True
            assert cap.has_task_usage_score is True
            assert cap.has_integration_score is True
    print("  PASS: test_judge_dimensions_detected")


def test_failed_trials_counted():
    """Failed trials are counted separately from successful."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trials = [_make_trial(i, "naive_concat") for i in range(10)]
        trials[3]["failed"] = True
        trials[7]["failed"] = True
        _write_methods(tmpdir, {"naive_concat": trials,
                                "vanilla_rag": [_make_trial(i, "vanilla_rag") for i in range(10)],
                                "mem0_default": [_make_trial(i, "mem0_default") for i in range(10)],
                                "kernel_shared": [_make_trial(i, "kernel_shared") for i in range(10)]})

        report = audit_result_files(tmpdir)
        nc_cap = next(c for c in report.methods if c.method == "naive_concat")
        assert nc_cap.trial_count == 10
        assert nc_cap.successful_trial_count == 8
    print("  PASS: test_failed_trials_counted")


def test_missing_model_metadata():
    """Missing model in metadata is flagged as ambiguous."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        for cap in report.methods:
            assert cap.has_model_metadata is False
            model_status = next(
                (fs for fs in cap.field_statuses if fs.field_name == "model"), None
            )
            assert model_status is not None
            assert model_status.status == "ambiguous"
    print("  PASS: test_missing_model_metadata")


def test_deterministic_output():
    """Same input produces same report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        r1 = audit_result_files(tmpdir)
        r2 = audit_result_files(tmpdir)
        assert _serialize_report(r1) == _serialize_report(r2)
    print("  PASS: test_deterministic_output")


def test_no_writes_to_source():
    """Audit does not modify source files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(5)] for m in methods}
        _write_methods(tmpdir, method_trials)

        # Record file sizes before
        sizes_before = {}
        for m in methods:
            p = os.path.join(tmpdir, f"gpt4o_{m}", f"results_{m}.json")
            sizes_before[p] = os.path.getsize(p)

        audit_result_files(tmpdir)

        # Verify sizes unchanged
        for p, size in sizes_before.items():
            assert os.path.getsize(p) == size
    print("  PASS: test_no_writes_to_source")


def test_overall_compatibility_false():
    """Standard results without question_type are not compatible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        assert report.supports_requested_design is False
    print("  PASS: test_overall_compatibility_false")


def test_blocking_issues_clear():
    """Blocking issues have clear descriptive messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        assert len(report.blocking_issues) > 0
        for issue in report.blocking_issues:
            assert len(issue) > 10  # Not empty/trivial
    print("  PASS: test_blocking_issues_clear")


def test_manifest_contract_structure():
    """Manifest contract defines required fields and provenance options."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(5)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        mc = report.manifest_contract
        assert "question_type" in mc.required_fields
        assert "display_context" in mc.required_fields
        assert "context_provenance" in mc.required_fields
        assert "stored" in mc.provenance_options
        assert "reconstructed" in mc.provenance_options
        assert "manually_verified" in mc.provenance_options
        assert "source_method" in mc.example_entry
    print("  PASS: test_manifest_contract_structure")


def test_summary_table_format():
    """Summary table is readable and contains method names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        methods = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]
        method_trials = {m: [_make_trial(i, m) for i in range(10)] for m in methods}
        _write_methods(tmpdir, method_trials)

        report = audit_result_files(tmpdir)
        table = format_summary_table(report)
        assert "kernel_shared" in table
        assert "naive_concat" in table
        assert "Missing" in table
        assert "Overall compatible: No" in table
    print("  PASS: test_summary_table_format")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def main():
    print("=== Artifact Audit Tests ===\n")
    test_complete_supported_artifact()
    test_missing_question_type()
    test_missing_exact_context()
    test_judge_dimensions_detected()
    test_failed_trials_counted()
    test_missing_model_metadata()
    test_deterministic_output()
    test_no_writes_to_source()
    test_overall_compatibility_false()
    test_blocking_issues_clear()
    test_manifest_contract_structure()
    test_summary_table_format()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
