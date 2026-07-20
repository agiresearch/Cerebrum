"""Unit tests for extended Pydantic model fields.

Verifies that:
- TrialResult has method field with default ""
- TrialResult has retrieved_context_count field with default None
- ExperimentMetadata has methods_run field with default []
- ConditionResults.condition can be set to a method name
- Existing construction sites still work (construct without new fields)

Validates: Requirements 8.1, 8.2, 8.3, 10.4
"""

import sys
sys.path.insert(0, ".")

from benchmarks.shared_memory.models import (
    TrialResult,
    ExperimentMetadata,
    ConditionResults,
    ExperimentResults,
    ConditionSummary,
    SummaryStatistics,
    MemoryCounts,
)


def _make_zero_stats():
    return SummaryStatistics(mean=0.0, std=0.0, min=0.0, max=0.0)


def _make_condition_summary():
    s = _make_zero_stats()
    return ConditionSummary(
        profile_usage=s,
        task_usage=s,
        integration=s,
        latency=s,
        memory_total=s,
        memory_shared=s,
        memory_private=s,
        injected_memories=s,
        total_trials=1,
        failed_trials=0,
    )


def test_trial_result_has_method_field():
    """TrialResult must have a 'method' field."""
    result = TrialResult(trial_index=0, condition="phase1")
    assert hasattr(result, "method"), "TrialResult missing 'method' field"
    print("PASSED: TrialResult has 'method' field")


def test_trial_result_method_default_is_empty_string():
    """TrialResult.method must default to '' when not specified."""
    result = TrialResult(trial_index=0, condition="phase1")
    assert result.method == "", (
        f"Expected method='', got {result.method!r}"
    )
    print("PASSED: TrialResult.method defaults to ''")


def test_trial_result_method_can_be_set():
    """TrialResult.method can be set to any method name."""
    for method_name in ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]:
        result = TrialResult(trial_index=0, condition=method_name, method=method_name)
        assert result.method == method_name, (
            f"Expected method={method_name!r}, got {result.method!r}"
        )
    print("PASSED: TrialResult.method can be set to any method name")


def test_trial_result_has_retrieved_context_count_field():
    """TrialResult must have a 'retrieved_context_count' field."""
    result = TrialResult(trial_index=0, condition="phase1")
    assert hasattr(result, "retrieved_context_count"), (
        "TrialResult missing 'retrieved_context_count' field"
    )
    print("PASSED: TrialResult has 'retrieved_context_count' field")


def test_trial_result_retrieved_context_count_default_is_none():
    """TrialResult.retrieved_context_count must default to None when not specified."""
    result = TrialResult(trial_index=0, condition="phase1")
    assert result.retrieved_context_count is None, (
        f"Expected retrieved_context_count=None, got {result.retrieved_context_count!r}"
    )
    print("PASSED: TrialResult.retrieved_context_count defaults to None")


def test_trial_result_retrieved_context_count_can_be_set():
    """TrialResult.retrieved_context_count can be set to an integer."""
    result = TrialResult(trial_index=0, condition="naive_concat", retrieved_context_count=0)
    assert result.retrieved_context_count == 0, (
        f"Expected retrieved_context_count=0, got {result.retrieved_context_count!r}"
    )

    result2 = TrialResult(trial_index=1, condition="vanilla_rag", retrieved_context_count=3)
    assert result2.retrieved_context_count == 3, (
        f"Expected retrieved_context_count=3, got {result2.retrieved_context_count!r}"
    )
    print("PASSED: TrialResult.retrieved_context_count can be set to an integer")


def test_experiment_metadata_has_methods_run_field():
    """ExperimentMetadata must have a 'methods_run' field."""
    meta = ExperimentMetadata(
        trials_per_condition=10,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=["phase1"],
    )
    assert hasattr(meta, "methods_run"), "ExperimentMetadata missing 'methods_run' field"
    print("PASSED: ExperimentMetadata has 'methods_run' field")


def test_experiment_metadata_methods_run_default_is_empty_list():
    """ExperimentMetadata.methods_run must default to [] when not specified."""
    meta = ExperimentMetadata(
        trials_per_condition=10,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=["phase1"],
    )
    assert meta.methods_run == [], (
        f"Expected methods_run=[], got {meta.methods_run!r}"
    )
    print("PASSED: ExperimentMetadata.methods_run defaults to []")


def test_experiment_metadata_methods_run_can_be_set():
    """ExperimentMetadata.methods_run can be set to a list of method names."""
    methods = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]
    meta = ExperimentMetadata(
        trials_per_condition=10,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=methods,
        methods_run=methods,
    )
    assert meta.methods_run == methods, (
        f"Expected methods_run={methods!r}, got {meta.methods_run!r}"
    )
    print("PASSED: ExperimentMetadata.methods_run can be set to a list of method names")


def test_condition_results_condition_can_be_method_name():
    """ConditionResults.condition can be set to a method name (not just phase1/phase2)."""
    for method_name in ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]:
        trial = TrialResult(trial_index=0, condition=method_name, method=method_name)
        cond_result = ConditionResults(
            condition=method_name,
            trials=[trial],
            summary=_make_condition_summary(),
        )
        assert cond_result.condition == method_name, (
            f"Expected condition={method_name!r}, got {cond_result.condition!r}"
        )
    print("PASSED: ConditionResults.condition can be set to any method name")


def test_existing_trial_result_construction_still_works():
    """Existing TrialResult construction without new fields still works."""
    # Minimal construction (only required fields)
    result = TrialResult(trial_index=0, condition="phase1")
    assert result.trial_index == 0
    assert result.condition == "phase1"
    assert result.method == ""
    assert result.retrieved_context_count is None
    print("PASSED: existing TrialResult construction without new fields works")


def test_existing_experiment_metadata_construction_still_works():
    """Existing ExperimentMetadata construction without methods_run still works."""
    meta = ExperimentMetadata(
        trials_per_condition=5,
        timestamp="2024-06-01T12:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=["phase1", "phase2"],
    )
    assert meta.trials_per_condition == 5
    assert meta.conditions_run == ["phase1", "phase2"]
    assert meta.methods_run == []
    print("PASSED: existing ExperimentMetadata construction without methods_run works")


def test_trial_result_with_all_new_fields():
    """TrialResult can be constructed with all new fields populated."""
    result = TrialResult(
        trial_index=5,
        condition="naive_concat",
        method="naive_concat",
        retrieved_context_count=0,
        profile_usage_score=4,
        task_usage_score=3,
        integration_score=4,
        latency_seconds=2.5,
    )
    assert result.method == "naive_concat"
    assert result.retrieved_context_count == 0
    assert result.profile_usage_score == 4
    print("PASSED: TrialResult with all new fields populated")


def test_experiment_results_with_methods_run():
    """ExperimentResults can be constructed with methods_run populated."""
    methods = ["kernel_shared", "naive_concat"]
    meta = ExperimentMetadata(
        trials_per_condition=2,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=methods,
        methods_run=methods,
    )
    trial_ks = TrialResult(trial_index=0, condition="kernel_shared", method="kernel_shared")
    trial_nc = TrialResult(trial_index=0, condition="naive_concat", method="naive_concat")

    cond_ks = ConditionResults(
        condition="kernel_shared",
        trials=[trial_ks],
        summary=_make_condition_summary(),
    )
    cond_nc = ConditionResults(
        condition="naive_concat",
        trials=[trial_nc],
        summary=_make_condition_summary(),
    )

    experiment = ExperimentResults(
        experiment_metadata=meta,
        conditions=[cond_ks, cond_nc],
    )

    assert experiment.experiment_metadata.methods_run == methods
    assert len(experiment.conditions) == 2
    print("PASSED: ExperimentResults with methods_run populated")


if __name__ == "__main__":
    test_trial_result_has_method_field()
    test_trial_result_method_default_is_empty_string()
    test_trial_result_method_can_be_set()
    test_trial_result_has_retrieved_context_count_field()
    test_trial_result_retrieved_context_count_default_is_none()
    test_trial_result_retrieved_context_count_can_be_set()
    test_experiment_metadata_has_methods_run_field()
    test_experiment_metadata_methods_run_default_is_empty_list()
    test_experiment_metadata_methods_run_can_be_set()
    test_condition_results_condition_can_be_method_name()
    test_existing_trial_result_construction_still_works()
    test_existing_experiment_metadata_construction_still_works()
    test_trial_result_with_all_new_fields()
    test_experiment_results_with_methods_run()
    print("\nAll extended model tests passed.")
