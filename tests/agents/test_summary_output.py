"""Unit tests for comparative summary output from ResultsWriter.print_summary.

Verifies that:
- Multi-method run prints delta column to stdout
- Single-method run omits delta column
- Trial counts (total and failed) appear in output

Uses io.StringIO to capture stdout.

Validates: Requirements 9.1, 9.4, 9.5
"""

import sys
sys.path.insert(0, ".")

import io
from contextlib import redirect_stdout

from benchmarks.shared_memory.models import (
    ConditionResults,
    ConditionSummary,
    ExperimentMetadata,
    ExperimentResults,
    MemoryCounts,
    SummaryStatistics,
    TrialResult,
)
from benchmarks.shared_memory.results import ResultsWriter


def _make_stats(mean=3.0, std=0.5, min_val=1.0, max_val=5.0):
    return SummaryStatistics(mean=mean, std=std, min=min_val, max=max_val)


def _make_condition_summary(
    profile_mean=3.0,
    task_mean=3.0,
    integ_mean=3.0,
    latency_mean=1.0,
    total_trials=5,
    failed_trials=1,
):
    return ConditionSummary(
        profile_usage=_make_stats(mean=profile_mean),
        task_usage=_make_stats(mean=task_mean),
        integration=_make_stats(mean=integ_mean),
        latency=_make_stats(mean=latency_mean),
        memory_total=_make_stats(mean=0.0),
        memory_shared=_make_stats(mean=0.0),
        memory_private=_make_stats(mean=0.0),
        injected_memories=_make_stats(mean=0.0),
        total_trials=total_trials,
        failed_trials=failed_trials,
    )


def _make_trial_result(trial_index, condition, method):
    return TrialResult(
        trial_index=trial_index,
        condition=condition,
        method=method,
        profile_usage_score=3,
        task_usage_score=3,
        integration_score=3,
        memory_counts=MemoryCounts(total=0, shared=0, private=0),
        latency_seconds=1.0,
    )


def _make_single_method_experiment(method_name, total_trials=5, failed_trials=1):
    """Build a minimal ExperimentResults for a single method."""
    trials = [_make_trial_result(i, method_name, method_name) for i in range(total_trials)]
    # Mark some as failed
    for i in range(failed_trials):
        trials[i] = TrialResult(
            trial_index=i,
            condition=method_name,
            method=method_name,
            failed=True,
        )

    cond = ConditionResults(
        condition=method_name,
        trials=trials,
        summary=_make_condition_summary(
            total_trials=total_trials,
            failed_trials=failed_trials,
        ),
    )
    meta = ExperimentMetadata(
        trials_per_condition=total_trials,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=[method_name],
        methods_run=[method_name],
    )
    return ExperimentResults(experiment_metadata=meta, conditions=[cond])


def _make_multi_method_experiment(total_trials=5, failed_trials=0):
    """Build a minimal ExperimentResults for all four methods."""
    methods = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]
    conditions = []
    for i, method_name in enumerate(methods):
        trials = [_make_trial_result(j, method_name, method_name) for j in range(total_trials)]
        cond = ConditionResults(
            condition=method_name,
            trials=trials,
            summary=_make_condition_summary(
                profile_mean=3.0 + i * 0.1,
                task_mean=3.0 + i * 0.2,
                integ_mean=3.0 + i * 0.15,
                total_trials=total_trials,
                failed_trials=failed_trials,
            ),
        )
        conditions.append(cond)

    meta = ExperimentMetadata(
        trials_per_condition=total_trials,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=methods,
        methods_run=methods,
    )
    return ExperimentResults(experiment_metadata=meta, conditions=conditions)


def _capture_summary(experiment):
    """Run print_summary and capture stdout as a string."""
    writer = ResultsWriter(output_dir="/tmp", write_csv=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        writer.print_summary(experiment)
    return buf.getvalue()


def test_multi_method_prints_delta_column():
    """Multi-method run must include a delta column in the summary output."""
    experiment = _make_multi_method_experiment()
    output = _capture_summary(experiment)

    # Delta column header should appear
    assert "Delta" in output, (
        f"Expected 'Delta' in multi-method summary output, but not found.\n"
        f"Output:\n{output}"
    )
    print("PASSED: multi-method summary includes delta column")


def test_single_method_omits_delta_column():
    """Single-method run must NOT include a delta column in the summary output."""
    experiment = _make_single_method_experiment("naive_concat")
    output = _capture_summary(experiment)

    assert "Delta" not in output, (
        f"Expected no 'Delta' in single-method summary output, but found it.\n"
        f"Output:\n{output}"
    )
    print("PASSED: single-method summary omits delta column")


def test_multi_method_prints_all_method_names():
    """Multi-method summary must include all four method names."""
    experiment = _make_multi_method_experiment()
    output = _capture_summary(experiment)

    for method_name in ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]:
        assert method_name in output, (
            f"Expected method name '{method_name}' in summary output.\n"
            f"Output:\n{output}"
        )
    print("PASSED: multi-method summary includes all method names")


def test_single_method_prints_method_name():
    """Single-method summary must include the method name."""
    experiment = _make_single_method_experiment("vanilla_rag")
    output = _capture_summary(experiment)

    assert "vanilla_rag" in output, (
        f"Expected 'vanilla_rag' in single-method summary output.\n"
        f"Output:\n{output}"
    )
    print("PASSED: single-method summary includes method name")


def test_trial_counts_appear_in_output():
    """Summary output must include total and failed trial counts."""
    total = 5
    failed = 2
    experiment = _make_single_method_experiment("naive_concat", total_trials=total, failed_trials=failed)
    output = _capture_summary(experiment)

    assert str(total) in output, (
        f"Expected total trial count '{total}' in summary output.\n"
        f"Output:\n{output}"
    )
    assert str(failed) in output, (
        f"Expected failed trial count '{failed}' in summary output.\n"
        f"Output:\n{output}"
    )
    print(f"PASSED: trial counts (total={total}, failed={failed}) appear in output")


def test_multi_method_trial_counts_appear():
    """Multi-method summary must include trial counts for each method."""
    total = 10
    experiment = _make_multi_method_experiment(total_trials=total, failed_trials=0)
    output = _capture_summary(experiment)

    assert str(total) in output, (
        f"Expected total trial count '{total}' in multi-method summary output.\n"
        f"Output:\n{output}"
    )
    print(f"PASSED: trial counts (total={total}) appear in multi-method summary")


def test_summary_output_is_non_empty():
    """print_summary must produce non-empty output."""
    experiment = _make_single_method_experiment("kernel_shared")
    output = _capture_summary(experiment)

    assert len(output.strip()) > 0, "Expected non-empty summary output"
    print("PASSED: summary output is non-empty")


def test_multi_method_delta_values_are_numeric():
    """Delta values in multi-method summary must be numeric (contain + or - sign)."""
    experiment = _make_multi_method_experiment()
    output = _capture_summary(experiment)

    # Delta values should have + or - prefix (from the format string)
    has_delta_values = "+" in output or "-" in output
    assert has_delta_values, (
        f"Expected numeric delta values (with + or - sign) in multi-method summary.\n"
        f"Output:\n{output}"
    )
    print("PASSED: multi-method summary contains numeric delta values")


def test_single_method_no_delta_values():
    """Single-method summary must not contain delta column headers."""
    experiment = _make_single_method_experiment("mem0_default")
    output = _capture_summary(experiment)

    # Should not have DeltaProfile, DeltaTask, DeltaInteg headers
    assert "DeltaProfile" not in output, (
        f"Expected no 'DeltaProfile' in single-method summary.\nOutput:\n{output}"
    )
    assert "DeltaTask" not in output, (
        f"Expected no 'DeltaTask' in single-method summary.\nOutput:\n{output}"
    )
    assert "DeltaInteg" not in output, (
        f"Expected no 'DeltaInteg' in single-method summary.\nOutput:\n{output}"
    )
    print("PASSED: single-method summary has no delta column headers")


def test_multi_method_has_delta_column_headers():
    """Multi-method summary must contain delta column headers."""
    experiment = _make_multi_method_experiment()
    output = _capture_summary(experiment)

    # Should have DeltaProfile, DeltaTask, DeltaInteg headers
    assert "DeltaProfile" in output, (
        f"Expected 'DeltaProfile' in multi-method summary.\nOutput:\n{output}"
    )
    assert "DeltaTask" in output, (
        f"Expected 'DeltaTask' in multi-method summary.\nOutput:\n{output}"
    )
    assert "DeltaInteg" in output, (
        f"Expected 'DeltaInteg' in multi-method summary.\nOutput:\n{output}"
    )
    print("PASSED: multi-method summary has delta column headers")


if __name__ == "__main__":
    test_multi_method_prints_delta_column()
    test_single_method_omits_delta_column()
    test_multi_method_prints_all_method_names()
    test_single_method_prints_method_name()
    test_trial_counts_appear_in_output()
    test_multi_method_trial_counts_appear()
    test_summary_output_is_non_empty()
    test_multi_method_delta_values_are_numeric()
    test_single_method_no_delta_values()
    test_multi_method_has_delta_column_headers()
    print("\nAll summary output tests passed.")
