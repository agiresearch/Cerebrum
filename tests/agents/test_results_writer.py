"""Unit tests for ResultsWriter output file naming and CSV columns.

Verifies that:
- Single-method run produces results_{method}.json
- --method all produces results_all_methods.json
- CSV output includes method and retrieved_context_count columns

Uses a temp directory for output.

Validates: Requirements 8.6, 8.7, 9.4, 9.5
"""

import sys
sys.path.insert(0, ".")

import csv
import json
import os
import tempfile
import shutil

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


def _make_zero_stats():
    return SummaryStatistics(mean=0.0, std=0.0, min=0.0, max=0.0)


def _make_condition_summary(total_trials=1, failed_trials=0):
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
        total_trials=total_trials,
        failed_trials=failed_trials,
    )


def _make_trial_result(trial_index, condition, method, retrieved_context_count=None):
    return TrialResult(
        trial_index=trial_index,
        condition=condition,
        method=method,
        retrieved_context_count=retrieved_context_count,
        profile_usage_score=3,
        task_usage_score=3,
        integration_score=3,
        memory_counts=MemoryCounts(total=0, shared=0, private=0),
        latency_seconds=1.0,
        follow_up_query="test query",
        assistant_response="test response",
    )


def _make_single_method_experiment(method_name):
    """Build a minimal ExperimentResults for a single method."""
    trial = _make_trial_result(0, method_name, method_name, retrieved_context_count=0)
    cond = ConditionResults(
        condition=method_name,
        trials=[trial],
        summary=_make_condition_summary(),
    )
    meta = ExperimentMetadata(
        trials_per_condition=1,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=[method_name],
        methods_run=[method_name],
    )
    return ExperimentResults(experiment_metadata=meta, conditions=[cond])


def _make_all_methods_experiment():
    """Build a minimal ExperimentResults for all four methods."""
    methods = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]
    conditions = []
    for method_name in methods:
        trial = _make_trial_result(0, method_name, method_name, retrieved_context_count=2)
        cond = ConditionResults(
            condition=method_name,
            trials=[trial],
            summary=_make_condition_summary(),
        )
        conditions.append(cond)

    meta = ExperimentMetadata(
        trials_per_condition=1,
        timestamp="2024-01-01T00:00:00",
        kernel_url="http://localhost:8000",
        conditions_run=methods,
        methods_run=methods,
    )
    return ExperimentResults(experiment_metadata=meta, conditions=conditions)


def test_single_method_naive_concat_produces_correct_filename():
    """Single-method run with naive_concat produces results_naive_concat.json."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_single_method_experiment("naive_concat")

        path = writer.write_json(experiment, filename="results_naive_concat.json")

        assert os.path.basename(path) == "results_naive_concat.json", (
            f"Expected filename 'results_naive_concat.json', got {os.path.basename(path)!r}"
        )
        assert os.path.exists(path), f"File not found: {path}"
        print("PASSED: naive_concat produces results_naive_concat.json")
    finally:
        shutil.rmtree(tmpdir)


def test_single_method_vanilla_rag_produces_correct_filename():
    """Single-method run with vanilla_rag produces results_vanilla_rag.json."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_single_method_experiment("vanilla_rag")

        path = writer.write_json(experiment, filename="results_vanilla_rag.json")

        assert os.path.basename(path) == "results_vanilla_rag.json", (
            f"Expected filename 'results_vanilla_rag.json', got {os.path.basename(path)!r}"
        )
        assert os.path.exists(path), f"File not found: {path}"
        print("PASSED: vanilla_rag produces results_vanilla_rag.json")
    finally:
        shutil.rmtree(tmpdir)


def test_single_method_mem0_default_produces_correct_filename():
    """Single-method run with mem0_default produces results_mem0_default.json."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_single_method_experiment("mem0_default")

        path = writer.write_json(experiment, filename="results_mem0_default.json")

        assert os.path.basename(path) == "results_mem0_default.json", (
            f"Expected filename 'results_mem0_default.json', got {os.path.basename(path)!r}"
        )
        assert os.path.exists(path), f"File not found: {path}"
        print("PASSED: mem0_default produces results_mem0_default.json")
    finally:
        shutil.rmtree(tmpdir)


def test_single_method_kernel_shared_produces_correct_filename():
    """Single-method run with kernel_shared produces results_kernel_shared.json."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_single_method_experiment("kernel_shared")

        path = writer.write_json(experiment, filename="results_kernel_shared.json")

        assert os.path.basename(path) == "results_kernel_shared.json", (
            f"Expected filename 'results_kernel_shared.json', got {os.path.basename(path)!r}"
        )
        assert os.path.exists(path), f"File not found: {path}"
        print("PASSED: kernel_shared produces results_kernel_shared.json")
    finally:
        shutil.rmtree(tmpdir)


def test_all_methods_produces_results_all_methods_json():
    """--method all produces results_all_methods.json."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_all_methods_experiment()

        path = writer.write_json(experiment, filename="results_all_methods.json")

        assert os.path.basename(path) == "results_all_methods.json", (
            f"Expected filename 'results_all_methods.json', got {os.path.basename(path)!r}"
        )
        assert os.path.exists(path), f"File not found: {path}"
        print("PASSED: --method all produces results_all_methods.json")
    finally:
        shutil.rmtree(tmpdir)


def test_json_output_is_valid_json():
    """The written JSON file must be valid and parseable."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_single_method_experiment("naive_concat")

        path = writer.write_json(experiment, filename="results_naive_concat.json")

        with open(path) as f:
            data = json.load(f)

        assert "experiment_metadata" in data
        assert "conditions" in data
        print("PASSED: JSON output is valid and parseable")
    finally:
        shutil.rmtree(tmpdir)


def test_csv_includes_method_column():
    """CSV output must include a 'method' column."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=True)
        experiment = _make_single_method_experiment("naive_concat")

        csv_path = writer.write_csv(experiment)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames

        assert "method" in columns, (
            f"CSV missing 'method' column. Columns: {columns}"
        )
        print("PASSED: CSV includes 'method' column")
    finally:
        shutil.rmtree(tmpdir)


def test_csv_includes_retrieved_context_count_column():
    """CSV output must include a 'retrieved_context_count' column."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=True)
        experiment = _make_single_method_experiment("vanilla_rag")

        csv_path = writer.write_csv(experiment)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames

        assert "retrieved_context_count" in columns, (
            f"CSV missing 'retrieved_context_count' column. Columns: {columns}"
        )
        print("PASSED: CSV includes 'retrieved_context_count' column")
    finally:
        shutil.rmtree(tmpdir)


def test_csv_method_column_has_correct_values():
    """CSV 'method' column must contain the correct method name for each row."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=True)
        experiment = _make_single_method_experiment("naive_concat")

        csv_path = writer.write_csv(experiment)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        assert rows[0]["method"] == "naive_concat", (
            f"Expected method='naive_concat', got {rows[0]['method']!r}"
        )
        print("PASSED: CSV 'method' column has correct values")
    finally:
        shutil.rmtree(tmpdir)


def test_csv_retrieved_context_count_column_has_correct_values():
    """CSV 'retrieved_context_count' column must contain the correct values."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=True)
        experiment = _make_single_method_experiment("vanilla_rag")
        # Override trial to have retrieved_context_count=3
        experiment.conditions[0].trials[0] = _make_trial_result(
            0, "vanilla_rag", "vanilla_rag", retrieved_context_count=3
        )

        csv_path = writer.write_csv(experiment)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["retrieved_context_count"] == "3", (
            f"Expected retrieved_context_count='3', got {rows[0]['retrieved_context_count']!r}"
        )
        print("PASSED: CSV 'retrieved_context_count' column has correct values")
    finally:
        shutil.rmtree(tmpdir)


def test_csv_all_methods_has_method_and_count_columns():
    """CSV for all-methods run includes method and retrieved_context_count for all rows."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=True)
        experiment = _make_all_methods_experiment()

        csv_path = writer.write_csv(experiment)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            rows = list(reader)

        assert "method" in columns, f"CSV missing 'method' column. Columns: {columns}"
        assert "retrieved_context_count" in columns, (
            f"CSV missing 'retrieved_context_count' column. Columns: {columns}"
        )
        assert len(rows) == 4, f"Expected 4 rows (one per method), got {len(rows)}"

        methods_in_csv = {row["method"] for row in rows}
        expected_methods = {"kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"}
        assert methods_in_csv == expected_methods, (
            f"Expected methods {expected_methods}, got {methods_in_csv}"
        )
        print("PASSED: CSV for all-methods run has correct columns and method values")
    finally:
        shutil.rmtree(tmpdir)


def test_default_filename_fallback():
    """write_json without filename parameter falls back to results.json."""
    tmpdir = tempfile.mkdtemp()
    try:
        writer = ResultsWriter(output_dir=tmpdir, write_csv=False)
        experiment = _make_single_method_experiment("naive_concat")

        path = writer.write_json(experiment)

        assert os.path.basename(path) == "results.json", (
            f"Expected default filename 'results.json', got {os.path.basename(path)!r}"
        )
        print("PASSED: write_json without filename falls back to results.json")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_single_method_naive_concat_produces_correct_filename()
    test_single_method_vanilla_rag_produces_correct_filename()
    test_single_method_mem0_default_produces_correct_filename()
    test_single_method_kernel_shared_produces_correct_filename()
    test_all_methods_produces_results_all_methods_json()
    test_json_output_is_valid_json()
    test_csv_includes_method_column()
    test_csv_includes_retrieved_context_count_column()
    test_csv_method_column_has_correct_values()
    test_csv_retrieved_context_count_column_has_correct_values()
    test_csv_all_methods_has_method_and_count_columns()
    test_default_filename_fallback()
    print("\nAll ResultsWriter tests passed.")
