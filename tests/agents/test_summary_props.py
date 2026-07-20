"""Property-based tests for ResultsWriter.print_summary delta arithmetic.

# Feature: personalization-baselines, Property 12: Comparative summary delta arithmetic
"""

import io
import math
import sys
sys.path.insert(0, ".")

from hypothesis import given, settings
from hypothesis.strategies import (
    composite,
    floats,
    integers,
    lists,
    text,
    from_regex,
)

from benchmarks.shared_memory.models import (
    ConditionResults,
    ConditionSummary,
    ExperimentMetadata,
    ExperimentResults,
    SummaryStatistics,
    TrialResult,
)
from benchmarks.shared_memory.results import ResultsWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(mean: float) -> SummaryStatistics:
    """Build a SummaryStatistics with a given mean and neutral other fields."""
    return SummaryStatistics(mean=mean, std=0.0, min=mean, max=mean)


def _make_summary(
    profile_mean: float,
    task_mean: float,
    integ_mean: float,
    latency_mean: float = 1.0,
    total: int = 10,
    failed: int = 0,
) -> ConditionSummary:
    """Build a ConditionSummary with specified means."""
    return ConditionSummary(
        profile_usage=_make_stats(profile_mean),
        task_usage=_make_stats(task_mean),
        integration=_make_stats(integ_mean),
        latency=_make_stats(latency_mean),
        memory_total=_make_stats(0.0),
        memory_shared=_make_stats(0.0),
        memory_private=_make_stats(0.0),
        injected_memories=_make_stats(0.0),
        total_trials=total,
        failed_trials=failed,
    )


def _make_experiment(conditions: list) -> ExperimentResults:
    """Build a minimal ExperimentResults from a list of ConditionResults."""
    return ExperimentResults(
        experiment_metadata=ExperimentMetadata(
            trials_per_condition=10,
            timestamp="2024-01-01T00:00:00",
            kernel_url="http://localhost:8000",
            conditions_run=[c.condition for c in conditions],
            methods_run=[c.condition for c in conditions],
        ),
        conditions=conditions,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

reasonable_float = floats(min_value=-5.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@composite
def method_summary_pair(draw):
    """Generate a (kernel_shared_means, baseline_means) pair of float triples."""
    ks_profile = draw(reasonable_float)
    ks_task = draw(reasonable_float)
    ks_integ = draw(reasonable_float)
    bl_profile = draw(reasonable_float)
    bl_task = draw(reasonable_float)
    bl_integ = draw(reasonable_float)
    baseline_name = draw(from_regex(r"[a-z][a-z0-9_]{1,15}", fullmatch=True))
    # Ensure baseline name is not "kernel_shared" to keep the test meaningful
    if baseline_name == "kernel_shared":
        baseline_name = "baseline_method"
    return {
        "ks": (ks_profile, ks_task, ks_integ),
        "bl": (bl_profile, bl_task, bl_integ),
        "baseline_name": baseline_name,
    }


# ---------------------------------------------------------------------------
# Property 12: Comparative summary delta arithmetic
# ---------------------------------------------------------------------------

# Feature: personalization-baselines, Property 12: Comparative summary delta arithmetic
class TestComparativeSummaryDeltaArithmetic:
    """**Validates: Requirements 9.3**"""

    @given(pair=method_summary_pair())
    @settings(max_examples=100)
    def test_delta_equals_arithmetic_difference(self, pair):
        """For any two method summaries, the delta values printed in the
        comparative summary table SHALL equal the arithmetic difference
        (baseline_method_mean - kernel_shared_mean) for each metric.

        We verify this by:
        1. Building an ExperimentResults with kernel_shared and one baseline.
        2. Capturing stdout from print_summary.
        3. Parsing the baseline row from the output.
        4. Asserting the printed delta values match the arithmetic difference.
        """
        ks_profile, ks_task, ks_integ = pair["ks"]
        bl_profile, bl_task, bl_integ = pair["bl"]
        baseline_name = pair["baseline_name"]

        ks_summary = _make_summary(ks_profile, ks_task, ks_integ)
        bl_summary = _make_summary(bl_profile, bl_task, bl_integ)

        ks_condition = ConditionResults(
            condition="kernel_shared",
            trials=[],
            summary=ks_summary,
        )
        bl_condition = ConditionResults(
            condition=baseline_name,
            trials=[],
            summary=bl_summary,
        )

        experiment = _make_experiment([ks_condition, bl_condition])
        writer = ResultsWriter(output_dir="/tmp/test_summary_props")

        # Capture stdout
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            writer.print_summary(experiment)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()

        # Find the baseline row in the output (line starting with baseline_name)
        baseline_row = None
        for line in output.splitlines():
            if line.strip().startswith(baseline_name):
                baseline_row = line.strip()
                break

        assert baseline_row is not None, (
            f"Could not find baseline row for '{baseline_name}' in output:\n{output}"
        )

        # Parse the numeric columns from the row
        # Row format: method  profile  task  integration  latency  delta_profile  delta_task  delta_integ  total  failed
        parts = baseline_row.split()
        # parts[0] is the method name; remaining are numeric values
        # We need at least 10 parts: name + 4 metrics + 3 deltas + total + failed
        assert len(parts) >= 10, (
            f"Expected at least 10 columns in baseline row, got {len(parts)}: {baseline_row}"
        )

        # Columns after method name: profile task integ latency delta_profile delta_task delta_integ total failed
        parsed_profile = float(parts[1])
        parsed_task = float(parts[2])
        parsed_integ = float(parts[3])
        # parts[4] is latency
        parsed_delta_profile = float(parts[5])
        parsed_delta_task = float(parts[6])
        parsed_delta_integ = float(parts[7])

        expected_delta_profile = bl_profile - ks_profile
        expected_delta_task = bl_task - ks_task
        expected_delta_integ = bl_integ - ks_integ

        assert math.isclose(parsed_profile, bl_profile, rel_tol=1e-3, abs_tol=1e-3), (
            f"Profile mean mismatch: printed {parsed_profile}, expected {bl_profile}"
        )
        assert math.isclose(parsed_task, bl_task, rel_tol=1e-3, abs_tol=1e-3), (
            f"Task mean mismatch: printed {parsed_task}, expected {bl_task}"
        )
        assert math.isclose(parsed_integ, bl_integ, rel_tol=1e-3, abs_tol=1e-3), (
            f"Integration mean mismatch: printed {parsed_integ}, expected {bl_integ}"
        )
        assert math.isclose(parsed_delta_profile, expected_delta_profile, rel_tol=1e-3, abs_tol=1e-3), (
            f"Delta profile mismatch: printed {parsed_delta_profile}, "
            f"expected {expected_delta_profile} ({bl_profile} - {ks_profile})"
        )
        assert math.isclose(parsed_delta_task, expected_delta_task, rel_tol=1e-3, abs_tol=1e-3), (
            f"Delta task mismatch: printed {parsed_delta_task}, "
            f"expected {expected_delta_task} ({bl_task} - {ks_task})"
        )
        assert math.isclose(parsed_delta_integ, expected_delta_integ, rel_tol=1e-3, abs_tol=1e-3), (
            f"Delta integration mismatch: printed {parsed_delta_integ}, "
            f"expected {expected_delta_integ} ({bl_integ} - {ks_integ})"
        )

    @given(pair=method_summary_pair())
    @settings(max_examples=100)
    def test_kernel_shared_delta_is_zero(self, pair):
        """The kernel_shared row's delta values SHALL be zero (it is the baseline
        against itself), verifying the delta formula is (method - kernel_shared)."""
        ks_profile, ks_task, ks_integ = pair["ks"]
        bl_profile, bl_task, bl_integ = pair["bl"]
        baseline_name = pair["baseline_name"]

        ks_summary = _make_summary(ks_profile, ks_task, ks_integ)
        bl_summary = _make_summary(bl_profile, bl_task, bl_integ)

        ks_condition = ConditionResults(
            condition="kernel_shared",
            trials=[],
            summary=ks_summary,
        )
        bl_condition = ConditionResults(
            condition=baseline_name,
            trials=[],
            summary=bl_summary,
        )

        experiment = _make_experiment([ks_condition, bl_condition])
        writer = ResultsWriter(output_dir="/tmp/test_summary_props")

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            writer.print_summary(experiment)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()

        # Find the kernel_shared row
        ks_row = None
        for line in output.splitlines():
            if line.strip().startswith("kernel_shared"):
                ks_row = line.strip()
                break

        assert ks_row is not None, (
            f"Could not find kernel_shared row in output:\n{output}"
        )

        parts = ks_row.split()
        assert len(parts) >= 10, (
            f"Expected at least 10 columns in kernel_shared row, got {len(parts)}: {ks_row}"
        )

        parsed_delta_profile = float(parts[5])
        parsed_delta_task = float(parts[6])
        parsed_delta_integ = float(parts[7])

        assert math.isclose(parsed_delta_profile, 0.0, abs_tol=1e-6), (
            f"kernel_shared delta_profile should be 0.0, got {parsed_delta_profile}"
        )
        assert math.isclose(parsed_delta_task, 0.0, abs_tol=1e-6), (
            f"kernel_shared delta_task should be 0.0, got {parsed_delta_task}"
        )
        assert math.isclose(parsed_delta_integ, 0.0, abs_tol=1e-6), (
            f"kernel_shared delta_integ should be 0.0, got {parsed_delta_integ}"
        )


if __name__ == "__main__":
    test = TestComparativeSummaryDeltaArithmetic()
    print("Running Property 12: Comparative summary delta arithmetic...")
    test.test_delta_equals_arithmetic_difference()
    print("PASSED: Property 12 (delta equals arithmetic difference)")
    test.test_kernel_shared_delta_is_zero()
    print("PASSED: Property 12 (kernel_shared delta is zero)")
