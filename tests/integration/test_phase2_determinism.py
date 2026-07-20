"""Integration test: Two-phase determinism — Phase 2 injection rate.

**Validates: Requirements 2.5**

This test harness verifies that the write-barrier fix produces deterministic
Phase 2 injection results when run against a kernel with the section 3 changes.

Test assertions:
1. Every trial's RetrievalLog.injection_status == "confirmed"
   and shared_memory_count >= 2 (profile + task per the synthetic generator)
2. Two consecutive runs with identical seeds and config produce identical
   aggregate counts (profile usage, task usage, integration scores)

Prerequisites:
- A running AIOS kernel built with the write-barrier fix (section 3 changes)
- Kernel restarted between aggregate phases per project-overview.md
- The benchmark harness available at benchmarks/shared_memory/

Usage:
    # Run against the fixed kernel (kernel must be running on localhost:8000)
    python tests/integration/test_phase2_determinism.py

    # Or with a custom kernel URL
    python tests/integration/test_phase2_determinism.py --kernel-url http://localhost:8001

    # Or with pre-generated results directories (offline validation)
    python tests/integration/test_phase2_determinism.py \\
        --run1-dir results/post_fix_phase2_run1/ \\
        --run2-dir results/post_fix_phase2_run2/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Data structures for parsed results
# ---------------------------------------------------------------------------


@dataclass
class TrialRecord:
    """Parsed trial data from CSV or JSON results."""

    trial_index: int
    condition: str
    injection_status: str
    shared_memory_count: int
    profile_usage_score: Optional[int]
    task_usage_score: Optional[int]
    integration_score: Optional[int]
    latency_seconds: Optional[float]
    failed: bool = False


@dataclass
class AggregateMetrics:
    """Aggregate metrics from a single run for determinism comparison."""

    total_trials: int
    failed_trials: int
    confirmed_count: int
    profile_usage_scores: List[Optional[int]]
    task_usage_scores: List[Optional[int]]
    integration_scores: List[Optional[int]]
    shared_memory_counts: List[int]

    @property
    def injection_rate(self) -> float:
        """Fraction of non-failed trials with injection_status == 'confirmed'."""
        non_failed = self.total_trials - self.failed_trials
        if non_failed == 0:
            return 0.0
        return self.confirmed_count / non_failed


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


def parse_csv_results(csv_path: str) -> List[TrialRecord]:
    """Parse a results.csv file produced by run_evaluation.py --csv.

    The CSV columns are:
        condition, method, retrieved_context_count, trial_index,
        profile_usage_score, task_usage_score, integration_score,
        memory_total, memory_shared, memory_private,
        shared_memory_count, cross_agent_found, latency_seconds,
        query, response

    Args:
        csv_path: Path to the results.csv file.

    Returns:
        List of TrialRecord objects.
    """
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_index = int(row.get("trial_index", 0))
            condition = row.get("condition", "")
            shared_memory_count = int(row.get("shared_memory_count", 0))

            # Derive injection_status from shared_memory_count and cross_agent_found
            # The CSV doesn't include injection_status directly; infer from counts
            cross_agent_found = row.get("cross_agent_found", "False") == "True"
            if shared_memory_count >= 2 and cross_agent_found:
                injection_status = "confirmed"
            elif shared_memory_count > 0:
                injection_status = "audit_inferred"
            else:
                injection_status = "unknown"

            profile_score = _parse_optional_int(row.get("profile_usage_score"))
            task_score = _parse_optional_int(row.get("task_usage_score"))
            integration_score = _parse_optional_int(row.get("integration_score"))
            latency = _parse_optional_float(row.get("latency_seconds"))

            records.append(TrialRecord(
                trial_index=trial_index,
                condition=condition,
                injection_status=injection_status,
                shared_memory_count=shared_memory_count,
                profile_usage_score=profile_score,
                task_usage_score=task_score,
                integration_score=integration_score,
                latency_seconds=latency,
                failed=False,
            ))
    return records


def parse_json_results(json_path: str) -> List[TrialRecord]:
    """Parse a results.json file produced by run_evaluation.py.

    Args:
        json_path: Path to the results.json file.

    Returns:
        List of TrialRecord objects.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for condition_data in data.get("conditions", []):
        for trial in condition_data.get("trials", []):
            retrieval_log = trial.get("retrieval_log") or {}
            injection_status = retrieval_log.get("injection_status", "unknown")
            shared_memory_count = retrieval_log.get("shared_memory_count", 0)

            records.append(TrialRecord(
                trial_index=trial.get("trial_index", 0),
                condition=trial.get("condition", ""),
                injection_status=injection_status,
                shared_memory_count=shared_memory_count,
                profile_usage_score=trial.get("profile_usage_score"),
                task_usage_score=trial.get("task_usage_score"),
                integration_score=trial.get("integration_score"),
                latency_seconds=trial.get("latency_seconds"),
                failed=trial.get("failed", False),
            ))
    return records


def _parse_optional_int(val) -> Optional[int]:
    """Safely parse an optional int from a CSV value."""
    if val is None or val == "" or val == "None":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_optional_float(val) -> Optional[float]:
    """Safely parse an optional float from a CSV value."""
    if val is None or val == "" or val == "None":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_results(results_dir: str) -> List[TrialRecord]:
    """Load trial records from a results directory (prefers JSON over CSV).

    Args:
        results_dir: Directory containing results.json and/or results.csv.

    Returns:
        List of TrialRecord objects.

    Raises:
        FileNotFoundError: If neither results.json nor results.csv exists.
    """
    json_path = os.path.join(results_dir, "results.json")
    csv_path = os.path.join(results_dir, "results.csv")

    if os.path.exists(json_path):
        return parse_json_results(json_path)
    elif os.path.exists(csv_path):
        return parse_csv_results(csv_path)
    else:
        raise FileNotFoundError(
            f"No results.json or results.csv found in {results_dir}"
        )


def compute_aggregate(records: List[TrialRecord]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of trial records.

    Args:
        records: List of TrialRecord objects from a single run.

    Returns:
        AggregateMetrics summarizing the run.
    """
    total = len(records)
    failed = sum(1 for r in records if r.failed)
    confirmed = sum(
        1 for r in records
        if not r.failed and r.injection_status == "confirmed"
    )

    return AggregateMetrics(
        total_trials=total,
        failed_trials=failed,
        confirmed_count=confirmed,
        profile_usage_scores=[r.profile_usage_score for r in records if not r.failed],
        task_usage_scores=[r.task_usage_score for r in records if not r.failed],
        integration_scores=[r.integration_score for r in records if not r.failed],
        shared_memory_counts=[r.shared_memory_count for r in records if not r.failed],
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_injection_rate(records: List[TrialRecord]) -> Tuple[bool, str]:
    """Assert every non-failed trial has injection_status == 'confirmed'
    and shared_memory_count >= 2.

    Args:
        records: Trial records from a Phase 2 run.

    Returns:
        Tuple of (passed, detail_message).
    """
    non_failed = [r for r in records if not r.failed]
    if not non_failed:
        return False, "No non-failed trials to evaluate."

    failures = []
    for r in non_failed:
        issues = []
        if r.injection_status != "confirmed":
            issues.append(f"injection_status={r.injection_status!r}")
        if r.shared_memory_count < 2:
            issues.append(f"shared_memory_count={r.shared_memory_count}")
        if issues:
            failures.append(f"  Trial {r.trial_index}: {', '.join(issues)}")

    if failures:
        detail = (
            f"{len(failures)}/{len(non_failed)} trials failed injection assertion:\n"
            + "\n".join(failures[:10])
        )
        if len(failures) > 10:
            detail += f"\n  ... and {len(failures) - 10} more"
        return False, detail

    return True, f"All {len(non_failed)} trials: injection_status='confirmed', shared_memory_count >= 2"


def assert_determinism(
    agg1: AggregateMetrics, agg2: AggregateMetrics
) -> Tuple[bool, str]:
    """Assert two runs produce identical aggregate counts.

    Compares profile_usage_scores, task_usage_scores, and integration_scores
    element-wise between two runs (deterministic outcome).

    Args:
        agg1: Aggregate metrics from run 1.
        agg2: Aggregate metrics from run 2.

    Returns:
        Tuple of (passed, detail_message).
    """
    differences = []

    if agg1.total_trials != agg2.total_trials:
        differences.append(
            f"total_trials: run1={agg1.total_trials}, run2={agg2.total_trials}"
        )

    if agg1.profile_usage_scores != agg2.profile_usage_scores:
        mismatches = _count_mismatches(agg1.profile_usage_scores, agg2.profile_usage_scores)
        differences.append(f"profile_usage_scores: {mismatches} mismatches")

    if agg1.task_usage_scores != agg2.task_usage_scores:
        mismatches = _count_mismatches(agg1.task_usage_scores, agg2.task_usage_scores)
        differences.append(f"task_usage_scores: {mismatches} mismatches")

    if agg1.integration_scores != agg2.integration_scores:
        mismatches = _count_mismatches(agg1.integration_scores, agg2.integration_scores)
        differences.append(f"integration_scores: {mismatches} mismatches")

    if agg1.shared_memory_counts != agg2.shared_memory_counts:
        mismatches = _count_mismatches(agg1.shared_memory_counts, agg2.shared_memory_counts)
        differences.append(f"shared_memory_counts: {mismatches} mismatches")

    if differences:
        detail = "Determinism check FAILED — runs produced different results:\n"
        detail += "\n".join(f"  {d}" for d in differences)
        return False, detail

    return True, (
        f"Determinism verified: both runs produced identical aggregate counts "
        f"across {agg1.total_trials} trials "
        f"(profile_usage, task_usage, integration, shared_memory_counts all match)"
    )


def _count_mismatches(list1: list, list2: list) -> int:
    """Count element-wise mismatches between two lists."""
    max_len = max(len(list1), len(list2))
    mismatches = abs(len(list1) - len(list2))
    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Run commands
# ---------------------------------------------------------------------------


def run_evaluation(
    trials: int = 30,
    output_dir: str = "results/post_fix_phase2/",
    kernel_url: Optional[str] = None,
) -> str:
    """Execute run_evaluation.py --condition phase2 --csv and return the output dir.

    Args:
        trials: Number of trials to run.
        output_dir: Output directory for results.
        kernel_url: Optional kernel URL override.

    Returns:
        The output directory path.

    Raises:
        subprocess.CalledProcessError: If the evaluation script fails.
    """
    cmd = [
        sys.executable,
        "benchmarks/shared_memory/run_evaluation.py",
        "--trials", str(trials),
        "--output", output_dir,
        "--condition", "phase2",
        "--csv",
    ]
    if kernel_url:
        cmd.extend(["--kernel-url", kernel_url])

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_project_root)
    return output_dir


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_test(
    trials: int = 30,
    kernel_url: Optional[str] = None,
    run1_dir: Optional[str] = None,
    run2_dir: Optional[str] = None,
) -> bool:
    """Execute the Phase 2 determinism integration test.

    When run1_dir and run2_dir are provided, validates pre-existing results
    (offline mode). Otherwise, runs two fresh evaluations against the kernel.

    Args:
        trials: Number of trials per run.
        kernel_url: Kernel URL override.
        run1_dir: Pre-existing results directory for run 1 (offline mode).
        run2_dir: Pre-existing results directory for run 2 (offline mode).

    Returns:
        True if all assertions pass, False otherwise.
    """
    print("=" * 70)
    print("Integration Test 5.1: Two-Phase Determinism — Phase 2 Injection Rate")
    print("Validates: Requirements 2.5")
    print("=" * 70)

    results = []

    # -----------------------------------------------------------------------
    # Step 1: Run (or load) first evaluation
    # -----------------------------------------------------------------------
    if run1_dir:
        print(f"\n[Step 1] Loading pre-existing run 1 results from: {run1_dir}")
        records1 = load_results(run1_dir)
    else:
        output1 = f"results/post_fix_phase2_run1/"
        print(f"\n[Step 1] Running evaluation (run 1, {trials} trials)...")
        print(f"  Command: python benchmarks/shared_memory/run_evaluation.py "
              f"--trials {trials} --output {output1} --condition phase2 --csv")
        try:
            run_evaluation(trials=trials, output_dir=output1, kernel_url=kernel_url)
            records1 = load_results(output1)
        except Exception as e:
            print(f"  ERROR: Run 1 failed: {e}")
            return False

    print(f"  Loaded {len(records1)} trial records from run 1.")

    # -----------------------------------------------------------------------
    # Step 2: Assert injection rate on run 1
    # -----------------------------------------------------------------------
    print(f"\n[Step 2] Asserting injection rate (run 1)...")
    print(f"  Expected: every trial injection_status='confirmed', shared_memory_count >= 2")
    passed, detail = assert_injection_rate(records1)
    results.append(("Injection Rate (Run 1)", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 3: Run (or load) second evaluation (same kernel, same seeds)
    # -----------------------------------------------------------------------
    if run2_dir:
        print(f"\n[Step 3] Loading pre-existing run 2 results from: {run2_dir}")
        records2 = load_results(run2_dir)
    else:
        output2 = f"results/post_fix_phase2_run2/"
        print(f"\n[Step 3] Running evaluation (run 2, {trials} trials, same config)...")
        print(f"  NOTE: Restart kernel between runs per project-overview.md")
        print(f"  Command: python benchmarks/shared_memory/run_evaluation.py "
              f"--trials {trials} --output {output2} --condition phase2 --csv")
        try:
            run_evaluation(trials=trials, output_dir=output2, kernel_url=kernel_url)
            records2 = load_results(output2)
        except Exception as e:
            print(f"  ERROR: Run 2 failed: {e}")
            return False

    print(f"  Loaded {len(records2)} trial records from run 2.")

    # -----------------------------------------------------------------------
    # Step 4: Assert injection rate on run 2
    # -----------------------------------------------------------------------
    print(f"\n[Step 4] Asserting injection rate (run 2)...")
    passed, detail = assert_injection_rate(records2)
    results.append(("Injection Rate (Run 2)", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 5: Assert determinism between run 1 and run 2
    # -----------------------------------------------------------------------
    print(f"\n[Step 5] Asserting determinism (run 1 vs run 2)...")
    print(f"  Expected: identical aggregate profile_usage, task_usage, integration scores")
    agg1 = compute_aggregate(records1)
    agg2 = compute_aggregate(records2)
    passed, detail = assert_determinism(agg1, agg2)
    results.append(("Determinism (Run 1 vs Run 2)", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_passed = sum(1 for _, p, _ in results if p)
    total_failed = sum(1 for _, p, _ in results if not p)

    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {len(results)}")

    if total_failed == 0:
        print(f"\n  ALL ASSERTIONS PASSED — Phase 2 injection is deterministic")
        print(f"  with the write-barrier fix (Requirement 2.5 validated).")
    else:
        print(f"\n  {total_failed} assertion(s) FAILED.")
        print(f"  The write-barrier fix may not be producing deterministic results.")

    print(f"{'=' * 70}")
    return total_failed == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Parse CLI args and run the integration test."""
    parser = argparse.ArgumentParser(
        description="Integration test 5.1: Two-phase determinism — Phase 2 injection rate",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of trials per run (default: 30).",
    )
    parser.add_argument(
        "--kernel-url",
        type=str,
        default=None,
        help="AIOS kernel base URL (e.g. http://localhost:8001).",
    )
    parser.add_argument(
        "--run1-dir",
        type=str,
        default=None,
        help="Pre-existing results directory for run 1 (offline validation).",
    )
    parser.add_argument(
        "--run2-dir",
        type=str,
        default=None,
        help="Pre-existing results directory for run 2 (offline validation).",
    )
    args = parser.parse_args()

    all_passed = run_test(
        trials=args.trials,
        kernel_url=args.kernel_url,
        run1_dir=args.run1_dir,
        run2_dir=args.run2_dir,
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
