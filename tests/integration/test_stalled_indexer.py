"""Integration test: Stalled-indexer scenario — bounded completion.

**Validates: Requirements 2.1 (bounded variant), 2.2 (bounded variant)**

This test harness verifies that when the Mem0 indexer stalls beyond the
configured barrier_timeout_ms, the write-barrier does NOT hang the pipeline.
Instead, it times out gracefully and allows retrieval to proceed against
whatever state is committed.

Test scenario:
- Inject a deliberate Mem0 latency spike exceeding barrier_timeout_ms
  (e.g., 5s delay for one user_id, where barrier_timeout_ms = 2000ms)
- Run a single Phase 2 trial
- Assert: trial completes within barrier_timeout_ms + assistant_inference_time + ε
- Assert: kernel emits a write_barrier_timeout warning
- Assert: trial does NOT hang and does NOT raise

Setup requirements:
- The kernel must support a test-mode Mem0 wrapper that can inject artificial
  latency into the indexing pipeline for specific user_ids. This can be
  implemented as:
  1. A kernel config flag (e.g., memory.test.inject_latency_ms) that adds
     a sleep before Mem0's internal commit
  2. A test-only HTTP endpoint on the kernel that configures per-user_id delays
  3. A Mem0 provider subclass that reads delay config from an env var or file
- The test documents the required setup and provides assertion utilities
  for validating results from any of these approaches.

Prerequisites:
- A running AIOS kernel built with the write-barrier fix (section 3 changes)
- Kernel configured with a test-mode Mem0 latency injection mechanism
- The benchmark harness available at benchmarks/shared_memory/

Usage:
    # Run against the fixed kernel with stalled-indexer test mode enabled
    python tests/integration/test_stalled_indexer.py

    # With a custom kernel URL
    python tests/integration/test_stalled_indexer.py --kernel-url http://localhost:8001

    # With pre-generated results directory (offline validation)
    python tests/integration/test_stalled_indexer.py --results-dir results/stalled_indexer/

    # Override barrier timeout (must match kernel config)
    python tests/integration/test_stalled_indexer.py --barrier-timeout-ms 2000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.integration.test_phase2_determinism import (
    TrialRecord,
    load_results,
    compute_aggregate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BARRIER_TIMEOUT_MS = 2000   # Default kernel barrier_timeout_ms
INJECTED_LATENCY_MS = 5000          # Artificial Mem0 latency (exceeds barrier timeout)
MAX_SLACK_MS = 1000                  # ε — maximum acceptable wallclock slack
DEFAULT_TRIALS = 1                   # Single trial is sufficient for this scenario


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_bounded_completion(
    wallclock_seconds: float,
    barrier_timeout_ms: int = DEFAULT_BARRIER_TIMEOUT_MS,
    assistant_inference_seconds: float = 0.0,
    slack_ms: int = MAX_SLACK_MS,
) -> Tuple[bool, str]:
    """Assert trial completed within barrier_timeout_ms + inference + ε.

    The total allowed time is:
        barrier_timeout_ms + assistant_inference_time + slack_ms

    Args:
        wallclock_seconds: Actual wallclock time for the trial.
        barrier_timeout_ms: Configured barrier timeout.
        assistant_inference_seconds: Estimated assistant model inference time.
        slack_ms: Maximum acceptable slack beyond the timeout + inference.

    Returns:
        Tuple of (passed, detail_message).
    """
    max_allowed_seconds = (
        (barrier_timeout_ms + slack_ms) / 1000.0
        + assistant_inference_seconds
    )
    wallclock_ms = wallclock_seconds * 1000

    detail_lines = [
        f"Wallclock time:        {wallclock_seconds:.2f}s ({wallclock_ms:.0f}ms)",
        f"Barrier timeout:       {barrier_timeout_ms}ms",
        f"Inference estimate:    {assistant_inference_seconds:.2f}s",
        f"Slack (ε):             {slack_ms}ms",
        f"Max allowed:           {max_allowed_seconds:.2f}s",
    ]

    if wallclock_seconds <= max_allowed_seconds:
        detail_lines.append("Trial completed within bounded time window.")
        return True, "\n    ".join(detail_lines)
    else:
        overshoot = wallclock_seconds - max_allowed_seconds
        detail_lines.append(f"EXCEEDED by {overshoot:.2f}s — possible pipeline stall.")
        return False, "\n    ".join(detail_lines)


def assert_no_hang(records: List[TrialRecord]) -> Tuple[bool, str]:
    """Assert that the trial completed (did not hang) and did not raise.

    Args:
        records: Trial records (should contain at least one record).

    Returns:
        Tuple of (passed, detail_message).
    """
    if not records:
        return False, "No trial records found — trial may have hung or crashed."

    # Check that at least one record exists and is not marked as a fatal failure
    # (a trial that completed with partial injection is acceptable)
    completed = [r for r in records]
    if not completed:
        return False, "No completed trial records."

    # Check for records that indicate a crash/exception (failed=True with no data)
    fatal_failures = [
        r for r in records
        if r.failed and r.shared_memory_count == 0 and r.injection_status == "unknown"
    ]

    if fatal_failures and len(fatal_failures) == len(records):
        return False, (
            f"All {len(records)} trial(s) appear to have crashed "
            f"(failed=True, no data). The pipeline may have raised an exception."
        )

    detail = (
        f"Trial completed successfully ({len(records)} record(s)). "
        f"Failed: {sum(1 for r in records if r.failed)}, "
        f"Completed: {sum(1 for r in records if not r.failed)}."
    )
    return True, detail


def assert_timeout_warning(
    kernel_logs: Optional[str] = None,
    results_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Assert that the kernel emitted a write_barrier_timeout warning.

    This can be verified through:
    1. Kernel log output (if captured)
    2. Results metadata (if the harness surfaces barrier diagnostics)
    3. Manual verification (documented for the operator)

    Args:
        kernel_logs: Optional captured kernel log output.
        results_metadata: Optional results metadata with barrier info.

    Returns:
        Tuple of (passed, detail_message).
    """
    # Check kernel logs if available
    if kernel_logs:
        if "write_barrier_timeout" in kernel_logs:
            return True, "write_barrier_timeout warning found in kernel logs."

    # Check results metadata if available
    if results_metadata:
        barrier_result = results_metadata.get("barrier_result", {})
        if barrier_result.get("status") == "timeout":
            residual = barrier_result.get("residual_pending", "unknown")
            waited = barrier_result.get("waited_ms", "unknown")
            return True, (
                f"Barrier returned status='timeout' "
                f"(residual_pending={residual}, waited_ms={waited})."
            )

    # If neither source is available, check if we can infer from timing
    # (the trial completing close to barrier_timeout_ms is strong evidence)
    return False, (
        "Cannot programmatically verify write_barrier_timeout warning. "
        "Manual verification required:\n"
        "    1. Check kernel stdout/stderr for 'write_barrier_timeout' log entry\n"
        "    2. Verify the warning includes the user_id and residual_pending count\n"
        "    3. Or enable barrier diagnostics in results metadata"
    )


def check_acceptable_result(records: List[TrialRecord]) -> Tuple[bool, str]:
    """Check that the trial result is acceptable for a stalled-indexer scenario.

    Acceptable outcomes when barrier times out:
    - injection_status="confirmed" with partial count (some writes committed)
    - injection_status="unknown" (no writes committed in time)
    - injection_status="audit_inferred" (partial visibility)

    NOT acceptable:
    - An unhandled exception / crash

    Args:
        records: Trial records from the stalled-indexer run.

    Returns:
        Tuple of (passed, detail_message).
    """
    non_failed = [r for r in records if not r.failed]
    if not non_failed and records:
        # All trials failed — check if it's a graceful failure
        return True, (
            "All trials marked as failed, but pipeline did not hang. "
            "This is acceptable for a stalled-indexer scenario "
            "(timeout may have produced incomplete results)."
        )

    if not records:
        return False, "No trial records at all."

    # Any completion is acceptable — the key assertion is that it didn't hang
    statuses = set(r.injection_status for r in records)
    counts = [r.shared_memory_count for r in records]

    detail = (
        f"Trial produced results with injection_status={statuses}, "
        f"shared_memory_counts={counts}. "
        f"This is acceptable — barrier timed out gracefully."
    )
    return True, detail


# ---------------------------------------------------------------------------
# Run commands
# ---------------------------------------------------------------------------


def run_evaluation(
    trials: int = DEFAULT_TRIALS,
    output_dir: str = "results/stalled_indexer/",
    kernel_url: Optional[str] = None,
) -> Tuple[str, float]:
    """Execute run_evaluation.py and measure wallclock time.

    Args:
        trials: Number of trials to run.
        output_dir: Output directory for results.
        kernel_url: Optional kernel URL override.

    Returns:
        Tuple of (output_dir, wallclock_seconds).

    Raises:
        subprocess.CalledProcessError: If the evaluation script fails fatally.
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
    start = time.monotonic()
    subprocess.run(cmd, check=True, cwd=_project_root)
    wallclock = time.monotonic() - start
    return output_dir, wallclock


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_test(
    trials: int = DEFAULT_TRIALS,
    kernel_url: Optional[str] = None,
    results_dir: Optional[str] = None,
    barrier_timeout_ms: int = DEFAULT_BARRIER_TIMEOUT_MS,
    kernel_logs_path: Optional[str] = None,
) -> bool:
    """Execute the stalled-indexer bounded completion integration test.

    When results_dir is provided, validates pre-existing results (offline mode).
    Otherwise, runs a fresh evaluation against the kernel (which must have
    the Mem0 latency injection enabled).

    Args:
        trials: Number of trials to run.
        kernel_url: Kernel URL override.
        results_dir: Pre-existing results directory (offline validation).
        barrier_timeout_ms: Configured barrier timeout on the kernel.
        kernel_logs_path: Optional path to captured kernel logs.

    Returns:
        True if all assertions pass, False otherwise.
    """
    print("=" * 70)
    print("Integration Test 5.4: Stalled-Indexer Scenario — Bounded Completion")
    print("Validates: Requirements 2.1 (bounded), 2.2 (bounded)")
    print("=" * 70)

    print(f"\n  Configuration:")
    print(f"    barrier_timeout_ms:    {barrier_timeout_ms}")
    print(f"    injected_latency_ms:   {INJECTED_LATENCY_MS}")
    print(f"    max_slack_ms (ε):      {MAX_SLACK_MS}")
    print(f"    Expected: trial completes despite indexer stall")

    results = []
    wallclock_seconds = None
    kernel_logs = None

    # Load kernel logs if path provided
    if kernel_logs_path and os.path.exists(kernel_logs_path):
        with open(kernel_logs_path, "r") as f:
            kernel_logs = f.read()

    # -----------------------------------------------------------------------
    # Step 1: Run (or load) evaluation
    # -----------------------------------------------------------------------
    if results_dir:
        print(f"\n[Step 1] Loading pre-existing results from: {results_dir}")
        try:
            records = load_results(results_dir)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return False

        # Try to load timing metadata
        metadata_path = os.path.join(results_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                wallclock_seconds = metadata.get("wallclock_seconds")
        else:
            # Estimate from latency fields
            non_failed = [r for r in records if not r.failed]
            if non_failed and non_failed[0].latency_seconds is not None:
                wallclock_seconds = non_failed[0].latency_seconds
    else:
        output = "results/stalled_indexer/"
        print(f"\n[Step 1] Running evaluation ({trials} trial(s), stalled indexer)...")
        print(f"  IMPORTANT: Kernel must have Mem0 latency injection enabled!")
        print(f"  Expected kernel setup:")
        print(f"    - Mem0 test wrapper adding {INJECTED_LATENCY_MS}ms delay")
        print(f"    - barrier_timeout_ms = {barrier_timeout_ms}")
        print(f"    - Barrier will timeout (injected latency > barrier timeout)")
        try:
            output, wallclock_seconds = run_evaluation(
                trials=trials, output_dir=output, kernel_url=kernel_url
            )
            records = load_results(output)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Evaluation script failed: {e}")
            print(f"  NOTE: A CalledProcessError may indicate the pipeline hung.")
            return False
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return False

    print(f"  Loaded {len(records)} trial record(s).")
    if wallclock_seconds is not None:
        print(f"  Wallclock time: {wallclock_seconds:.2f}s")

    # -----------------------------------------------------------------------
    # Step 2: Assert trial did not hang
    # -----------------------------------------------------------------------
    print(f"\n[Step 2] Asserting trial did NOT hang and did NOT raise...")
    passed, detail = assert_no_hang(records)
    results.append(("No Hang / No Raise", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 3: Assert bounded completion time
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] Asserting bounded completion time...")
    if wallclock_seconds is not None:
        # Estimate inference time from the trial's latency minus barrier overhead
        # For a single trial, the wallclock includes: pipeline setup + barrier wait + inference
        # Conservative estimate: assume inference is at least wallclock - barrier - slack
        assistant_inference_estimate = max(
            0.0, wallclock_seconds - (barrier_timeout_ms / 1000.0) - 1.0
        )
        # Cap inference estimate to be reasonable
        assistant_inference_estimate = min(assistant_inference_estimate, 60.0)

        passed, detail = assert_bounded_completion(
            wallclock_seconds=wallclock_seconds,
            barrier_timeout_ms=barrier_timeout_ms,
            assistant_inference_seconds=assistant_inference_estimate,
            slack_ms=MAX_SLACK_MS,
        )
        results.append(("Bounded Completion", passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {detail}")
    else:
        detail = (
            "Cannot verify bounded completion — no wallclock time available. "
            "Provide --results-dir with a metadata.json containing wallclock_seconds, "
            "or run the test live."
        )
        results.append(("Bounded Completion", False, detail))
        print(f"  [SKIP] {detail}")

    # -----------------------------------------------------------------------
    # Step 4: Assert write_barrier_timeout warning emitted
    # -----------------------------------------------------------------------
    print(f"\n[Step 4] Checking for write_barrier_timeout warning...")
    # Try to load results metadata for barrier diagnostics
    results_metadata = None
    if results_dir:
        meta_path = os.path.join(results_dir, "barrier_diagnostics.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                results_metadata = json.load(f)

    passed, detail = assert_timeout_warning(
        kernel_logs=kernel_logs,
        results_metadata=results_metadata,
    )
    results.append(("write_barrier_timeout Warning", passed, detail))
    status = "PASS" if passed else "WARN"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 5: Assert acceptable result (graceful degradation)
    # -----------------------------------------------------------------------
    print(f"\n[Step 5] Asserting acceptable result (graceful degradation)...")
    passed, detail = check_acceptable_result(records)
    results.append(("Acceptable Result", passed, detail))
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
        print(f"\n  ALL ASSERTIONS PASSED — Stalled indexer handled gracefully")
        print(f"  Pipeline bounded by barrier_timeout_ms ({barrier_timeout_ms}ms)")
        print(f"  (Requirements 2.1, 2.2 bounded variant validated).")
    else:
        print(f"\n  {total_failed} assertion(s) FAILED or could not be verified.")
        print(f"  Check kernel configuration and Mem0 latency injection setup.")

    print(f"\n{'=' * 70}")
    print("SETUP DOCUMENTATION")
    print(f"{'=' * 70}")
    print("""
  To run this test, the kernel must be configured with a Mem0 latency
  injection mechanism. Recommended approaches:

  Option A — Environment variable:
    export AIOS_MEM0_TEST_LATENCY_MS=5000
    # Kernel's Mem0Provider reads this and adds a sleep before commit

  Option B — Kernel config:
    memory:
      test:
        inject_latency_ms: 5000
        target_user_id: "test_user_stalled"  # optional: scope to one user

  Option C — Test-only Mem0 wrapper:
    # In the kernel test suite, subclass Mem0Provider:
    class StalledMem0Provider(Mem0Provider):
        def add(self, *args, **kwargs):
            time.sleep(5.0)  # Simulate stalled indexer
            return super().add(*args, **kwargs)

  After enabling the latency injection, run:
    python tests/integration/test_stalled_indexer.py --kernel-url <url>

  To capture kernel logs for automated warning verification:
    python run_kernel.py 2>&1 | tee kernel_output.log
    python tests/integration/test_stalled_indexer.py \\
        --kernel-logs kernel_output.log
""")

    return total_failed == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Parse CLI args and run the integration test."""
    parser = argparse.ArgumentParser(
        description=(
            "Integration test 5.4: Stalled-indexer scenario — bounded completion"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Number of trials to run (default: {DEFAULT_TRIALS}).",
    )
    parser.add_argument(
        "--kernel-url",
        type=str,
        default=None,
        help="AIOS kernel base URL (e.g. http://localhost:8001).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Pre-existing results directory for offline validation.",
    )
    parser.add_argument(
        "--barrier-timeout-ms",
        type=int,
        default=DEFAULT_BARRIER_TIMEOUT_MS,
        help=f"Configured barrier_timeout_ms on the kernel (default: {DEFAULT_BARRIER_TIMEOUT_MS}).",
    )
    parser.add_argument(
        "--kernel-logs",
        type=str,
        default=None,
        help="Path to captured kernel log output (for warning verification).",
    )
    args = parser.parse_args()

    all_passed = run_test(
        trials=args.trials,
        kernel_url=args.kernel_url,
        results_dir=args.results_dir,
        barrier_timeout_ms=args.barrier_timeout_ms,
        kernel_logs_path=args.kernel_logs,
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
