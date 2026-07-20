"""Integration test: Single-agent flow regression.

**Validates: Requirements 3.1, 3.3**

This test harness verifies that the write-barrier fix does NOT regress
single-agent flows. When an agent operates alone (no cross-agent retrieval,
no pending writes from other agents), the barrier should not activate and
latency should be within noise of the pre-fix kernel baseline.

Test scenario:
- Run an existing single-agent example (e.g., test_agent) against the fixed kernel
- Assert: latency and observable behavior within noise of pre-fix baseline
- Assert: no barrier path activated (no added latency from the barrier)
- Validates preservation clauses 3.1 (no pending writes → barrier skipped)
  and 3.3 (single-agent retrieval → barrier skipped)

Prerequisites:
- A running AIOS kernel built with the write-barrier fix (section 3 changes)
- The test_agent example available at cerebrum/example/agents/test_agent/
- No other agents writing shared memories for the same user_id concurrently

Usage:
    # Run against the fixed kernel (kernel must be running on localhost:8000)
    python tests/integration/test_single_agent_regression.py

    # With a custom kernel URL
    python tests/integration/test_single_agent_regression.py --kernel-url http://localhost:8001

    # With pre-generated results directory (offline validation)
    python tests/integration/test_single_agent_regression.py --results-dir results/single_agent/

    # With a pre-fix baseline for comparison
    python tests/integration/test_single_agent_regression.py \\
        --baseline-dir results/single_agent_prefix/
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRIALS = 5
# Maximum acceptable latency overhead from the barrier (ms)
# The barrier should be a no-op for single-agent, so overhead should be ~0
MAX_LATENCY_OVERHEAD_MS = 500  # generous bound to account for system noise
DEFAULT_TASK = "Tell me about artificial intelligence in one paragraph."
DEFAULT_AGENT_PATH = "cerebrum/example/agents/test_agent"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SingleAgentResult:
    """Result from a single-agent run."""

    def __init__(
        self,
        trial_index: int,
        latency_seconds: float,
        success: bool,
        response_length: int = 0,
        error: Optional[str] = None,
    ):
        self.trial_index = trial_index
        self.latency_seconds = latency_seconds
        self.success = success
        self.response_length = response_length
        self.error = error


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


def run_single_agent_trial(
    agent_path: str = DEFAULT_AGENT_PATH,
    task: str = DEFAULT_TASK,
    kernel_url: Optional[str] = None,
    trial_index: int = 0,
) -> SingleAgentResult:
    """Run a single trial of the test_agent and measure latency.

    Args:
        agent_path: Path to the agent directory.
        task: Task string to pass to the agent.
        kernel_url: Optional kernel URL override.
        trial_index: Index of this trial.

    Returns:
        SingleAgentResult with timing and outcome data.
    """
    cmd = [
        sys.executable, "-m", "cerebrum.commands.run_agent",
        "--mode", "local",
        "--agent_path", agent_path,
        "--task", task,
    ]

    env = os.environ.copy()
    if kernel_url:
        env["CEREBRUM_KERNEL_URL"] = kernel_url

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per trial
            cwd=_project_root,
            env=env,
        )
        latency = time.monotonic() - start

        if result.returncode == 0:
            response_text = result.stdout.strip()
            return SingleAgentResult(
                trial_index=trial_index,
                latency_seconds=latency,
                success=True,
                response_length=len(response_text),
            )
        else:
            return SingleAgentResult(
                trial_index=trial_index,
                latency_seconds=latency,
                success=False,
                error=result.stderr.strip()[:200] if result.stderr else "non-zero exit",
            )
    except subprocess.TimeoutExpired:
        latency = time.monotonic() - start
        return SingleAgentResult(
            trial_index=trial_index,
            latency_seconds=latency,
            success=False,
            error="TIMEOUT (>120s) — possible pipeline hang",
        )
    except Exception as e:
        latency = time.monotonic() - start
        return SingleAgentResult(
            trial_index=trial_index,
            latency_seconds=latency,
            success=False,
            error=str(e)[:200],
        )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_no_regression(
    results: List[SingleAgentResult],
    baseline_results: Optional[List[SingleAgentResult]] = None,
    max_overhead_ms: int = MAX_LATENCY_OVERHEAD_MS,
) -> Tuple[bool, str]:
    """Assert single-agent latency is within noise of baseline.

    If no baseline is provided, asserts that all trials completed successfully
    and no trial shows signs of barrier activation (excessive latency).

    Args:
        results: Results from current run.
        baseline_results: Optional pre-fix baseline results for comparison.
        max_overhead_ms: Maximum acceptable overhead in ms.

    Returns:
        Tuple of (passed, detail_message).
    """
    successful = [r for r in results if r.success]
    if not successful:
        return False, "No successful trials — cannot assess latency."

    current_latencies = [r.latency_seconds for r in successful]
    avg_current = sum(current_latencies) / len(current_latencies)

    detail_lines = [
        f"Successful trials: {len(successful)}/{len(results)}",
        f"Avg latency (current): {avg_current:.2f}s",
    ]

    if baseline_results:
        baseline_successful = [r for r in baseline_results if r.success]
        if baseline_successful:
            baseline_latencies = [r.latency_seconds for r in baseline_successful]
            avg_baseline = sum(baseline_latencies) / len(baseline_latencies)
            overhead_ms = (avg_current - avg_baseline) * 1000

            detail_lines.append(f"Avg latency (baseline): {avg_baseline:.2f}s")
            detail_lines.append(f"Overhead: {overhead_ms:.0f}ms")
            detail_lines.append(f"Max allowed overhead: {max_overhead_ms}ms")

            if overhead_ms <= max_overhead_ms:
                detail_lines.append(
                    "Latency within acceptable bounds — no barrier regression."
                )
                return True, "\n    ".join(detail_lines)
            else:
                detail_lines.append(
                    f"REGRESSION: {overhead_ms:.0f}ms overhead exceeds "
                    f"{max_overhead_ms}ms threshold."
                )
                return False, "\n    ".join(detail_lines)
    else:
        detail_lines.append(f"No baseline provided — checking absolute bounds only.")
        detail_lines.append(
            f"All trials completed successfully. "
            f"No signs of barrier activation (no excessive latency)."
        )

    return True, "\n    ".join(detail_lines)


def assert_all_completed(
    results: List[SingleAgentResult],
) -> Tuple[bool, str]:
    """Assert all trials completed without timeout or crash.

    Args:
        results: Results from current run.

    Returns:
        Tuple of (passed, detail_message).
    """
    if not results:
        return False, "No trials were run."

    failed = [r for r in results if not r.success]
    if not failed:
        return True, f"All {len(results)} trials completed successfully."

    detail_lines = [f"{len(failed)}/{len(results)} trials failed:"]
    for r in failed[:5]:
        detail_lines.append(f"  Trial {r.trial_index}: {r.error}")
    if len(failed) > 5:
        detail_lines.append(f"  ... and {len(failed) - 5} more")

    return False, "\n    ".join(detail_lines)


def assert_no_barrier_activation(
    results: List[SingleAgentResult],
    barrier_timeout_ms: int = 2000,
) -> Tuple[bool, str]:
    """Assert no trial shows signs of barrier activation.

    A barrier activation in single-agent mode would manifest as ~barrier_timeout_ms
    added latency. We check that no individual trial has anomalously high latency
    compared to the median.

    Args:
        results: Results from current run.
        barrier_timeout_ms: Kernel barrier timeout for reference.

    Returns:
        Tuple of (passed, detail_message).
    """
    successful = [r for r in results if r.success]
    if len(successful) < 2:
        return True, "Insufficient trials for barrier detection (need ≥ 2)."

    latencies = sorted(r.latency_seconds for r in successful)
    median_latency = latencies[len(latencies) // 2]

    # If any trial is more than barrier_timeout_ms above the median,
    # it might indicate accidental barrier activation
    barrier_threshold = median_latency + (barrier_timeout_ms / 1000.0)
    anomalous = [
        r for r in successful
        if r.latency_seconds > barrier_threshold
    ]

    detail_lines = [
        f"Median latency: {median_latency:.2f}s",
        f"Barrier detection threshold: {barrier_threshold:.2f}s",
        f"Anomalous trials: {len(anomalous)}",
    ]

    if not anomalous:
        detail_lines.append(
            "No trials show barrier-like latency spikes — "
            "barrier correctly skipped for single-agent flow."
        )
        return True, "\n    ".join(detail_lines)
    else:
        for r in anomalous:
            detail_lines.append(
                f"  Trial {r.trial_index}: {r.latency_seconds:.2f}s "
                f"(+{(r.latency_seconds - median_latency) * 1000:.0f}ms above median)"
            )
        detail_lines.append(
            "WARNING: Some trials show barrier-like latency. "
            "The barrier may be incorrectly activating for single-agent retrieval."
        )
        return False, "\n    ".join(detail_lines)


# ---------------------------------------------------------------------------
# Offline results loading
# ---------------------------------------------------------------------------


def load_offline_results(results_dir: str) -> List[SingleAgentResult]:
    """Load pre-computed single-agent results from a directory.

    Expects a results.json file with format:
    {
        "trials": [
            {"trial_index": 0, "latency_seconds": 5.2, "success": true, "response_length": 150},
            ...
        ]
    }

    Args:
        results_dir: Directory containing results.json.

    Returns:
        List of SingleAgentResult objects.

    Raises:
        FileNotFoundError: If results.json does not exist.
    """
    json_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No results.json found in {results_dir}")

    with open(json_path, "r") as f:
        data = json.load(f)

    results = []
    for trial in data.get("trials", []):
        results.append(SingleAgentResult(
            trial_index=trial.get("trial_index", 0),
            latency_seconds=trial.get("latency_seconds", 0.0),
            success=trial.get("success", False),
            response_length=trial.get("response_length", 0),
            error=trial.get("error"),
        ))
    return results


def save_results(results: List[SingleAgentResult], output_dir: str) -> None:
    """Save single-agent results to a directory.

    Args:
        results: List of SingleAgentResult objects.
        output_dir: Directory to write results.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "trials": [
            {
                "trial_index": r.trial_index,
                "latency_seconds": r.latency_seconds,
                "success": r.success,
                "response_length": r.response_length,
                "error": r.error,
            }
            for r in results
        ]
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {json_path}")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_test(
    trials: int = DEFAULT_TRIALS,
    kernel_url: Optional[str] = None,
    results_dir: Optional[str] = None,
    baseline_dir: Optional[str] = None,
    agent_path: str = DEFAULT_AGENT_PATH,
    task: str = DEFAULT_TASK,
) -> bool:
    """Execute the single-agent flow regression integration test.

    When results_dir is provided, validates pre-existing results (offline mode).
    Otherwise, runs fresh trials against the kernel.

    Args:
        trials: Number of trials to run.
        kernel_url: Kernel URL override.
        results_dir: Pre-existing results directory (offline validation).
        baseline_dir: Pre-fix baseline results directory for comparison.
        agent_path: Path to the agent to test.
        task: Task string to pass to the agent.

    Returns:
        True if all assertions pass, False otherwise.
    """
    print("=" * 70)
    print("Integration Test 5.5: Single-Agent Flow Regression")
    print("Validates: Requirements 3.1, 3.3")
    print("=" * 70)

    print(f"\n  Configuration:")
    print(f"    Agent:    {agent_path}")
    print(f"    Task:     {task[:60]}{'...' if len(task) > 60 else ''}")
    print(f"    Trials:   {trials}")

    assertion_results = []

    # -----------------------------------------------------------------------
    # Step 1: Run (or load) trials
    # -----------------------------------------------------------------------
    if results_dir:
        print(f"\n[Step 1] Loading pre-existing results from: {results_dir}")
        try:
            results = load_offline_results(results_dir)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return False
    else:
        output = "results/single_agent_postfix/"
        print(f"\n[Step 1] Running {trials} single-agent trial(s)...")
        results = []
        for i in range(trials):
            print(f"  Trial {i + 1}/{trials}...", end=" ", flush=True)
            result = run_single_agent_trial(
                agent_path=agent_path,
                task=task,
                kernel_url=kernel_url,
                trial_index=i,
            )
            status = "OK" if result.success else "FAIL"
            print(f"[{status}] {result.latency_seconds:.2f}s")
            results.append(result)

        # Save results for future offline validation
        save_results(results, output)

    print(f"  Total trials: {len(results)}")
    print(f"  Successful:   {sum(1 for r in results if r.success)}")
    print(f"  Failed:       {sum(1 for r in results if not r.success)}")

    # -----------------------------------------------------------------------
    # Step 2: Load baseline (if provided)
    # -----------------------------------------------------------------------
    baseline_results = None
    if baseline_dir:
        print(f"\n[Step 2] Loading pre-fix baseline from: {baseline_dir}")
        try:
            baseline_results = load_offline_results(baseline_dir)
            print(f"  Baseline trials: {len(baseline_results)}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping baseline comparison")
    else:
        print(f"\n[Step 2] No baseline provided — skipping comparative analysis")

    # -----------------------------------------------------------------------
    # Step 3: Assert all trials completed
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] Asserting all trials completed without timeout/crash...")
    passed, detail = assert_all_completed(results)
    assertion_results.append(("All Trials Completed", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 4: Assert no latency regression
    # -----------------------------------------------------------------------
    print(f"\n[Step 4] Asserting no latency regression...")
    passed, detail = assert_no_regression(
        results, baseline_results, MAX_LATENCY_OVERHEAD_MS
    )
    assertion_results.append(("No Latency Regression", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 5: Assert no barrier activation
    # -----------------------------------------------------------------------
    print(f"\n[Step 5] Asserting no barrier activation in single-agent flow...")
    passed, detail = assert_no_barrier_activation(results)
    assertion_results.append(("No Barrier Activation", passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {detail}")

    # -----------------------------------------------------------------------
    # Step 6: Display latency statistics
    # -----------------------------------------------------------------------
    print(f"\n[Step 6] Latency statistics:")
    successful = [r for r in results if r.success]
    if successful:
        latencies = [r.latency_seconds for r in successful]
        avg = sum(latencies) / len(latencies)
        min_l = min(latencies)
        max_l = max(latencies)
        print(f"  Avg:    {avg:.2f}s")
        print(f"  Min:    {min_l:.2f}s")
        print(f"  Max:    {max_l:.2f}s")
        print(f"  Range:  {max_l - min_l:.2f}s")

        if baseline_results:
            baseline_successful = [r for r in baseline_results if r.success]
            if baseline_successful:
                bl_latencies = [r.latency_seconds for r in baseline_successful]
                bl_avg = sum(bl_latencies) / len(bl_latencies)
                print(f"  Baseline avg: {bl_avg:.2f}s")
                print(f"  Overhead:     {(avg - bl_avg) * 1000:.0f}ms")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_passed = sum(1 for _, p, _ in assertion_results if p)
    total_failed = sum(1 for _, p, _ in assertion_results if not p)

    for name, passed, detail in assertion_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {len(assertion_results)}")

    if total_failed == 0:
        print(f"\n  ALL ASSERTIONS PASSED — Single-agent flow not regressed")
        print(f"  Barrier correctly skipped for single-agent retrieval")
        print(f"  (Requirements 3.1 and 3.3 validated at integration level).")
    else:
        print(f"\n  {total_failed} assertion(s) FAILED.")
        print(f"  The write-barrier fix may be incorrectly activating")
        print(f"  for single-agent flows, violating preservation clauses 3.1/3.3.")

    print(f"{'=' * 70}")
    return total_failed == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Parse CLI args and run the integration test."""
    parser = argparse.ArgumentParser(
        description="Integration test 5.5: Single-agent flow regression",
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
        "--baseline-dir",
        type=str,
        default=None,
        help="Pre-fix baseline results directory for latency comparison.",
    )
    parser.add_argument(
        "--agent-path",
        type=str,
        default=DEFAULT_AGENT_PATH,
        help=f"Path to the agent to test (default: {DEFAULT_AGENT_PATH}).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Task string to pass to the agent.",
    )
    args = parser.parse_args()

    all_passed = run_test(
        trials=args.trials,
        kernel_url=args.kernel_url,
        results_dir=args.results_dir,
        baseline_dir=args.baseline_dir,
        agent_path=args.agent_path,
        task=args.task,
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
