#!/usr/bin/env python3
"""Post-run injection-rate analyzer for the shared-memory benchmark.

Loads the per-trial output written by ``benchmarks/shared_memory/run_evaluation.py``
(``results.json``, a ``results_*.json`` variant, or ``results.csv``) and
computes the rate of trials where the kernel-managed write barrier produced a
confirmed cross-agent injection.

The rate is the proportion of non-failed trials whose
``retrieval_log.injection_status == "confirmed"``. The script exits 0 only
when both:

* the rate is greater than or equal to ``--min-rate`` (default ``0.99``)
* the analyzed trial count is greater than or equal to ``--min-trials``
  (default ``100``)

Use this script to gate kernel-memory-write-barrier integration tasks:

* Task 5.1 — Phase 2 determinism (per-run rate, ``--min-rate 1.0``)
* Task 5.2 — Fast assistant model (GPT-4o), ``--min-rate 0.99``
* Task 5.3 — Slow assistant model (qwen2.5:7b), ``--min-rate 0.99``
* Task 5.5 — Single-agent regression (rate is a baseline check; pass
  ``--min-rate 0.0`` so the trial-count assertion is the only gate)

Stdlib only — no new dependencies. Compatible with Python 3.10+.

Output format::

    [assert_injection_rate] source: results/post_fix_gpt4o/results.json (json)
    [assert_injection_rate] analyzed trials: 100
    [assert_injection_rate] failed trials (excluded): 0
    [assert_injection_rate] injection_status histogram:
      confirmed       : 100
    [assert_injection_rate] confirmed rate: 1.0000 (100/100)
    [assert_injection_rate] 95% Wilson lower bound: 0.9633
    [assert_injection_rate] threshold: --min-rate=0.99 --min-trials=100
    [assert_injection_rate] PASS: confirmed rate 1.0000 >= 0.99 over 100 trials
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from typing import List, Tuple

# Status emitted by ``benchmarks/shared_memory/pipeline.py`` when the kernel's
# auto_inject path returned diagnostics confirming that one or more cross-agent
# memories were prepended to the AssistantAgent prompt.
CONFIRMED_STATUS = "confirmed"

# Synthetic statuses used when no per-trial ``injection_status`` is available
# (e.g. when only the CSV file is present — see ``_load_from_csv``).
INFERRED_CONFIRMED = "confirmed_inferred"
INFERRED_UNKNOWN = "unknown_inferred"

# Synthetic statuses used during JSON parsing for trials that failed or are
# missing a retrieval log entirely.
STATUS_FAILED = "<failed-trial>"
STATUS_MISSING_LOG = "<missing-retrieval-log>"


def _wilson_score_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Compute the Wilson-score 95% lower bound for a proportion.

    Args:
        successes: Number of confirmed-injection trials.
        total: Total number of trials counted toward the rate.
        z: Standard normal critical value (default 1.96 for ~95% CI).

    Returns:
        Lower bound of the Wilson score interval, in [0.0, 1.0]. Returns
        0.0 when ``total`` is zero.
    """
    if total <= 0:
        return 0.0
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt(
        (p_hat * (1.0 - p_hat) + (z * z) / (4 * total)) / total
    ) / denom
    return max(0.0, center - margin)


def _iter_trials_from_json(payload: dict):
    """Yield raw trial dicts from a parsed ``results.json`` payload.

    The harness writes either a single-condition or a multi-condition
    ``ExperimentResults`` document. Both shapes nest trials under
    ``conditions[*].trials``.
    """
    for cond in payload.get("conditions") or []:
        for trial in cond.get("trials") or []:
            yield trial


def _load_from_json(path: str) -> Tuple[List[str], int]:
    """Load injection-status values from a results.json document.

    Args:
        path: Absolute path to a ``results.json`` (or ``results_*.json``)
            file emitted by ``run_evaluation.py``.

    Returns:
        Tuple ``(statuses, failed_count)`` where ``statuses`` contains one
        entry per non-failed trial and ``failed_count`` is the number of
        trials marked ``failed=True`` in the document.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    statuses: List[str] = []
    failed = 0
    for trial in _iter_trials_from_json(payload):
        if trial.get("failed"):
            failed += 1
            continue
        log = trial.get("retrieval_log")
        if not isinstance(log, dict):
            statuses.append(STATUS_MISSING_LOG)
            continue
        statuses.append(str(log.get("injection_status", STATUS_MISSING_LOG)))
    return statuses, failed


def _load_from_csv(path: str) -> Tuple[List[str], int]:
    """Infer injection status from a results.csv document.

    The CSV emitted by ``ResultsWriter.write_csv`` does not include the raw
    ``injection_status`` field — it only has ``shared_memory_count`` and
    ``cross_agent_found``. Treat ``shared_memory_count > 0`` AND
    ``cross_agent_found == True`` as a confirmed injection (synthetic
    status ``confirmed_inferred``); everything else is unknown.

    Args:
        path: Absolute path to a ``results.csv`` file.

    Returns:
        Tuple ``(statuses, failed_count)``. CSV rows do not encode trial
        failure, so ``failed_count`` is always ``0`` here.
    """
    statuses: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                count = int((row.get("shared_memory_count") or "0").strip())
            except ValueError:
                count = 0
            cross_raw = (row.get("cross_agent_found") or "").strip().lower()
            cross = cross_raw in ("true", "1", "yes")
            if count > 0 and cross:
                statuses.append(INFERRED_CONFIRMED)
            else:
                statuses.append(INFERRED_UNKNOWN)
    return statuses, 0


def _resolve_results_file(results_dir: str) -> Tuple[str, str]:
    """Pick the best results file in ``results_dir`` and report its format.

    JSON is preferred because it carries the explicit
    ``retrieval_log.injection_status`` field. CSV is the documented
    fallback and uses an inferred status (less precise — see
    ``_load_from_csv``).

    Returns:
        Tuple ``(absolute_path, source_label)`` where ``source_label`` is
        either ``"json"`` or ``"csv"``.

    Raises:
        FileNotFoundError: when neither a JSON nor a CSV results file is
            present in ``results_dir``.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results-dir is not a directory: {results_dir}")

    json_candidates: List[str] = []
    csv_candidates: List[str] = []
    for name in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, name)
        if not os.path.isfile(full):
            continue
        if name.endswith(".json"):
            json_candidates.append(full)
        elif name.endswith(".csv"):
            csv_candidates.append(full)

    # Prefer the canonical "results.json" name when present; otherwise the
    # first json candidate (e.g. results_all_methods.json or
    # results_kernel_shared.json under the --method workflow).
    for candidate in json_candidates:
        if os.path.basename(candidate) == "results.json":
            return candidate, "json"
    if json_candidates:
        return json_candidates[0], "json"
    if csv_candidates:
        return csv_candidates[0], "csv"
    raise FileNotFoundError(
        f"No results.json or results.csv found in {results_dir}"
    )


def analyze(results_dir: str, min_rate: float, min_trials: int) -> int:
    """Run the analysis and print the pass/fail report.

    Args:
        results_dir: Directory containing the harness output.
        min_rate: Minimum required confirmed-injection rate, in [0.0, 1.0].
        min_trials: Minimum number of analyzed (non-failed) trials.

    Returns:
        Process exit code: ``0`` on PASS, ``1`` on FAIL.
    """
    path, source = _resolve_results_file(results_dir)
    if source == "json":
        statuses, failed_count = _load_from_json(path)
    else:
        statuses, failed_count = _load_from_csv(path)

    histogram = Counter(statuses)
    total = len(statuses)
    confirmed = histogram.get(CONFIRMED_STATUS, 0) + histogram.get(
        INFERRED_CONFIRMED, 0
    )
    rate = (confirmed / total) if total else 0.0
    wilson = _wilson_score_lower_bound(confirmed, total)

    print(f"[assert_injection_rate] source: {path} ({source})")
    print(f"[assert_injection_rate] analyzed trials: {total}")
    print(f"[assert_injection_rate] failed trials (excluded): {failed_count}")
    print("[assert_injection_rate] injection_status histogram:")
    if histogram:
        # Sort by count desc, then by name for stable output.
        for status, count in sorted(
            histogram.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"  {status:<24}: {count}")
    else:
        print("  (no trial rows found)")
    print(
        f"[assert_injection_rate] confirmed rate: {rate:.4f} "
        f"({confirmed}/{total})"
    )
    print(f"[assert_injection_rate] 95% Wilson lower bound: {wilson:.4f}")
    print(
        f"[assert_injection_rate] threshold: --min-rate={min_rate} "
        f"--min-trials={min_trials}"
    )

    if source == "csv":
        print(
            "[assert_injection_rate] note: CSV input does not carry the "
            "explicit injection_status field; the rate uses an inferred "
            "'shared_memory_count > 0 AND cross_agent_found' rule. Re-run "
            "the harness with the JSON output for an authoritative rate."
        )

    if total < min_trials:
        print(
            f"[assert_injection_rate] FAIL: only {total} trials analyzed "
            f"(< --min-trials={min_trials})"
        )
        return 1
    if rate < min_rate:
        print(
            f"[assert_injection_rate] FAIL: confirmed rate {rate:.4f} "
            f"< --min-rate={min_rate} over {total} trials"
        )
        return 1

    print(
        f"[assert_injection_rate] PASS: confirmed rate {rate:.4f} "
        f">= {min_rate} over {total} trials"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="assert_injection_rate.py",
        description=(
            "Assert kernel auto_inject confirmed-injection rate over a "
            "results directory written by run_evaluation.py."
        ),
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help=(
            "Directory containing results.json (preferred) or results.csv "
            "from benchmarks/shared_memory/run_evaluation.py."
        ),
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.99,
        help=(
            "Minimum required confirmed-injection rate (default: 0.99). "
            "Pass 0.0 to skip the rate gate (e.g. for the single-agent "
            "regression in task 5.5 where the barrier should not engage)."
        ),
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=100,
        help="Minimum number of analyzed (non-failed) trials (default: 100).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (0.0 <= args.min_rate <= 1.0):
        parser.error("--min-rate must be in [0.0, 1.0]")
    if args.min_trials < 0:
        parser.error("--min-trials must be non-negative")
    try:
        return analyze(
            results_dir=args.results_dir,
            min_rate=args.min_rate,
            min_trials=args.min_trials,
        )
    except FileNotFoundError as exc:
        print(f"[assert_injection_rate] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
