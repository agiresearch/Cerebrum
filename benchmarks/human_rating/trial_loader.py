"""Trial discovery, normalization, and eligibility validation.

Loads finalized GPT-4o benchmark result files and converts them into
validated SourceTrial objects using the compatible protocol:
- integration_score as judge_score
- Source-grounded reference context (not exact model-visible)
- No profile/task stratification
- 6 unique trials per method

Public functions:
- discover_result_files() — find result JSONs by method
- load_result_file() — parse a single result file into SourceTrial objects
- load_eligible_trials() — load all methods and validate pool eligibility
- summarize_trial_pool() — produce a developer-facing summary
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from benchmarks.human_rating.schemas import (
    SourceTrial,
    VALID_SOURCE_METHODS,
    EVALUATION_DIMENSION,
    UNIQUE_ITEMS_PER_METHOD,
)
from benchmarks.human_rating.context_formatter import format_reference_context

logger = logging.getLogger(__name__)

ALLOWED_METHODS: list[str] = sorted(VALID_SOURCE_METHODS)


# ---------------------------------------------------------------------------
# Exclusion reporting
# ---------------------------------------------------------------------------

@dataclass
class ExclusionEntry:
    """One excluded trial with structured reason."""
    method: str
    trial_index: int
    reason: str
    source_file: str


@dataclass
class ExclusionReport:
    """Structured report of all excluded trials."""
    entries: list[ExclusionEntry] = field(default_factory=list)

    @property
    def total_excluded(self) -> int:
        return len(self.entries)

    def by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.reason] = counts.get(e.reason, 0) + 1
        return counts

    def format_summary(self) -> str:
        if not self.entries:
            return "No trials excluded."
        lines = [f"Excluded {self.total_excluded} trial(s):"]
        for reason, count in sorted(self.by_reason().items()):
            lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_result_files(
    results_root: Path | str,
    methods: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Discover finalized result JSON files by method."""
    results_root = Path(results_root)
    if methods is None:
        methods = ALLOWED_METHODS

    discovered: dict[str, Path] = {}
    for method in sorted(methods):
        candidates: list[Path] = []
        p1 = results_root / f"gpt4o_{method}" / f"results_{method}.json"
        if p1.exists():
            candidates.append(p1)
        p2 = results_root / f"results_{method}.json"
        if p2.exists() and p2 not in candidates:
            candidates.append(p2)

        if not candidates:
            raise FileNotFoundError(
                f"No result file for method '{method}' under '{results_root}'"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple result files for method '{method}': {candidates}"
            )
        discovered[method] = candidates[0]

    return discovered


# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------

def load_result_file(
    path: Path | str,
    expected_method: str,
    exclusion_report: ExclusionReport | None = None,
) -> list[SourceTrial]:
    """Load and normalize trials from a single result JSON file.

    Uses integration_score as judge_score. Formats reference context from
    stored synthetic_profile + synthetic_task_context.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    if exclusion_report is None:
        exclusion_report = ExclusionReport()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conditions = data.get("conditions", [])
    if not conditions:
        raise ValueError(f"No conditions in {path}")

    target = None
    for cond in conditions:
        if cond.get("condition") == expected_method:
            target = cond
            break
    if target is None:
        raise ValueError(f"{path} does not contain method '{expected_method}'")

    trials_raw = target.get("trials", [])
    if not trials_raw:
        raise ValueError(f"No trials for method '{expected_method}' in {path}")

    source_trials: list[SourceTrial] = []
    seen_ids: set[str] = set()

    for trial in trials_raw:
        trial_index = trial.get("trial_index")
        if trial_index is None:
            raise ValueError(f"Trial in {path} missing 'trial_index'")

        if trial.get("failed", False):
            exclusion_report.entries.append(ExclusionEntry(
                method=expected_method, trial_index=trial_index,
                reason="failed_trial", source_file=str(path),
            ))
            continue

        original_trial_id = str(trial_index)
        if (expected_method, original_trial_id) in seen_ids:
            raise ValueError(f"Duplicate trial {original_trial_id} in {path}")
        seen_ids.add((expected_method, original_trial_id))

        question = trial.get("follow_up_query", "")
        response = trial.get("assistant_response", "")

        if not question or not question.strip():
            exclusion_report.entries.append(ExclusionEntry(
                method=expected_method, trial_index=trial_index,
                reason="empty_question", source_file=str(path),
            ))
            continue
        if not response or not response.strip():
            exclusion_report.entries.append(ExclusionEntry(
                method=expected_method, trial_index=trial_index,
                reason="empty_response", source_file=str(path),
            ))
            continue

        # Extract all three scores
        pu = trial.get("profile_usage_score")
        tu = trial.get("task_usage_score")
        ig = trial.get("integration_score")

        scores_valid = True
        for name, val in [("profile_usage_score", pu), ("task_usage_score", tu), ("integration_score", ig)]:
            if val is None or not isinstance(val, int) or val < 1 or val > 5:
                exclusion_report.entries.append(ExclusionEntry(
                    method=expected_method, trial_index=trial_index,
                    reason=f"invalid_{name}", source_file=str(path),
                ))
                scores_valid = False
                break
        if not scores_valid:
            continue

        # Format reference context from synthetic data
        syn_profile = trial.get("synthetic_profile")
        syn_task = trial.get("synthetic_task_context")
        if not syn_profile and not syn_task:
            exclusion_report.entries.append(ExclusionEntry(
                method=expected_method, trial_index=trial_index,
                reason="missing_synthetic_context", source_file=str(path),
            ))
            continue

        try:
            reference_context = format_reference_context(syn_profile, syn_task)
        except ValueError:
            exclusion_report.entries.append(ExclusionEntry(
                method=expected_method, trial_index=trial_index,
                reason="context_format_error", source_file=str(path),
            ))
            continue

        source_trials.append(SourceTrial(
            source_method=expected_method,
            original_trial_id=original_trial_id,
            model="gpt-4o",
            evaluation_dimension=EVALUATION_DIMENSION,
            reference_context=reference_context,
            reference_context_provenance="synthetic_source_context",
            is_exact_model_visible_context=False,
            question=question,
            response=response,
            judge_score=ig,
            profile_usage_score=pu,
            task_usage_score=tu,
            integration_score=ig,
        ))

    return source_trials


# ---------------------------------------------------------------------------
# Multi-file loading with eligibility
# ---------------------------------------------------------------------------

def load_eligible_trials(
    method_paths: dict[str, str | Path] | None = None,
    results_root: str | Path | None = None,
    methods: Sequence[str] | None = None,
    min_per_method: int = UNIQUE_ITEMS_PER_METHOD,
) -> tuple[list[SourceTrial], ExclusionReport]:
    """Load all methods and validate pool eligibility."""
    if methods is None:
        methods = ALLOWED_METHODS

    if method_paths is not None:
        resolved = {m: Path(p) for m, p in method_paths.items()}
        missing = set(methods) - set(resolved.keys())
        if missing:
            raise ValueError(f"Missing method paths: {sorted(missing)}")
    elif results_root is not None:
        resolved = discover_result_files(results_root, methods)
    else:
        raise ValueError("Either method_paths or results_root required")

    exclusion_report = ExclusionReport()
    all_trials: list[SourceTrial] = []

    for method in sorted(methods):
        trials = load_result_file(resolved[method], method, exclusion_report)
        logger.info("Loaded %d trials from %s", len(trials), resolved[method])
        all_trials.extend(trials)

    # Eligibility: each method must have at least min_per_method trials
    for method in sorted(methods):
        count = sum(1 for t in all_trials if t.source_method == method)
        if count < min_per_method:
            raise ValueError(
                f"Insufficient trials for method={method}: "
                f"required {min_per_method}, found {count}."
            )

    return all_trials, exclusion_report


# ---------------------------------------------------------------------------
# Pool summary
# ---------------------------------------------------------------------------

@dataclass
class TrialPoolSummary:
    """Developer-facing summary."""
    total_trials: int
    methods: list[str]
    method_counts: dict[str, int]
    score_distribution: dict[int, int]

    def format_table(self) -> str:
        lines = [f"{'Method':<20} {'Count':>6}"]
        lines.append("-" * 28)
        for m in self.methods:
            lines.append(f"{m:<20} {self.method_counts[m]:>6}")
        lines.append("-" * 28)
        lines.append(f"{'TOTAL':<20} {self.total_trials:>6}")
        return "\n".join(lines)


def summarize_trial_pool(trials: Sequence[SourceTrial]) -> TrialPoolSummary:
    methods = sorted({t.source_method for t in trials})
    counts = {m: sum(1 for t in trials if t.source_method == m) for m in methods}
    scores: dict[int, int] = {}
    for t in trials:
        scores[t.judge_score] = scores.get(t.judge_score, 0) + 1
    return TrialPoolSummary(
        total_trials=len(trials), methods=methods,
        method_counts=counts, score_distribution=scores,
    )
