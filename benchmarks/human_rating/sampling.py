"""Deterministic stratified sampling of unique trials for human rating.

Selects exactly six eligible unique trials from each method using a
reproducible seed, producing a 24-item internal selection.

This module performs selection only. It does not create duplicate items,
blinded IDs, a rater-facing queue, or the private answer key.

Sampling design:
- 4 methods × 6 unique trials = 24 total
- Method pools are sampled independently (per-method derived seed)
- Input ordering does not affect selection
- Same seed + same trials = identical output
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from benchmarks.human_rating.schemas import (
    SourceTrial,
    VALID_SOURCE_METHODS,
    EVALUATION_DIMENSION,
    UNIQUE_ITEMS_PER_METHOD,
)


# ---------------------------------------------------------------------------
# Stable sorting key
# ---------------------------------------------------------------------------

def stable_trial_key(trial: SourceTrial) -> tuple[str, str]:
    """Stable sort key independent of input order."""
    return (trial.source_method, trial.original_trial_id)


# ---------------------------------------------------------------------------
# Per-method seed derivation
# ---------------------------------------------------------------------------

def derive_method_seed(seed: int, method: str) -> int:
    """Derive a deterministic per-method seed using SHA-256.

    This ensures one method's candidate count or ordering cannot change
    the selections for another method.

    Args:
        seed: Master sampling seed.
        method: Method name string.

    Returns:
        A deterministic integer seed for this method's RNG.
    """
    digest = hashlib.sha256(f"{seed}:{method}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Eligibility validation
# ---------------------------------------------------------------------------

def _validate_trial(trial: SourceTrial) -> str | None:
    """Validate a single trial for sampling eligibility.

    Returns None if valid, or an error message string if invalid.
    """
    if trial.source_method not in VALID_SOURCE_METHODS:
        return f"invalid source_method: {trial.source_method!r}"
    if trial.evaluation_dimension != EVALUATION_DIMENSION:
        return f"wrong evaluation_dimension: {trial.evaluation_dimension!r}"
    if trial.reference_context_provenance != "synthetic_source_context":
        return f"wrong provenance: {trial.reference_context_provenance!r}"
    if trial.is_exact_model_visible_context is not False:
        return "is_exact_model_visible_context must be False"
    if not trial.reference_context or not trial.reference_context.strip():
        return "empty reference_context"
    if not trial.question or not trial.question.strip():
        return "empty question"
    if not trial.response or not trial.response.strip():
        return "empty response"
    if not isinstance(trial.judge_score, int) or trial.judge_score < 1 or trial.judge_score > 5:
        return f"invalid judge_score: {trial.judge_score!r}"
    if trial.judge_score != trial.integration_score:
        return f"judge_score ({trial.judge_score}) != integration_score ({trial.integration_score})"
    return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingSummary:
    """Summary of the sampling operation."""
    seed: int
    eligible_count_by_method: dict[str, int]
    selected_count_by_method: dict[str, int]
    selected_trial_ids_by_method: dict[str, tuple[str, ...]]

    def format_table(self) -> str:
        """Format as a readable developer table."""
        lines = []
        header = f"{'Method':<20} {'Eligible':>9} {'Selected':>9}  Trial IDs"
        lines.append(header)
        lines.append("-" * 80)
        for method in sorted(self.eligible_count_by_method.keys()):
            eligible = self.eligible_count_by_method[method]
            selected = self.selected_count_by_method[method]
            ids = ", ".join(self.selected_trial_ids_by_method[method])
            lines.append(f"{method:<20} {eligible:>9} {selected:>9}  {ids}")
        lines.append("-" * 80)
        total_eligible = sum(self.eligible_count_by_method.values())
        total_selected = sum(self.selected_count_by_method.values())
        lines.append(f"{'TOTAL':<20} {total_eligible:>9} {total_selected:>9}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main sampling function
# ---------------------------------------------------------------------------

def sample_unique_trials(
    trials: Sequence[SourceTrial],
    *,
    seed: int,
    items_per_method: int = UNIQUE_ITEMS_PER_METHOD,
) -> tuple[list[SourceTrial], SamplingSummary]:
    """Select exactly items_per_method unique trials from each method.

    Args:
        trials: Complete pool of eligible SourceTrial objects.
        seed: Master seed for reproducibility.
        items_per_method: Number of trials to select per method (default 6).

    Returns:
        Tuple of (selected trials in deterministic order, SamplingSummary).

    Raises:
        ValueError: If any method has insufficient eligible trials,
            if unexpected methods are present, if required methods are
            missing, or if duplicate composite identities exist.
    """
    # --- Validate all trials ---
    for trial in trials:
        err = _validate_trial(trial)
        if err is not None:
            raise ValueError(
                f"Trial {trial.source_method}/{trial.original_trial_id} "
                f"failed eligibility: {err}"
            )

    # --- Check for duplicate composite identities ---
    seen_keys: set[tuple[str, str]] = set()
    for trial in trials:
        key = (trial.source_method, trial.original_trial_id)
        if key in seen_keys:
            raise ValueError(
                f"Duplicate composite trial identity: "
                f"method={trial.source_method}, id={trial.original_trial_id}"
            )
        seen_keys.add(key)

    # --- Build per-method pools (sorted for stability) ---
    method_pools: dict[str, list[SourceTrial]] = {m: [] for m in sorted(VALID_SOURCE_METHODS)}

    for trial in trials:
        if trial.source_method not in VALID_SOURCE_METHODS:
            raise ValueError(f"Unexpected method: {trial.source_method!r}")
        method_pools[trial.source_method].append(trial)

    # Sort each pool by stable key
    for method in method_pools:
        method_pools[method].sort(key=stable_trial_key)

    # --- Validate all four methods present with sufficient trials ---
    for method in sorted(VALID_SOURCE_METHODS):
        count = len(method_pools[method])
        if count == 0:
            raise ValueError(f"Missing method in pool: {method}")
        if count < items_per_method:
            raise ValueError(
                f"Insufficient eligible trials for {method}: "
                f"required {items_per_method}, found {count}."
            )

    # --- Sample independently per method ---
    selected: list[SourceTrial] = []
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    selected_ids: dict[str, tuple[str, ...]] = {}

    for method in sorted(VALID_SOURCE_METHODS):
        pool = method_pools[method]
        eligible_counts[method] = len(pool)

        method_seed = derive_method_seed(seed, method)
        rng = random.Random(method_seed)
        chosen = rng.sample(pool, k=items_per_method)

        # Sort chosen by stable key for deterministic internal order
        chosen.sort(key=stable_trial_key)

        selected.extend(chosen)
        selected_counts[method] = len(chosen)
        selected_ids[method] = tuple(t.original_trial_id for t in chosen)

    summary = SamplingSummary(
        seed=seed,
        eligible_count_by_method=eligible_counts,
        selected_count_by_method=selected_counts,
        selected_trial_ids_by_method=selected_ids,
    )

    return selected, summary
