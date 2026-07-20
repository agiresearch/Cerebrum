"""Duplicate selection, blinded ID assignment, and constrained ordering.

Transforms 24 selected unique trials into a 30-item blinded plan by:
1. Selecting 6 trials for repeat rating (balanced across methods)
2. Assigning opaque blinded IDs to all 30 appearances
3. Producing a deterministic shuffled order with minimum duplicate separation

This module creates the in-memory plan only. Writing the rater queue and
private answer-key files belongs to a later subtask.
"""

import hashlib
import hmac
import random
from dataclasses import dataclass
from typing import Sequence

from benchmarks.human_rating.schemas import (
    SourceTrial,
    VALID_SOURCE_METHODS,
    DUPLICATE_COUNT,
    TOTAL_RATING_ITEMS,
    UNIQUE_ITEMS_PER_METHOD,
)

# Maximum shuffle attempts before failing
MAX_SHUFFLE_ATTEMPTS = 10_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedRatingItem:
    """A single appearance in the blinded rating plan."""
    blinded_id: str
    source_trial: SourceTrial
    duplicate_of_blinded_id: str | None
    appearance_index: int


@dataclass(frozen=True)
class BlindedPlan:
    """Complete blinded rating plan (private developer data)."""
    seed: int
    items: tuple[PlannedRatingItem, ...]
    duplicated_source_identities: tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# Blinded ID generation
# ---------------------------------------------------------------------------

def _generate_blinded_id(seed: int, appearance_key: str) -> str:
    """Generate an opaque blinded ID using HMAC-SHA256.

    Format: HR-XXXXXX (6 uppercase hex characters from HMAC digest).
    """
    digest = hmac.new(
        key=str(seed).encode("utf-8"),
        msg=appearance_key.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    # Use first 6 hex chars, uppercased
    return f"HR-{digest[:6].upper()}"


# ---------------------------------------------------------------------------
# Duplicate selection
# ---------------------------------------------------------------------------

def _select_duplicates(
    selected_trials: list[SourceTrial],
    *,
    seed: int,
    duplicate_count: int,
) -> list[tuple[str, str]]:
    """Select distinct source trials for duplication, balanced across methods.

    Allocation rule:
    - At least 1 duplicate from every method (4 methods × 1 = 4)
    - Remaining 2 assigned to methods chosen deterministically by seed
    - No method contributes more than 2 duplicates

    Args:
        selected_trials: The 24 unique trials (sorted by stable key).
        seed: Blinding seed for deterministic selection.
        duplicate_count: Number of duplicates to create (6).

    Returns:
        List of (source_method, original_trial_id) tuples to duplicate.
    """
    methods = sorted(VALID_SOURCE_METHODS)
    assert duplicate_count == 6, f"Expected 6 duplicates, got {duplicate_count}"

    # Build per-method pools (already sorted by caller)
    method_pools: dict[str, list[SourceTrial]] = {m: [] for m in methods}
    for t in selected_trials:
        method_pools[t.source_method].append(t)

    rng = random.Random(seed)

    # Step 1: Select 1 from each method (guaranteed minimum)
    chosen: list[tuple[str, str]] = []
    for method in methods:
        pool = method_pools[method]
        pick = rng.choice(pool)
        chosen.append((pick.source_method, pick.original_trial_id))

    # Step 2: Select 2 more from methods chosen by seed (no method > 2 total)
    # Determine which methods get a second duplicate
    eligible_for_second = list(methods)  # all methods eligible initially
    extra_methods = rng.sample(eligible_for_second, k=2)

    for method in extra_methods:
        pool = method_pools[method]
        # Exclude already-chosen trial from this method
        already_chosen_ids = {tid for m, tid in chosen if m == method}
        remaining = [t for t in pool if t.original_trial_id not in already_chosen_ids]
        if not remaining:
            # Fallback: pick from another method that still has room
            # This shouldn't happen with 6 trials per method
            raise ValueError(
                f"Cannot select second duplicate from {method}: "
                f"no remaining trials after first selection"
            )
        pick = rng.choice(remaining)
        chosen.append((pick.source_method, pick.original_trial_id))

    assert len(chosen) == duplicate_count
    return chosen


# ---------------------------------------------------------------------------
# Constrained shuffle
# ---------------------------------------------------------------------------

def _satisfies_separation(
    items: list[PlannedRatingItem],
    min_separation: int,
) -> bool:
    """Check that all duplicate pairs satisfy minimum positional distance."""
    # Build mapping: source identity -> list of positions
    identity_positions: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(items):
        key = (item.source_trial.source_method, item.source_trial.original_trial_id)
        identity_positions.setdefault(key, []).append(i)

    for positions in identity_positions.values():
        if len(positions) == 2:
            if abs(positions[0] - positions[1]) < min_separation:
                return False
    return True


def _constrained_shuffle(
    items: list[PlannedRatingItem],
    *,
    rng: random.Random,
    min_separation: int,
    max_attempts: int = MAX_SHUFFLE_ATTEMPTS,
) -> list[PlannedRatingItem]:
    """Shuffle items with minimum duplicate separation constraint.

    Raises ValueError if no valid ordering is found within max_attempts.
    """
    for attempt in range(max_attempts):
        candidate = list(items)
        rng.shuffle(candidate)
        if _satisfies_separation(candidate, min_separation):
            return candidate

    raise ValueError(
        f"Unable to produce a valid blinded order with minimum duplicate "
        f"separation {min_separation} after {max_attempts} attempts."
    )


# ---------------------------------------------------------------------------
# Main plan builder
# ---------------------------------------------------------------------------

def build_blinded_plan(
    selected_trials: Sequence[SourceTrial],
    *,
    seed: int,
    duplicate_count: int = DUPLICATE_COUNT,
    min_duplicate_separation: int = 3,
) -> BlindedPlan:
    """Build a 30-item blinded rating plan from 24 selected unique trials.

    Args:
        selected_trials: Exactly 24 unique source trials.
        seed: Blinding seed for deterministic ID generation and ordering.
        duplicate_count: Number of duplicates to create (default 6).
        min_duplicate_separation: Minimum positions between duplicate pair.

    Returns:
        BlindedPlan with 30 items in deterministic shuffled order.

    Raises:
        ValueError: If input is invalid or separation constraint unsatisfiable.
    """
    expected_unique = len(VALID_SOURCE_METHODS) * UNIQUE_ITEMS_PER_METHOD

    # Check for duplicate composite identities in input (before size check,
    # so that a 24-item input with a repeated identity gets a clear
    # duplicate-specific error rather than a misleading size error)
    seen: set[tuple[str, str]] = set()
    for t in selected_trials:
        key = (t.source_method, t.original_trial_id)
        if key in seen:
            raise ValueError(
                f"Duplicate source trial identity: "
                f"{t.source_method} / {t.original_trial_id}"
            )
        seen.add(key)

    if len(selected_trials) != expected_unique:
        raise ValueError(
            f"Expected exactly {expected_unique} unique trials, "
            f"got {len(selected_trials)}"
        )

    # Sort for stability
    sorted_trials = sorted(selected_trials, key=lambda t: (t.source_method, t.original_trial_id))

    # Select which trials to duplicate
    dup_identities = _select_duplicates(
        sorted_trials, seed=seed, duplicate_count=duplicate_count,
    )

    # Build original appearances
    originals: list[PlannedRatingItem] = []
    original_id_map: dict[tuple[str, str], str] = {}  # source identity -> blinded_id

    for i, trial in enumerate(sorted_trials):
        appearance_key = f"original:{trial.source_method}:{trial.original_trial_id}"
        blinded_id = _generate_blinded_id(seed, appearance_key)
        original_id_map[(trial.source_method, trial.original_trial_id)] = blinded_id
        originals.append(PlannedRatingItem(
            blinded_id=blinded_id,
            source_trial=trial,
            duplicate_of_blinded_id=None,
            appearance_index=-1,  # will be set after shuffle
        ))

    # Build duplicate appearances
    duplicates: list[PlannedRatingItem] = []
    for method, trial_id in dup_identities:
        trial = next(
            t for t in sorted_trials
            if t.source_method == method and t.original_trial_id == trial_id
        )
        appearance_key = f"duplicate:{method}:{trial_id}"
        blinded_id = _generate_blinded_id(seed, appearance_key)
        original_blinded_id = original_id_map[(method, trial_id)]
        duplicates.append(PlannedRatingItem(
            blinded_id=blinded_id,
            source_trial=trial,
            duplicate_of_blinded_id=original_blinded_id,
            appearance_index=-1,
        ))

    # Validate uniqueness of all blinded IDs
    all_ids = [item.blinded_id for item in originals + duplicates]
    if len(set(all_ids)) != len(all_ids):
        # Extremely unlikely with HMAC, but check anyway
        raise ValueError("Blinded ID collision detected")

    # Combine and shuffle with separation constraint
    all_items = originals + duplicates
    rng = random.Random(seed)
    shuffled = _constrained_shuffle(
        all_items, rng=rng, min_separation=min_duplicate_separation,
    )

    # Assign final appearance indices
    final_items = tuple(
        PlannedRatingItem(
            blinded_id=item.blinded_id,
            source_trial=item.source_trial,
            duplicate_of_blinded_id=item.duplicate_of_blinded_id,
            appearance_index=idx,
        )
        for idx, item in enumerate(shuffled)
    )

    return BlindedPlan(
        seed=seed,
        items=final_items,
        duplicated_source_identities=tuple(dup_identities),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_blinded_plan(
    plan: BlindedPlan,
    *,
    expected_unique_count: int = 24,
    expected_duplicate_count: int = DUPLICATE_COUNT,
    minimum_duplicate_separation: int = 3,
) -> None:
    """Validate a blinded plan comprehensively.

    Raises ValueError with a descriptive message on any violation.
    """
    items = plan.items
    total = expected_unique_count + expected_duplicate_count

    # Total count
    if len(items) != total:
        raise ValueError(f"Expected {total} items, got {len(items)}")

    # Unique blinded IDs
    ids = [item.blinded_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate blinded IDs found")

    # Count originals and duplicates
    originals = [i for i in items if i.duplicate_of_blinded_id is None]
    duplicates = [i for i in items if i.duplicate_of_blinded_id is not None]

    if len(originals) != expected_unique_count:
        raise ValueError(
            f"Expected {expected_unique_count} originals, got {len(originals)}"
        )
    if len(duplicates) != expected_duplicate_count:
        raise ValueError(
            f"Expected {expected_duplicate_count} duplicates, got {len(duplicates)}"
        )

    # Duplicate references resolve to originals
    original_ids = {i.blinded_id for i in originals}
    for dup in duplicates:
        if dup.duplicate_of_blinded_id not in original_ids:
            raise ValueError(
                f"Duplicate {dup.blinded_id} references non-original "
                f"{dup.duplicate_of_blinded_id}"
            )

    # Original and duplicate content match
    id_to_item = {i.blinded_id: i for i in items}
    for dup in duplicates:
        original = id_to_item[dup.duplicate_of_blinded_id]
        if dup.source_trial != original.source_trial:
            raise ValueError(
                f"Duplicate {dup.blinded_id} content differs from "
                f"original {original.blinded_id}"
            )

    # Source identities appear at most twice
    identity_counts: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.source_trial.source_method, item.source_trial.original_trial_id)
        identity_counts[key] = identity_counts.get(key, 0) + 1
    for key, count in identity_counts.items():
        if count > 2:
            raise ValueError(f"Source identity {key} appears {count} times (max 2)")

    # Exactly 24 unique source identities
    if len(identity_counts) != expected_unique_count:
        raise ValueError(
            f"Expected {expected_unique_count} unique source identities, "
            f"got {len(identity_counts)}"
        )

    # Exactly 6 duplicated source identities
    duplicated_identities = {k for k, v in identity_counts.items() if v == 2}
    if len(duplicated_identities) != expected_duplicate_count:
        raise ValueError(
            f"Expected {expected_duplicate_count} duplicated identities, "
            f"got {len(duplicated_identities)}"
        )

    # All four methods represented
    methods = {item.source_trial.source_method for item in items}
    if methods != VALID_SOURCE_METHODS:
        raise ValueError(f"Missing methods: {VALID_SOURCE_METHODS - methods}")

    # No method contributes more than 2 duplicates
    method_dup_counts: dict[str, int] = {}
    for m, tid in plan.duplicated_source_identities:
        method_dup_counts[m] = method_dup_counts.get(m, 0) + 1
    for method, count in method_dup_counts.items():
        if count > 2:
            raise ValueError(f"Method {method} contributes {count} duplicates (max 2)")

    # At least 1 duplicate from every method
    methods_with_dups = set(method_dup_counts.keys())
    if methods_with_dups != VALID_SOURCE_METHODS:
        missing = VALID_SOURCE_METHODS - methods_with_dups
        raise ValueError(f"Methods without duplicates: {missing}")

    # Duplicate separation
    identity_positions: dict[tuple[str, str], list[int]] = {}
    for item in items:
        key = (item.source_trial.source_method, item.source_trial.original_trial_id)
        identity_positions.setdefault(key, []).append(item.appearance_index)

    for key, positions in identity_positions.items():
        if len(positions) == 2:
            distance = abs(positions[0] - positions[1])
            if distance < minimum_duplicate_separation:
                raise ValueError(
                    f"Duplicate pair {key} has distance {distance} "
                    f"(minimum {minimum_duplicate_separation})"
                )

    # Blinded IDs contain no method names or trial IDs
    for item in items:
        bid_lower = item.blinded_id.lower()
        if item.source_trial.source_method in bid_lower:
            raise ValueError(
                f"Blinded ID {item.blinded_id} contains method name"
            )
        if item.source_trial.original_trial_id in bid_lower:
            # Only flag if the trial ID is long enough to be meaningful
            if len(item.source_trial.original_trial_id) > 2:
                raise ValueError(
                    f"Blinded ID {item.blinded_id} contains trial ID"
                )
