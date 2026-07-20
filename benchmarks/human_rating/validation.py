"""Validation helpers for the human-rating workflow file formats.

Validates:
- Rating queue JSON structure
- Answer key JSON structure and referential integrity
- Rating session JSONL records
- Evaluation manifest
"""

import json
from typing import Any

from benchmarks.human_rating.schemas import (
    VALID_RATINGS,
    VALID_SOURCE_METHODS,
    EVALUATION_DIMENSION,
    UNIQUE_ITEMS_PER_METHOD,
    DUPLICATE_COUNT,
    TOTAL_RATING_ITEMS,
)


# ---------------------------------------------------------------------------
# Rating queue validation
# ---------------------------------------------------------------------------

_QUEUE_REQUIRED_TOP_KEYS = {"schema_version", "run_id", "item_count", "items"}
_QUEUE_ITEM_REQUIRED_KEYS = {"blinded_id", "reference_context", "question", "response"}
_QUEUE_FORBIDDEN_KEYS = {
    "source_method", "method", "model", "original_trial_id",
    "judge_score", "judge_score_dimension",
    "profile_usage_score", "task_usage_score", "integration_score",
    "duplicate_of", "duplicate_status", "is_duplicate",
    "question_type", "evaluation_dimension",
    "sampling_seed", "shuffle_seed", "source_path", "source_file",
    "inference_context", "inference_context_text",
}


def validate_rating_queue(data: dict[str, Any]) -> list[str]:
    """Validate a rating queue JSON structure."""
    errors: list[str] = []

    missing_top = _QUEUE_REQUIRED_TOP_KEYS - set(data.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        return errors

    if data["schema_version"] != 1:
        errors.append(f"Unsupported schema_version: {data['schema_version']}")

    items = data.get("items", [])
    if not isinstance(items, list):
        errors.append("'items' must be a list")
        return errors

    if data.get("item_count") != len(items):
        errors.append(f"item_count ({data.get('item_count')}) != actual count ({len(items)})")

    seen_ids: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dict")
            continue

        missing_item = _QUEUE_ITEM_REQUIRED_KEYS - set(item.keys())
        if missing_item:
            errors.append(f"Item {i}: missing keys {sorted(missing_item)}")

        leaked = _QUEUE_FORBIDDEN_KEYS & set(item.keys())
        if leaked:
            errors.append(f"Item {i}: contains forbidden keys {sorted(leaked)}")

        bid = item.get("blinded_id", "")
        if bid in seen_ids:
            errors.append(f"Item {i}: duplicate blinded_id '{bid}'")
        seen_ids.add(bid)

        if not item.get("question", "").strip():
            errors.append(f"Item {i} ('{bid}'): empty question")
        if not item.get("response", "").strip():
            errors.append(f"Item {i} ('{bid}'): empty response")

    return errors


# ---------------------------------------------------------------------------
# Answer key validation
# ---------------------------------------------------------------------------

_KEY_REQUIRED_TOP_KEYS = {
    "schema_version", "benchmark_name", "sampling_seed",
    "duplicate_seed", "id_assignment_seed", "presentation_seed",
    "created_at", "items",
}
_KEY_ITEM_REQUIRED_KEYS = {
    "blinded_id", "source_method", "model", "original_trial_id",
    "judge_score", "judge_score_dimension",
    "profile_usage_score", "task_usage_score", "integration_score",
    "duplicate_of",
}


def validate_answer_key(data: dict[str, Any], expected_count: int = 30) -> list[str]:
    """Validate an answer key JSON structure."""
    errors: list[str] = []

    missing_top = _KEY_REQUIRED_TOP_KEYS - set(data.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        return errors

    if data["schema_version"] != 1:
        errors.append(f"Unsupported schema_version: {data['schema_version']}")

    items = data.get("items", [])
    if not isinstance(items, list):
        errors.append("'items' must be a list")
        return errors

    if len(items) != expected_count:
        errors.append(f"Expected {expected_count} items, got {len(items)}")

    seen_ids: set[str] = set()
    duplicates: list[dict] = []
    id_to_item: dict[str, dict] = {}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dict")
            continue

        missing_item = _KEY_ITEM_REQUIRED_KEYS - set(item.keys())
        if missing_item:
            errors.append(f"Item {i}: missing keys {sorted(missing_item)}")
            continue

        bid = item["blinded_id"]
        if bid in seen_ids:
            errors.append(f"Item {i}: duplicate blinded_id '{bid}'")
        seen_ids.add(bid)
        id_to_item[bid] = item

        if item["source_method"] not in VALID_SOURCE_METHODS:
            errors.append(f"Item {i} ('{bid}'): invalid source_method '{item['source_method']}'")

        if item["judge_score_dimension"] != EVALUATION_DIMENSION:
            errors.append(f"Item {i} ('{bid}'): judge_score_dimension must be '{EVALUATION_DIMENSION}'")

        for sf in ("judge_score", "profile_usage_score", "task_usage_score", "integration_score"):
            sv = item.get(sf)
            if not isinstance(sv, int) or sv < 1 or sv > 5:
                errors.append(f"Item {i} ('{bid}'): invalid {sf} {sv!r}")

        if item.get("judge_score") != item.get("integration_score"):
            errors.append(f"Item {i} ('{bid}'): judge_score must equal integration_score")

        if item["duplicate_of"] is not None:
            duplicates.append(item)

    if len(duplicates) != DUPLICATE_COUNT:
        errors.append(f"Expected {DUPLICATE_COUNT} duplicates, got {len(duplicates)}")

    for dup in duplicates:
        target = dup["duplicate_of"]
        bid = dup["blinded_id"]
        if target not in id_to_item:
            errors.append(f"Duplicate '{bid}': target '{target}' does not exist")
            continue
        if target == bid:
            errors.append(f"Duplicate '{bid}': points to itself")
            continue
        original = id_to_item[target]
        for field in ("source_method", "model", "original_trial_id"):
            if dup[field] != original[field]:
                errors.append(f"Duplicate '{bid}': {field} mismatch with '{target}'")

    return errors


# ---------------------------------------------------------------------------
# Ratings session validation
# ---------------------------------------------------------------------------

_RATING_REQUIRED_KEYS = {"blinded_id", "rating", "note", "flagged", "rated_at"}
_RATING_FORBIDDEN_KEYS = {
    "source_method", "method", "model", "original_trial_id",
    "judge_score", "judge_score_dimension",
    "profile_usage_score", "task_usage_score", "integration_score",
    "duplicate_of", "question_type", "evaluation_dimension",
}


def validate_rating_record(record: dict[str, Any]) -> list[str]:
    """Validate a single rating record."""
    errors: list[str] = []

    missing = _RATING_REQUIRED_KEYS - set(record.keys())
    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")
        return errors

    leaked = _RATING_FORBIDDEN_KEYS & set(record.keys())
    if leaked:
        errors.append(f"Contains forbidden keys: {sorted(leaked)}")

    rating = record["rating"]
    if not isinstance(rating, int) or rating not in VALID_RATINGS:
        errors.append(f"Invalid rating: {rating!r} (must be int 1–5)")

    if not record["blinded_id"] or not str(record["blinded_id"]).strip():
        errors.append("blinded_id must be nonempty")

    if not record["rated_at"] or not str(record["rated_at"]).strip():
        errors.append("rated_at must be nonempty")

    if not isinstance(record["flagged"], bool):
        errors.append(f"flagged must be bool, got {type(record['flagged']).__name__}")

    return errors


def validate_ratings_session(lines: list[dict[str, Any]]) -> list[str]:
    """Validate a complete ratings session."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, record in enumerate(lines):
        for err in validate_rating_record(record):
            errors.append(f"Record {i}: {err}")
        bid = record.get("blinded_id", "")
        if bid in seen_ids:
            errors.append(f"Record {i}: duplicate blinded_id '{bid}' (ratings are immutable)")
        seen_ids.add(bid)

    return errors


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

def validate_manifest(data: dict[str, Any]) -> list[str]:
    """Validate the evaluation manifest."""
    errors: list[str] = []

    if data.get("schema_version") != 1:
        errors.append(f"Unsupported schema_version: {data.get('schema_version')}")

    protocol = data.get("protocol", {})
    if not protocol:
        errors.append("Missing 'protocol' section")
        return errors

    if protocol.get("assistant_model") != "gpt-4o":
        errors.append(f"assistant_model must be 'gpt-4o', got {protocol.get('assistant_model')!r}")

    if protocol.get("judge_score_dimension") != "integration":
        errors.append(f"judge_score_dimension must be 'integration', got {protocol.get('judge_score_dimension')!r}")

    if protocol.get("unique_items_per_method") != UNIQUE_ITEMS_PER_METHOD:
        errors.append(f"unique_items_per_method must be {UNIQUE_ITEMS_PER_METHOD}")

    if protocol.get("duplicate_count") != DUPLICATE_COUNT:
        errors.append(f"duplicate_count must be {DUPLICATE_COUNT}")

    expected_total = UNIQUE_ITEMS_PER_METHOD * len(VALID_SOURCE_METHODS) + DUPLICATE_COUNT
    if protocol.get("total_rating_items") != expected_total:
        errors.append(f"total_rating_items must be {expected_total}")

    if "minimum_duplicate_separation" in protocol:
        sep = protocol["minimum_duplicate_separation"]
        if not isinstance(sep, int) or sep < 1:
            errors.append(f"minimum_duplicate_separation must be a positive integer")

    if protocol.get("question_stratification") is not None:
        strat = protocol["question_stratification"]
        if strat in ("profile", "task", "profile/task"):
            errors.append(f"question_stratification cannot claim profile/task: {strat!r}")

    if protocol.get("is_exact_model_visible_context") is True:
        errors.append("is_exact_model_visible_context must be false")

    sources = data.get("sources", {})
    if not sources:
        errors.append("Missing 'sources' section")
    else:
        methods_in_sources = set(sources.keys())
        if methods_in_sources != VALID_SOURCE_METHODS:
            missing = VALID_SOURCE_METHODS - methods_in_sources
            extra = methods_in_sources - VALID_SOURCE_METHODS
            if missing:
                errors.append(f"Missing methods in sources: {sorted(missing)}")
            if extra:
                errors.append(f"Unknown methods in sources: {sorted(extra)}")

        for method, source_info in sources.items():
            if not source_info.get("path"):
                errors.append(f"Source '{method}' missing 'path'")

    return errors
