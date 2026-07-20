"""Phase 3: Compile human ratings with the private answer key.

Joins blinded ratings with the answer key, computes human-versus-judge
agreement metrics, method-level summaries, and duplicate consistency.

Usage:
    python -m benchmarks.human_rating.compile_ratings \
        --queue .../rater/rating_queue.json \
        --ratings .../rater/ratings.jsonl \
        --session .../rater/rating_session.json \
        --answer-key .../private/answer_key.json \
        --output-dir .../private/compiled
"""

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from benchmarks.human_rating.rate import _queue_fingerprint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledItem:
    run_id: str
    queue_position: int
    blinded_id: str
    source_method: str
    model: str
    original_trial_id: str
    human_rating: int
    judge_score: int
    score_difference: int
    absolute_difference: int
    exact_agreement: bool
    within_one_agreement: bool
    note: str | None
    flagged: bool
    rated_at: str
    is_duplicate_appearance: bool
    duplicate_of_blinded_id: str | None
    profile_usage_score: int
    task_usage_score: int
    integration_score: int
    evaluation_dimension: str
    reference_context_provenance: str
    is_exact_model_visible_context: bool


@dataclass(frozen=True)
class AgreementSummary:
    item_count: int
    mean_human_rating: float
    mean_judge_score: float
    mean_signed_difference: float
    mean_absolute_error: float
    exact_agreement_count: int
    exact_agreement_rate: float
    within_one_count: int
    within_one_rate: float


@dataclass(frozen=True)
class MethodAgreementSummary:
    method: str
    item_count: int
    mean_human_rating: float
    median_human_rating: float
    human_rating_std: float
    mean_judge_score: float
    mean_signed_difference: float
    mean_absolute_error: float
    exact_agreement_rate: float
    within_one_rate: float
    flagged_count: int
    noted_count: int


@dataclass(frozen=True)
class DuplicateConsistencyRecord:
    source_method: str
    original_trial_id: str
    original_blinded_id: str
    duplicate_blinded_id: str
    original_queue_position: int
    duplicate_queue_position: int
    positional_distance: int
    original_rating: int
    duplicate_rating: int
    signed_difference: int
    absolute_difference: int
    exact_match: bool
    within_one: bool


@dataclass(frozen=True)
class DuplicateConsistencySummary:
    pair_count: int
    exact_match_count: int
    exact_match_rate: float
    within_one_count: int
    within_one_rate: float
    mean_absolute_difference: float
    max_absolute_difference: int
    mean_positional_distance: float


@dataclass(frozen=True)
class CompilationResult:
    primary_items: tuple[CompiledItem, ...]
    all_appearances: tuple[CompiledItem, ...]
    duplicate_pairs: tuple[DuplicateConsistencyRecord, ...]
    overall_summary: AgreementSummary
    method_summaries: tuple[MethodAgreementSummary, ...]
    appearance_sensitivity_summary: AgreementSummary
    appearance_method_summaries: tuple[MethodAgreementSummary, ...]
    duplicate_summary: DuplicateConsistencySummary
    confusion_matrix: dict[int, dict[int, int]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_agreement(items: tuple[CompiledItem, ...] | list) -> AgreementSummary:
    n = len(items)
    if n == 0:
        return AgreementSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0.0)
    hr = [i.human_rating for i in items]
    js = [i.judge_score for i in items]
    sd = [i.score_difference for i in items]
    ad = [i.absolute_difference for i in items]
    exact = sum(1 for i in items if i.exact_agreement)
    w1 = sum(1 for i in items if i.within_one_agreement)
    return AgreementSummary(
        item_count=n,
        mean_human_rating=statistics.mean(hr),
        mean_judge_score=statistics.mean(js),
        mean_signed_difference=statistics.mean(sd),
        mean_absolute_error=statistics.mean(ad),
        exact_agreement_count=exact,
        exact_agreement_rate=exact / n,
        within_one_count=w1,
        within_one_rate=w1 / n,
    )


def _compute_method_summary(method: str, items: list[CompiledItem]) -> MethodAgreementSummary:
    n = len(items)
    hr = [i.human_rating for i in items]
    js = [i.judge_score for i in items]
    sd = [i.score_difference for i in items]
    ad = [i.absolute_difference for i in items]
    exact = sum(1 for i in items if i.exact_agreement)
    w1 = sum(1 for i in items if i.within_one_agreement)
    return MethodAgreementSummary(
        method=method, item_count=n,
        mean_human_rating=statistics.mean(hr),
        median_human_rating=float(statistics.median(hr)),
        human_rating_std=statistics.stdev(hr) if n > 1 else 0.0,
        mean_judge_score=statistics.mean(js),
        mean_signed_difference=statistics.mean(sd),
        mean_absolute_error=statistics.mean(ad),
        exact_agreement_rate=exact / n,
        within_one_rate=w1 / n,
        flagged_count=sum(1 for i in items if i.flagged),
        noted_count=sum(1 for i in items if i.note),
    )


def _build_confusion_matrix(items: list[CompiledItem]) -> dict[int, dict[int, int]]:
    matrix: dict[int, dict[int, int]] = {h: {j: 0 for j in range(1, 6)} for h in range(1, 6)}
    for item in items:
        matrix[item.human_rating][item.judge_score] += 1
    return matrix


# ---------------------------------------------------------------------------
# Core compilation
# ---------------------------------------------------------------------------

def compile_rating_run(
    *,
    queue_path: Path,
    ratings_path: Path,
    session_path: Path,
    answer_key_path: Path,
) -> CompilationResult:
    """Compile a complete rating run into analysis results."""
    # Load all files
    with open(queue_path) as f:
        queue = json.load(f)
    with open(answer_key_path) as f:
        answer_key = json.load(f)
    with open(session_path) as f:
        session = json.load(f)

    ratings: list[dict] = []
    with open(ratings_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Malformed JSONL at line {line_num}")
            if not isinstance(rec, dict):
                raise ValueError(f"Non-object JSONL record at line {line_num}")
            ratings.append(rec)

    # --- Validate queue ---
    q_items = queue.get("items", [])
    if len(q_items) != 30:
        raise ValueError(f"Queue must have 30 items, found {len(q_items)}")
    q_ids = [it["blinded_id"] for it in q_items]
    if len(set(q_ids)) != 30:
        raise ValueError("Queue has duplicate blinded IDs")

    # --- Validate ratings count and content ---
    if len(ratings) > 30:
        raise ValueError(f"Extra ratings: found {len(ratings)}, expected 30.")
    if len(ratings) != 30:
        raise ValueError(
            f"Cannot compile an incomplete rating session: "
            f"{len(ratings)} of 30 queue items have ratings."
        )

    # Validate rating order matches queue order
    r_ids = [r["blinded_id"] for r in ratings]
    if r_ids != q_ids:
        # Check for duplicates first
        if len(set(r_ids)) != 30:
            seen = set()
            for i, rid in enumerate(r_ids):
                if rid in seen:
                    raise ValueError(f"Duplicate rating ID at position {i+1}: {rid}")
                seen.add(rid)
        # Check for unknown IDs
        unknown = set(r_ids) - set(q_ids)
        if unknown:
            raise ValueError(f"Unknown rating ID(s): {unknown}")
        # Otherwise order mismatch
        raise ValueError("Rating order does not match queue order")

    run_id = queue.get("run_id", "")
    rater_id = session.get("rater_id", "")

    for i, r in enumerate(ratings):
        if r.get("run_id") != run_id:
            raise ValueError(
                f"Run ID mismatch in rating at position {i+1}: "
                f"'{r.get('run_id')}' != '{run_id}'"
            )
        if r.get("rater_id") != rater_id:
            raise ValueError(
                f"Rater ID mismatch in rating at position {i+1}: "
                f"'{r.get('rater_id')}' != '{rater_id}'"
            )
        rating_val = r.get("rating")
        if not isinstance(rating_val, int) or rating_val < 1 or rating_val > 5:
            raise ValueError(f"Invalid rating value at position {i+1}: {rating_val}")

    # --- Validate session ---
    sess_required = {"schema_version", "run_id", "rater_id", "queue_fingerprint",
                     "queue_item_count", "ratings_file"}
    missing_sess = sess_required - set(session.keys())
    if missing_sess:
        raise ValueError(f"Session missing fields: {sorted(missing_sess)}")
    queue_fp = _queue_fingerprint(queue)
    if session.get("queue_fingerprint") != queue_fp:
        raise ValueError("Queue fingerprint mismatch with session")
    if session.get("queue_item_count") != 30:
        raise ValueError(
            f"Session item count mismatch: {session.get('queue_item_count')} != 30"
        )
    if session.get("ratings_file") != ratings_path.name:
        raise ValueError(
            f"Session ratings file mismatch: "
            f"'{session.get('ratings_file')}' != '{ratings_path.name}'"
        )

    # --- Validate answer key ---
    k_items = answer_key.get("items", [])
    if len(k_items) != 30:
        raise ValueError(f"Answer key must have 30 items, found {len(k_items)}")
    k_ids = [it["blinded_id"] for it in k_items]
    if k_ids != q_ids:
        raise ValueError("Answer key blinded-ID order does not match queue")
    if answer_key.get("run_id") != run_id:
        raise ValueError("Answer key run_id mismatch")

    # Validate positions
    for i, k in enumerate(k_items):
        if k.get("queue_position") != i + 1:
            raise ValueError(
                f"Answer key position mismatch at index {i}: "
                f"expected {i+1}, got {k.get('queue_position')}"
            )

    originals = [it for it in k_items if it.get("duplicate_of_blinded_id") is None]
    duplicates = [it for it in k_items if it.get("duplicate_of_blinded_id") is not None]
    if len(originals) != 24:
        raise ValueError(f"Expected 24 originals in key, found {len(originals)}")
    if len(duplicates) != 6:
        raise ValueError(f"Expected 6 duplicates in key, found {len(duplicates)}")

    # Validate duplicate references
    orig_ids_set = {it["blinded_id"] for it in originals}
    dup_ids_set = {it["blinded_id"] for it in duplicates}
    key_by_id = {it["blinded_id"]: it for it in k_items}

    for dup in duplicates:
        ref = dup["duplicate_of_blinded_id"]
        if ref not in orig_ids_set:
            if ref in dup_ids_set:
                raise ValueError(
                    f"Duplicate {dup['blinded_id']} points to another duplicate: {ref}"
                )
            raise ValueError(f"Broken duplicate reference: {dup['blinded_id']} -> {ref}")
        # Validate source metadata matches
        orig = key_by_id[ref]
        for field in ("source_method", "original_trial_id", "model"):
            if dup.get(field) != orig.get(field):
                raise ValueError(
                    f"Duplicate {dup['blinded_id']} {field} mismatch with original {ref}"
                )

    # Validate answer-key metadata fields
    for k in k_items:
        if k.get("model") != "gpt-4o":
            raise ValueError(f"Wrong model in key item {k['blinded_id']}: {k.get('model')}")
        if k.get("evaluation_dimension") != "integration":
            raise ValueError(f"Wrong evaluation_dimension in {k['blinded_id']}")
        if k.get("judge_score") != k.get("integration_score"):
            raise ValueError(f"judge_score != integration_score in {k['blinded_id']}")
        if k.get("reference_context_provenance") != "synthetic_source_context":
            raise ValueError(f"Wrong provenance in {k['blinded_id']}")
        if k.get("is_exact_model_visible_context") is not False:
            raise ValueError(f"is_exact_model_visible_context must be false in {k['blinded_id']}")

    # Validate duplicate queue content matches original
    q_by_id = {it["blinded_id"]: it for it in q_items}
    for dup in duplicates:
        ref = dup["duplicate_of_blinded_id"]
        dup_q = q_by_id[dup["blinded_id"]]
        orig_q = q_by_id[ref]
        for field in ("reference_context", "question", "response"):
            if dup_q.get(field) != orig_q.get(field):
                raise ValueError(
                    f"Duplicate {dup['blinded_id']} queue {field} differs from original {ref}"
                )

    # --- Build ID lookups ---
    rating_by_id = {r["blinded_id"]: r for r in ratings}

    # --- Compile items ---
    all_compiled: list[CompiledItem] = []
    for q_item in q_items:
        bid = q_item["blinded_id"]
        r = rating_by_id[bid]
        k = key_by_id[bid]

        human_rating = r["rating"]
        judge_score = k["judge_score"]
        diff = human_rating - judge_score

        all_compiled.append(CompiledItem(
            run_id=run_id,
            queue_position=k["queue_position"],
            blinded_id=bid,
            source_method=k["source_method"],
            model=k["model"],
            original_trial_id=k["original_trial_id"],
            human_rating=human_rating,
            judge_score=judge_score,
            score_difference=diff,
            absolute_difference=abs(diff),
            exact_agreement=(human_rating == judge_score),
            within_one_agreement=(abs(diff) <= 1),
            note=r.get("note"),
            flagged=r.get("flagged", False),
            rated_at=r.get("rated_at", ""),
            is_duplicate_appearance=(k.get("duplicate_of_blinded_id") is not None),
            duplicate_of_blinded_id=k.get("duplicate_of_blinded_id"),
            profile_usage_score=k["profile_usage_score"],
            task_usage_score=k["task_usage_score"],
            integration_score=k["integration_score"],
            evaluation_dimension=k["evaluation_dimension"],
            reference_context_provenance=k["reference_context_provenance"],
            is_exact_model_visible_context=k["is_exact_model_visible_context"],
        ))

    # --- Split populations ---
    primary = [i for i in all_compiled if not i.is_duplicate_appearance]
    assert len(primary) == 24

    # --- Primary overall agreement ---
    overall = _compute_agreement(primary)

    # --- Method summaries (primary only) ---
    methods = sorted(set(i.source_method for i in primary))
    method_sums = []
    for m in methods:
        m_items = [i for i in primary if i.source_method == m]
        if len(m_items) != 6:
            raise ValueError(f"Method {m} has {len(m_items)} primary items, expected 6")
        method_sums.append(_compute_method_summary(m, m_items))

    # --- Sensitivity (all 30) ---
    sensitivity = _compute_agreement(all_compiled)

    # --- Duplicate consistency ---
    dup_records: list[DuplicateConsistencyRecord] = []
    for dup_key in duplicates:
        dup_bid = dup_key["blinded_id"]
        orig_bid = dup_key["duplicate_of_blinded_id"]
        dup_item = next(i for i in all_compiled if i.blinded_id == dup_bid)
        orig_item = next(i for i in all_compiled if i.blinded_id == orig_bid)
        diff_r = dup_item.human_rating - orig_item.human_rating
        dup_records.append(DuplicateConsistencyRecord(
            source_method=dup_key["source_method"],
            original_trial_id=dup_key["original_trial_id"],
            original_blinded_id=orig_bid,
            duplicate_blinded_id=dup_bid,
            original_queue_position=orig_item.queue_position,
            duplicate_queue_position=dup_item.queue_position,
            positional_distance=abs(dup_item.queue_position - orig_item.queue_position),
            original_rating=orig_item.human_rating,
            duplicate_rating=dup_item.human_rating,
            signed_difference=diff_r,
            absolute_difference=abs(diff_r),
            exact_match=(orig_item.human_rating == dup_item.human_rating),
            within_one=(abs(diff_r) <= 1),
        ))

    exact_dup = sum(1 for d in dup_records if d.exact_match)
    w1_dup = sum(1 for d in dup_records if d.within_one)
    abs_diffs = [d.absolute_difference for d in dup_records]
    pos_dists = [d.positional_distance for d in dup_records]

    dup_summary = DuplicateConsistencySummary(
        pair_count=6,
        exact_match_count=exact_dup,
        exact_match_rate=exact_dup / 6,
        within_one_count=w1_dup,
        within_one_rate=w1_dup / 6,
        mean_absolute_difference=statistics.mean(abs_diffs),
        max_absolute_difference=max(abs_diffs),
        mean_positional_distance=statistics.mean(pos_dists),
    )

    # --- Confusion matrix (primary only) ---
    matrix = _build_confusion_matrix(primary)

    # --- Appearance-weighted method summaries (all 30) ---
    appearance_method_sums = []
    for m in methods:
        m_items = [i for i in all_compiled if i.source_method == m]
        appearance_method_sums.append(_compute_method_summary(m, m_items))

    return CompilationResult(
        primary_items=tuple(primary),
        all_appearances=tuple(all_compiled),
        duplicate_pairs=tuple(dup_records),
        overall_summary=overall,
        method_summaries=tuple(method_sums),
        appearance_sensitivity_summary=sensitivity,
        appearance_method_summaries=tuple(appearance_method_sums),
        duplicate_summary=dup_summary,
        confusion_matrix=matrix,
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_compilation_outputs(
    result: CompilationResult,
    *,
    output_dir: Path,
    run_id: str,
    rater_id: str,
    protocol_name: str,
    overwrite: bool = False,
) -> None:
    """Write all compilation outputs atomically via staging directory."""
    output_dir = Path(output_dir)

    if not overwrite and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory not empty: {output_dir}. Use --overwrite."
        )

    # Build summary JSON
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "rater_id": rater_id,
        "protocol_name": protocol_name,
        "primary_analysis": asdict(result.overall_summary),
        "primary_method_summaries": [asdict(ms) for ms in result.method_summaries],
        "appearance_sensitivity_analysis": asdict(result.appearance_sensitivity_summary),
        "appearance_method_summaries": [asdict(ms) for ms in result.appearance_method_summaries],
        "duplicate_consistency": asdict(result.duplicate_summary),
        "confusion_matrix": {str(h): row for h, row in result.confusion_matrix.items()},
        "limitations": {
            "question_stratification": None,
            "display_context_provenance": "synthetic_source_context",
            "is_exact_model_visible_context": False,
            "primary_judge_dimension": "integration",
        },
    }

    # Stage all outputs in a temporary directory, then publish atomically
    staging_dir = None
    try:
        staging_dir = Path(tempfile.mkdtemp(
            prefix=".compile_staging_", dir=str(output_dir.parent)
        ))

        _write_json(staging_dir / "summary.json", summary)
        _write_items_csv(staging_dir / "primary_items.csv", result.primary_items)
        _write_items_csv(staging_dir / "all_appearances.csv", result.all_appearances)
        _write_method_csv(staging_dir / "method_summary.csv", result.method_summaries)
        _write_dup_csv(staging_dir / "duplicate_consistency.csv", result.duplicate_pairs)
        _write_confusion_csv(staging_dir / "confusion_matrix.csv", result.confusion_matrix)

        # Set permissions on staged files
        for f in staging_dir.iterdir():
            try:
                f.chmod(0o600)
            except OSError:
                pass
        try:
            staging_dir.chmod(0o700)
        except OSError:
            pass

        # Publish: replace target directory
        if output_dir.exists() and overwrite:
            # Backup existing
            backup_dir = output_dir.with_name(output_dir.name + ".backup")
            if backup_dir.exists():
                import shutil
                shutil.rmtree(backup_dir)
            os.rename(str(output_dir), str(backup_dir))
            try:
                os.rename(str(staging_dir), str(output_dir))
                # Success — remove backup
                import shutil
                shutil.rmtree(backup_dir)
            except Exception:
                # Restore backup
                if not output_dir.exists() and backup_dir.exists():
                    os.rename(str(backup_dir), str(output_dir))
                raise
        else:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(staging_dir), str(output_dir))

        staging_dir = None  # Successfully published

    finally:
        # Clean up staging on failure
        if staging_dir and staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)


def _write_json(path: Path, data: dict) -> None:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent), suffix=".tmp",
            delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(data, tmp, indent=2, sort_keys=True, ensure_ascii=False)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, str(path))
    except Exception:
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def _write_items_csv(path: Path, items: tuple[CompiledItem, ...]) -> None:
    fields = [
        "queue_position", "blinded_id", "source_method", "original_trial_id",
        "human_rating", "judge_score", "score_difference", "absolute_difference",
        "exact_agreement", "within_one_agreement", "is_duplicate_appearance",
        "flagged", "note", "rated_at",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in items:
            row = {k: getattr(item, k) for k in fields}
            w.writerow(row)


def _write_method_csv(path: Path, summaries: tuple[MethodAgreementSummary, ...]) -> None:
    fields = [
        "method", "item_count", "mean_human_rating", "median_human_rating",
        "human_rating_std", "mean_judge_score", "mean_signed_difference",
        "mean_absolute_error", "exact_agreement_rate", "within_one_rate",
        "flagged_count", "noted_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(asdict(s))


def _write_dup_csv(path: Path, pairs: tuple[DuplicateConsistencyRecord, ...]) -> None:
    fields = [
        "source_method", "original_trial_id", "original_blinded_id",
        "duplicate_blinded_id", "original_queue_position", "duplicate_queue_position",
        "positional_distance", "original_rating", "duplicate_rating",
        "signed_difference", "absolute_difference", "exact_match", "within_one",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow(asdict(p))


def _write_confusion_csv(path: Path, matrix: dict[int, dict[int, int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["human\\judge", "1", "2", "3", "4", "5"])
        for h in range(1, 6):
            w.writerow([str(h)] + [str(matrix[h][j]) for j in range(1, 6)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile human ratings with the answer key.",
    )
    parser.add_argument("--queue", type=str, required=True)
    parser.add_argument("--ratings", type=str, required=True)
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--answer-key", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = compile_rating_run(
            queue_path=Path(args.queue),
            ratings_path=Path(args.ratings),
            session_path=Path(args.session),
            answer_key_path=Path(args.answer_key),
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Read run_id and rater_id from session
    with open(args.session) as f:
        session = json.load(f)
    with open(args.answer_key) as f:
        key = json.load(f)

    write_compilation_outputs(
        result,
        output_dir=Path(args.output_dir),
        run_id=session.get("run_id", ""),
        rater_id=session.get("rater_id", ""),
        protocol_name=key.get("protocol_name", ""),
        overwrite=args.overwrite,
    )

    s = result.overall_summary
    d = result.duplicate_summary
    print("Compilation complete.")
    print(f"  Primary items: {s.item_count}")
    print(f"  All appearances: {len(result.all_appearances)}")
    print(f"  Methods: {len(result.method_summaries)}")
    print(f"  Duplicate pairs: {d.pair_count}")
    print(f"  Exact agreement: {s.exact_agreement_rate:.1%}")
    print(f"  Within-one agreement: {s.within_one_rate:.1%}")
    print(f"  Duplicate exact consistency: {d.exact_match_rate:.1%}")
    print(f"  Duplicate within-one consistency: {d.within_one_rate:.1%}")


if __name__ == "__main__":
    main()
