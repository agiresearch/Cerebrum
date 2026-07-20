"""Phase 2: Interactive CLI for rating blinded items.

Usage:
    python -m benchmarks.human_rating.rate \
        --queue benchmarks/human_rating/runs/human-rating-v1/rater/rating_queue.json

This command:
1. Loads the blinded rating queue.
2. Presents items one at a time with the scoring rubric.
3. Records ratings to an append-only JSONL session file.
4. Supports resumption (skips already-rated items).

The rating command:
- Receives ONLY the queue path (ratings path is derived).
- Does NOT accept an answer-key argument.
- Does NOT import compilation code or the answer key.
- Does NOT scan parent directories.
- Does NOT print the private path.
- Operates correctly when the private directory is unavailable.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.human_rating.rubric import RATING_RUBRIC
from benchmarks.human_rating.validation import validate_rating_queue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NOTE_LENGTH = 2000
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Fields that must never appear in the queue
_FORBIDDEN_FIELDS = {
    "source_method", "method", "model", "original_trial_id", "trial_index",
    "judge_score", "profile_usage_score", "task_usage_score",
    "integration_score", "evaluation_dimension",
    "duplicate_of", "duplicate_of_blinded_id", "appearance_index",
    "queue_position", "sampling_seed", "blinding_seed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_display(text: str) -> str:
    """Remove ANSI escape sequences from text for safe terminal display."""
    return _ANSI_ESCAPE.sub("", text)


def _queue_fingerprint(queue_data: dict) -> str:
    """Compute a deterministic SHA-256 fingerprint of the queue."""
    canonical = json.dumps(queue_data, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Lock file management
# ---------------------------------------------------------------------------

class _LockFile:
    """Simple exclusive file lock for single-machine concurrency control."""

    def __init__(self, path: str):
        self.path = path
        self._fd = None

    def acquire(self) -> None:
        self._fd = open(self.path, "w")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            self._fd.close()
            self._fd = None
            raise RuntimeError(
                "Another rating session appears to be using this ratings file."
            )
        # Write PID for diagnostics
        self._fd.write(str(os.getpid()))
        self._fd.flush()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def release(self) -> None:
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
            except OSError:
                pass
            try:
                os.unlink(self.path)
            except OSError:
                pass
            self._fd = None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _load_or_create_session(
    session_path: Path,
    run_id: str,
    rater_id: str,
    queue_fp: str,
    item_count: int,
    ratings_filename: str = "ratings.jsonl",
) -> dict:
    """Load existing session metadata or create a new one."""
    if session_path.exists():
        with open(session_path, "r") as f:
            session = json.load(f)

        # Validate required fields present
        required_fields = {"schema_version", "run_id", "rater_id",
                           "queue_fingerprint", "queue_item_count", "ratings_file"}
        missing = required_fields - set(session.keys())
        if missing:
            raise ValueError(
                f"Invalid rating session metadata: missing fields {sorted(missing)}"
            )

        # Validate values
        if session.get("queue_fingerprint") != queue_fp:
            raise ValueError(
                "The queue does not match the existing rating session. "
                "Refusing to resume against modified items."
            )
        if session.get("rater_id") != rater_id:
            raise ValueError(
                f"Rater ID mismatch: session has '{session.get('rater_id')}', "
                f"but you specified '{rater_id}'."
            )
        if session.get("run_id") != run_id:
            raise ValueError(
                f"Run ID mismatch: session has '{session.get('run_id')}', "
                f"queue has '{run_id}'."
            )
        if session.get("queue_item_count") != item_count:
            raise ValueError(
                f"Invalid rating session metadata: queue item count does not "
                f"match the current queue ({session.get('queue_item_count')} vs {item_count})."
            )
        if session.get("ratings_file") != ratings_filename:
            raise ValueError(
                f"Invalid rating session metadata: ratings file mismatch; "
                f"expected {ratings_filename!r}, found {session.get('ratings_file')!r}."
            )
        return session

    session = {
        "schema_version": 1,
        "run_id": run_id,
        "rater_id": rater_id,
        "queue_fingerprint": queue_fp,
        "queue_item_count": item_count,
        "started_at": _now_iso(),
        "ratings_file": ratings_filename,
    }
    session_path.parent.mkdir(parents=True, exist_ok=True)
    with open(session_path, "w") as f:
        json.dump(session, f, indent=2)
    try:
        session_path.chmod(0o600)
    except OSError:
        pass
    return session


def _load_existing_ratings(
    ratings_path: Path,
    queue_ids: set[str],
    run_id: str,
    rater_id: str,
) -> set[str]:
    """Load and validate existing JSONL ratings. Returns set of completed IDs."""
    completed: set[str] = set()
    if not ratings_path.exists():
        return completed

    with open(ratings_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Malformed JSON at line {line_num} in {ratings_path}"
                )

            if not isinstance(record, dict):
                raise ValueError(
                    f"Malformed ratings record on line {line_num}: expected object"
                )

            bid = record.get("blinded_id", "")
            if not bid:
                raise ValueError(f"Empty blinded_id at line {line_num}")
            if bid not in queue_ids:
                raise ValueError(
                    f"Unknown blinded_id '{bid}' at line {line_num} "
                    f"(not in queue)"
                )
            if bid in completed:
                raise ValueError(
                    f"Duplicate blinded_id '{bid}' at line {line_num} "
                    f"(ratings are immutable)"
                )
            if record.get("run_id") != run_id:
                raise ValueError(
                    f"Run ID mismatch at line {line_num}: "
                    f"'{record.get('run_id')}' != '{run_id}'"
                )
            if record.get("rater_id") != rater_id:
                raise ValueError(
                    f"Rater ID mismatch at line {line_num}: "
                    f"'{record.get('rater_id')}' != '{rater_id}'"
                )
            rating = record.get("rating")
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                raise ValueError(
                    f"Invalid rating at line {line_num}: {rating!r}"
                )
            # Check for forbidden keys
            for key in record:
                if key in _FORBIDDEN_FIELDS:
                    raise ValueError(
                        f"Forbidden key '{key}' at line {line_num}"
                    )

            completed.add(bid)

    return completed


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _display_item(item: dict, position: int, total: int) -> None:
    """Display a single queue item for rating."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f" Item {position} of {total}")
    print(f" Blinded ID: {item['blinded_id']}")
    print(sep)

    ctx = _sanitize_display(item.get("reference_context", ""))
    if ctx:
        print(f"\nREFERENCE USER AND TASK CONTEXT\n{ctx}")

    question = _sanitize_display(item.get("question", ""))
    print(f"\nQUESTION\n{question}")

    response = _sanitize_display(item.get("response", ""))
    print(f"\nRESPONSE\n{response}")

    print(f"\n{'-' * 60}")
    print("Rate this response using the 1–5 rubric.")
    print("Enter h for rubric help, n for note, f to flag, or q to quit.")


def _display_rubric() -> None:
    """Display the full rating rubric."""
    print("\n--- RATING RUBRIC ---")
    for score in sorted(RATING_RUBRIC.keys(), reverse=True):
        print(f"  {score}: {RATING_RUBRIC[score]}")
    print("---")


# ---------------------------------------------------------------------------
# Core rating loop
# ---------------------------------------------------------------------------

def _run_rating_session(
    queue_data: dict,
    ratings_path: Path,
    session_path: Path,
    run_id: str,
    rater_id: str,
    *,
    input_fn=None,
    output_fn=None,
) -> None:
    """Run the interactive rating loop.

    Args:
        input_fn: Override for input() (for testing).
        output_fn: Override for print() (for testing).
    """
    if input_fn is None:
        input_fn = input
    if output_fn is None:
        output_fn = print

    items = queue_data["items"]
    total = len(items)
    queue_ids = {it["blinded_id"] for it in items}

    # Load existing ratings
    completed = _load_existing_ratings(ratings_path, queue_ids, run_id, rater_id)

    if len(completed) >= total:
        output_fn(f"This rating session is already complete: {total} of {total} items rated.")
        return

    output_fn(f"Loaded {len(completed)} completed ratings. "
              f"Resuming at item {len(completed) + 1} of {total}.")

    # Per-item state
    current_note: str | None = None
    current_flagged: bool = False

    for i, item in enumerate(items):
        bid = item["blinded_id"]
        if bid in completed:
            continue

        position = i + 1
        current_note = None
        current_flagged = False
        _display_item(item, position, total)

        while True:
            try:
                raw = input_fn("Rating: ")
            except (EOFError, KeyboardInterrupt):
                output_fn(f"\n\nSession saved. Completed: {len(completed)} of {total}")
                output_fn(f"Remaining: {total - len(completed)}")
                output_fn("Resume with the same queue, ratings path, and rater ID.")
                return

            if raw is None:
                # EOF from test input
                output_fn(f"\nSession saved. Completed: {len(completed)} of {total}")
                return

            cmd = raw.strip().lower()

            if cmd in ("q", "quit", "exit"):
                output_fn(f"\nSession saved. Completed: {len(completed)} of {total}")
                output_fn(f"Remaining: {total - len(completed)}")
                output_fn("Resume with the same queue, ratings path, and rater ID.")
                return

            if cmd in ("h", "help", "?"):
                _display_rubric()
                continue

            if cmd == "n":
                try:
                    note_input = input_fn("Enter note, or leave blank to clear: ")
                except (EOFError, KeyboardInterrupt):
                    output_fn(f"\nSession saved. Completed: {len(completed)} of {total}")
                    return
                if note_input is None:
                    output_fn(f"\nSession saved. Completed: {len(completed)} of {total}")
                    return
                note_text = note_input.strip()
                if not note_text:
                    current_note = None
                    output_fn("Note cleared for this item.")
                elif len(note_text) > MAX_NOTE_LENGTH:
                    output_fn(f"Note too long ({len(note_text)} chars, max {MAX_NOTE_LENGTH}).")
                else:
                    current_note = note_text
                    output_fn("Note saved for this item.")
                continue

            if cmd == "f":
                current_flagged = not current_flagged
                state = "ON" if current_flagged else "OFF"
                output_fn(f"Item flag: {state}")
                continue

            # Try to parse as rating
            try:
                rating = int(cmd)
            except ValueError:
                output_fn("Invalid rating. Enter an integer from 1 through 5.")
                continue

            if rating < 1 or rating > 5:
                output_fn("Invalid rating. Enter an integer from 1 through 5.")
                continue

            # Valid rating — save immediately
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "rater_id": rater_id,
                "blinded_id": bid,
                "rating": rating,
                "note": current_note,
                "flagged": current_flagged,
                "rated_at": _now_iso(),
            }

            with open(ratings_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            # Set permissions on first write
            try:
                ratings_path.chmod(0o600)
            except OSError:
                pass

            completed.add(bid)
            output_fn(f"Saved rating {rating} for {bid}. Progress: {len(completed)} of {total}.")
            break

    # All done
    flagged_count = 0
    note_count = 0
    if ratings_path.exists():
        with open(ratings_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("flagged"):
                    flagged_count += 1
                if r.get("note"):
                    note_count += 1

    output_fn("\nRating session complete.")
    output_fn(f"  Completed items: {total}")
    output_fn(f"  Flagged items: {flagged_count}")
    output_fn(f"  Notes recorded: {note_count}")
    output_fn(f"  Ratings file: {ratings_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for rate.

    Note: No --answer-key argument exists by design.
    """
    parser = argparse.ArgumentParser(
        description="Rate blinded evaluation items on a 1–5 personalization "
        "scale. Ratings are append-only and cannot be edited.",
    )
    parser.add_argument(
        "--queue",
        type=str,
        required=True,
        help="Path to the blinded rating_queue.json file.",
    )
    parser.add_argument(
        "--ratings",
        type=str,
        default=None,
        help="Path to ratings JSONL file (default: ratings.jsonl in queue dir).",
    )
    parser.add_argument(
        "--rater-id",
        type=str,
        default="rater-01",
        help="Rater identifier for this session (default: rater-01).",
    )
    return parser


def main() -> None:
    """Entry point for rate CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Error: Queue file not found: {queue_path}", file=sys.stderr)
        sys.exit(1)

    # Load and validate queue
    try:
        with open(queue_path, "r", encoding="utf-8") as f:
            queue_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in queue file: {e}", file=sys.stderr)
        sys.exit(1)

    errors = validate_rating_queue(queue_data)
    if errors:
        print("Error: Queue validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)

    # Enforce exactly 30 items for production use
    actual_count = len(queue_data.get("items", []))
    if actual_count != 30:
        print(
            f"Invalid rating queue: expected exactly 30 items, found {actual_count}.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_id = queue_data.get("run_id", "unknown")
    rater_id = args.rater_id

    # Derive paths
    rater_dir = queue_path.parent
    ratings_path = Path(args.ratings) if args.ratings else rater_dir / "ratings.jsonl"
    session_path = rater_dir / "rating_session.json"
    lock_path = str(ratings_path) + ".lock"

    # Compute queue fingerprint
    queue_fp = _queue_fingerprint(queue_data)

    # Session management
    try:
        _load_or_create_session(
            session_path, run_id, rater_id, queue_fp, len(queue_data["items"]),
            ratings_filename=ratings_path.name,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Acquire lock
    lock = _LockFile(lock_path)
    try:
        lock.acquire()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set permissions on ratings file if it exists
    if ratings_path.exists():
        try:
            ratings_path.chmod(0o600)
        except OSError:
            pass

    try:
        _run_rating_session(queue_data, ratings_path, session_path, run_id, rater_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
