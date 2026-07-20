"""Tests for the interactive rating CLI — session integrity and protocol enforcement.

All tests use 30-item queues. Output is captured (not printed to terminal).

Run:
    python benchmarks/human_rating/tests/test_rate.py
"""

import ast
import fcntl
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.rate import (
    _LockFile, _load_existing_ratings, _load_or_create_session,
    _queue_fingerprint, _run_rating_session, _sanitize_display,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_queue(run_id="test-run"):
    items = [
        {"blinded_id": f"HR-{i:04d}", "reference_context": f"Ctx {i}",
         "question": f"Q{i}", "response": f"R{i}"}
        for i in range(30)
    ]
    return {"schema_version": 1, "run_id": run_id, "item_count": 30, "items": items}


def _mock_input(responses):
    it = iter(responses)
    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            return None
    return _input


def _capture():
    lines = []
    def _print(*args, **kwargs):
        lines.append(" ".join(str(a) for a in args))
    return _print, lines


def _write_queue_file(tmpdir, queue_data):
    p = Path(tmpdir) / "rater" / "rating_queue.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(queue_data))
    return p


def _run_session(tmpdir, inputs, queue=None, run_id="test-run", rater_id="rater-01"):
    """Helper that runs a session and returns (records, output_lines, ratings_path)."""
    if queue is None:
        queue = _make_queue(run_id)
    ratings_path = Path(tmpdir) / "ratings.jsonl"
    session_path = Path(tmpdir) / "session.json"
    out_fn, lines = _capture()
    _run_rating_session(
        queue, ratings_path, session_path, run_id, rater_id,
        input_fn=_mock_input(inputs), output_fn=out_fn,
    )
    records = []
    if ratings_path.exists():
        for l in ratings_path.read_text().strip().split("\n"):
            if l.strip():
                records.append(json.loads(l))
    return records, lines, ratings_path


# ===========================================================================
# Quit/EOF (explicit)
# ===========================================================================

def test_quit_preserves_progress():
    with tempfile.TemporaryDirectory() as d:
        records, lines, rp = _run_session(d, ["4", "q"])
        assert len(records) == 1
        assert records[0]["blinded_id"] == "HR-0000"
        # Resume starts at item 2
        records2, _, _ = _run_session(d, ["5", "q"])
        assert len(records2) == 2
        assert records2[1]["blinded_id"] == "HR-0001"
        assert not any("Traceback" in l for l in lines)
    print("  PASS: test_quit_preserves_progress")


def test_eof_preserves_progress():
    with tempfile.TemporaryDirectory() as d:
        records, lines, _ = _run_session(d, ["3"])  # EOF after 1 rating
        assert len(records) == 1
        assert records[0]["blinded_id"] == "HR-0000"
        # Resume
        records2, _, _ = _run_session(d, ["2", "q"])
        assert len(records2) == 2
        assert records2[1]["blinded_id"] == "HR-0001"
        assert not any("Traceback" in l for l in lines)
    print("  PASS: test_eof_preserves_progress")


# ===========================================================================
# 2. Lock release on every exit path
# ===========================================================================

def _lock_path_for(tmpdir):
    return str(Path(tmpdir) / "ratings.jsonl.lock")


def _can_acquire_lock(lock_path):
    """Try acquiring the lock; returns True if successful."""
    try:
        fd = open(lock_path, "w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()
        return True
    except (OSError, IOError):
        return False


def test_lock_released_after_normal_completion():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        qp = _write_queue_file(d, queue)
        lock_path = str(Path(d) / "rater" / "ratings.jsonl.lock")
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", str(qp), "--ratings", str(Path(d) / "rater" / "ratings.jsonl")],
            input="\n".join(["3"] * 30) + "\n",
            capture_output=True, text=True, cwd=_project_root, timeout=30,
        )
        # Lock file should be gone or acquirable
        assert not Path(lock_path).exists() or _can_acquire_lock(lock_path)
    print("  PASS: test_lock_released_after_normal_completion")


def test_lock_released_after_quit():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        qp = _write_queue_file(d, queue)
        lock_path = str(Path(d) / "rater" / "ratings.jsonl.lock")
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", str(qp), "--ratings", str(Path(d) / "rater" / "ratings.jsonl")],
            input="4\nq\n", capture_output=True, text=True, cwd=_project_root, timeout=10,
        )
        assert not Path(lock_path).exists() or _can_acquire_lock(lock_path)
    print("  PASS: test_lock_released_after_quit")


def test_lock_released_after_startup_validation_error():
    with tempfile.TemporaryDirectory() as d:
        # Queue with wrong item count triggers validation failure
        q = _make_queue()
        q["items"] = q["items"][:5]
        q["item_count"] = 5
        qp = _write_queue_file(d, q)
        lock_path = str(Path(d) / "rater" / "ratings.jsonl.lock")
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", str(qp)],
            capture_output=True, text=True, cwd=_project_root, timeout=10,
        )
        # Lock never acquired or released
        assert not Path(lock_path).exists() or _can_acquire_lock(lock_path)
    print("  PASS: test_lock_released_after_startup_validation_error")


def test_lock_released_after_keyboard_interrupt():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        qp = _write_queue_file(d, queue)
        ratings_path = Path(d) / "rater" / "ratings.jsonl"
        lock_path = str(ratings_path) + ".lock"

        # Use subprocess with input that rates one item then sends nothing
        # (simulating interrupt via closed stdin after one rating)
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", str(qp), "--ratings", str(ratings_path)],
            input="4\n", capture_output=True, text=True, cwd=_project_root, timeout=10,
        )
        # Lock released
        assert not Path(lock_path).exists() or _can_acquire_lock(lock_path)
        # Rating preserved
        if ratings_path.exists():
            records = [json.loads(l) for l in ratings_path.read_text().strip().split("\n") if l.strip()]
            assert len(records) == 1
    print("  PASS: test_lock_released_after_keyboard_interrupt")


def test_lock_released_after_eof():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        qp = _write_queue_file(d, queue)
        ratings_path = Path(d) / "rater" / "ratings.jsonl"
        lock_path = str(ratings_path) + ".lock"

        # Send EOF immediately (empty stdin)
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", str(qp), "--ratings", str(ratings_path)],
            input="", capture_output=True, text=True, cwd=_project_root, timeout=10,
        )
        # Lock released
        assert not Path(lock_path).exists() or _can_acquire_lock(lock_path)
    print("  PASS: test_lock_released_after_eof")


# ===========================================================================
# 3. Rater-side permissions (POSIX)
# ===========================================================================

def test_rating_session_permissions_posix():
    if os.name != "posix":
        print("  SKIP: test_rating_session_permissions_posix")
        return
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        session_path = Path(d) / "session.json"
        fp = _queue_fingerprint(queue)
        _load_or_create_session(session_path, "test-run", "rater-01", fp, 30)
        assert stat.S_IMODE(os.stat(session_path).st_mode) == 0o600
    print("  PASS: test_rating_session_permissions_posix")


def test_ratings_file_permissions_posix():
    if os.name != "posix":
        print("  SKIP: test_ratings_file_permissions_posix")
        return
    with tempfile.TemporaryDirectory() as d:
        records, _, rp = _run_session(d, ["4", "q"])
        assert rp.exists()
        assert stat.S_IMODE(os.stat(rp).st_mode) == 0o600
    print("  PASS: test_ratings_file_permissions_posix")


def test_lock_file_permissions_posix():
    if os.name != "posix":
        print("  SKIP: test_lock_file_permissions_posix")
        return
    with tempfile.TemporaryDirectory() as d:
        lock_path = os.path.join(d, "test.lock")
        lock = _LockFile(lock_path)
        lock.acquire()
        assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o600
        lock.release()
    print("  PASS: test_lock_file_permissions_posix")


# ===========================================================================
# 4. Session-metadata consistency
# ===========================================================================

def test_session_item_count_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        session_path = Path(d) / "session.json"
        fp = _queue_fingerprint(queue)
        _load_or_create_session(session_path, "test-run", "rater-01", fp, 30)
        # Now try with different item_count
        try:
            _load_or_create_session(session_path, "test-run", "rater-01", fp, 25)
            assert False
        except ValueError as e:
            assert "item count" in str(e).lower()
    print("  PASS: test_session_item_count_mismatch_rejected")


def test_session_missing_required_field_rejected():
    with tempfile.TemporaryDirectory() as d:
        session_path = Path(d) / "session.json"
        # Write incomplete session
        session_path.write_text(json.dumps({"schema_version": 1, "run_id": "test-run"}))
        queue = _make_queue()
        fp = _queue_fingerprint(queue)
        try:
            _load_or_create_session(session_path, "test-run", "rater-01", fp, 30)
            assert False
        except ValueError as e:
            assert "missing" in str(e).lower()
    print("  PASS: test_session_missing_required_field_rejected")


def test_session_queue_fingerprint_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        session_path = Path(d) / "session.json"
        fp = _queue_fingerprint(queue)
        _load_or_create_session(session_path, "test-run", "rater-01", fp, 30)
        try:
            _load_or_create_session(session_path, "test-run", "rater-01", "sha256:different", 30)
            assert False
        except ValueError as e:
            assert "does not match" in str(e)
    print("  PASS: test_session_queue_fingerprint_mismatch_rejected")


def test_session_queue_fingerprint_missing_rejected():
    """Missing queue_fingerprint field specifically rejected."""
    with tempfile.TemporaryDirectory() as d:
        session_path = Path(d) / "session.json"
        # Write session without queue_fingerprint
        session = {
            "schema_version": 1, "run_id": "test-run", "rater_id": "rater-01",
            "queue_item_count": 30, "ratings_file": "ratings.jsonl",
        }
        session_path.write_text(json.dumps(session))
        original_bytes = session_path.read_bytes()
        queue = _make_queue()
        fp = _queue_fingerprint(queue)
        try:
            _load_or_create_session(session_path, "test-run", "rater-01", fp, 30)
            assert False
        except ValueError as e:
            assert "missing" in str(e).lower()
            assert "queue_fingerprint" in str(e)
        # Sidecar unchanged
        assert session_path.read_bytes() == original_bytes
    print("  PASS: test_session_queue_fingerprint_missing_rejected")


def test_session_ratings_filename_mismatch_rejected():
    """Session with different ratings_file is rejected."""
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        session_path = Path(d) / "session.json"
        fp = _queue_fingerprint(queue)
        # Create session with default filename
        _load_or_create_session(session_path, "test-run", "rater-01", fp, 30, "ratings.jsonl")
        original_bytes = session_path.read_bytes()
        # Try to resume with a different filename
        try:
            _load_or_create_session(session_path, "test-run", "rater-01", fp, 30, "other-ratings.jsonl")
            assert False
        except ValueError as e:
            assert "ratings file mismatch" in str(e).lower()
        # Sidecar unchanged
        assert session_path.read_bytes() == original_bytes
    print("  PASS: test_session_ratings_filename_mismatch_rejected")


# ===========================================================================
# 5. JSONL rater/run ID validation (per-record)
# ===========================================================================

def test_rating_record_rater_id_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        rp = Path(d) / "ratings.jsonl"
        rec = {"schema_version": 1, "run_id": "test-run", "rater_id": "wrong",
               "blinded_id": "HR-0000", "rating": 3, "note": None, "flagged": False, "rated_at": "t"}
        rp.write_text(json.dumps(rec) + "\n")
        queue_ids = {it["blinded_id"] for it in queue["items"]}
        try:
            _load_existing_ratings(rp, queue_ids, "test-run", "rater-01")
            assert False
        except ValueError as e:
            assert "Rater ID" in str(e)
    print("  PASS: test_rating_record_rater_id_mismatch_rejected")


def test_rating_record_run_id_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        rp = Path(d) / "ratings.jsonl"
        rec = {"schema_version": 1, "run_id": "wrong-run", "rater_id": "rater-01",
               "blinded_id": "HR-0000", "rating": 3, "note": None, "flagged": False, "rated_at": "t"}
        rp.write_text(json.dumps(rec) + "\n")
        queue_ids = {it["blinded_id"] for it in queue["items"]}
        try:
            _load_existing_ratings(rp, queue_ids, "test-run", "rater-01")
            assert False
        except ValueError as e:
            assert "Run ID" in str(e)
    print("  PASS: test_rating_record_run_id_mismatch_rejected")


# ===========================================================================
# 6. Invalid input does not append or sync
# ===========================================================================

def test_invalid_input_no_append_no_fsync():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        ratings_path = Path(d) / "ratings.jsonl"
        session_path = Path(d) / "session.json"
        out_fn, _ = _capture()

        fsync_calls = []
        orig_fsync = os.fsync
        def tracking_fsync(fd):
            fsync_calls.append(fd)
            return orig_fsync(fd)

        with patch("benchmarks.human_rating.rate.os.fsync", side_effect=tracking_fsync):
            _run_rating_session(
                queue, ratings_path, session_path, "test-run", "rater-01",
                input_fn=_mock_input(["0", "6", "3.5", "four", "4 extra", "q"]),
                output_fn=out_fn,
            )

        # No valid rating → no file, no fsync
        assert not ratings_path.exists() or ratings_path.read_text().strip() == ""
        assert fsync_calls == []
    print("  PASS: test_invalid_input_no_append_no_fsync")


# ===========================================================================
# 7. Completion summary blinded
# ===========================================================================

def test_completion_summary_contains_no_private_metadata():
    with tempfile.TemporaryDirectory() as d:
        records, lines, _ = _run_session(d, ["3"] * 30)
        assert len(records) == 30
        # Summary output present
        combined = " ".join(lines).lower()
        assert "complete" in combined
        assert "30" in combined
        # No private metadata
        for forbidden in ("naive_concat", "vanilla_rag", "mem0_default",
                          "kernel_shared", "judge_score", "integration_score",
                          "duplicate", "answer_key"):
            assert forbidden not in combined, f"Found '{forbidden}' in completion output"
    print("  PASS: test_completion_summary_contains_no_private_metadata")


# ===========================================================================
# Existing core tests (30-item queues)
# ===========================================================================

def test_30_item_enforcement_cli():
    with tempfile.TemporaryDirectory() as d:
        q = _make_queue()
        q["items"] = q["items"][:29]
        q["item_count"] = 29
        qp = _write_queue_file(d, q)
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate", "--queue", str(qp)],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert result.returncode != 0
        assert "30" in result.stderr
    print("  PASS: test_30_item_enforcement_cli")


def test_keyboard_interrupt_preserves():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        ratings_path = Path(d) / "ratings.jsonl"
        session_path = Path(d) / "session.json"
        out_fn, _ = _capture()
        call_count = [0]
        def interrupt_input(prompt=""):
            call_count[0] += 1
            if call_count[0] == 1:
                return "4"
            raise KeyboardInterrupt()
        _run_rating_session(
            queue, ratings_path, session_path, "test-run", "rater-01",
            input_fn=interrupt_input, output_fn=out_fn,
        )
        records = [json.loads(l) for l in ratings_path.read_text().strip().split("\n") if l.strip()]
        assert len(records) == 1
    print("  PASS: test_keyboard_interrupt_preserves")


def test_append_only_bytes():
    with tempfile.TemporaryDirectory() as d:
        queue = _make_queue()
        ratings_path = Path(d) / "ratings.jsonl"
        session_path = Path(d) / "session.json"
        out_fn, _ = _capture()
        _run_session(d, ["3", "4", "q"])
        original_bytes = ratings_path.read_bytes()
        _run_session(d, ["5", "q"])
        new_bytes = ratings_path.read_bytes()
        assert new_bytes.startswith(original_bytes)
    print("  PASS: test_append_only_bytes")


def test_cli_no_private_access():
    rate_path = os.path.join(_project_root, "benchmarks", "human_rating", "rate.py")
    with open(rate_path) as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "answer_key" not in module.lower()
            assert "artifact_writer" not in module.lower()
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.human_rating.rate", "--help"],
        capture_output=True, text=True, cwd=_project_root,
    )
    for forbidden in ("answer-key", "answer_key", "private", "manifest"):
        assert forbidden not in result.stdout.lower()
    print("  PASS: test_cli_no_private_access")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def main():
    print("=== Rating CLI Tests ===\n")

    print("Quit/EOF:")
    test_quit_preserves_progress()
    test_eof_preserves_progress()

    print("\nLock release:")
    test_lock_released_after_normal_completion()
    test_lock_released_after_quit()
    test_lock_released_after_startup_validation_error()
    test_lock_released_after_keyboard_interrupt()
    test_lock_released_after_eof()

    print("\nPermissions:")
    test_rating_session_permissions_posix()
    test_ratings_file_permissions_posix()
    test_lock_file_permissions_posix()

    print("\nSession metadata:")
    test_session_item_count_mismatch_rejected()
    test_session_missing_required_field_rejected()
    test_session_queue_fingerprint_mismatch_rejected()
    test_session_queue_fingerprint_missing_rejected()
    test_session_ratings_filename_mismatch_rejected()

    print("\nPer-record validation:")
    test_rating_record_rater_id_mismatch_rejected()
    test_rating_record_run_id_mismatch_rejected()

    print("\nInvalid input:")
    test_invalid_input_no_append_no_fsync()

    print("\nCompletion:")
    test_completion_summary_contains_no_private_metadata()

    print("\nCore:")
    test_30_item_enforcement_cli()
    test_keyboard_interrupt_preserves()
    test_append_only_bytes()
    test_cli_no_private_access()

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
