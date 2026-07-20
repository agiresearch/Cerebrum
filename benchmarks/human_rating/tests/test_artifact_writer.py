"""Tests for artifact writing, atomicity, and filesystem isolation.

Run:
    python benchmarks/human_rating/tests/test_artifact_writer.py
"""

import ast
import glob
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

from benchmarks.human_rating.artifact_writer import (
    ALLOWED_QUEUE_ITEM_KEYS, WrittenArtifacts,
    build_answer_key, build_rating_queue, validate_artifact_pair,
    validate_artifact_paths, validate_queue_document,
    write_rating_artifacts,
)
from benchmarks.human_rating.blinding import build_blinded_plan
from benchmarks.human_rating.paths import get_run_paths
from benchmarks.human_rating.schemas import SourceTrial, VALID_SOURCE_METHODS


def _make_trial(method, tid, score=3):
    return SourceTrial(
        source_method=method, original_trial_id=str(tid), model="gpt-4o",
        evaluation_dimension="integration",
        reference_context=f"Ctx {method}/{tid}",
        reference_context_provenance="synthetic_source_context",
        is_exact_model_visible_context=False,
        question=f"Q{tid}", response=f"R{tid}",
        judge_score=score, profile_usage_score=score,
        task_usage_score=score, integration_score=score,
    )


def _make_plan(seed=42):
    trials = []
    for method in sorted(VALID_SOURCE_METHODS):
        for i in range(6):
            trials.append(_make_trial(method, i))
    return build_blinded_plan(trials, seed=seed)


def _std_write(tmpdir, run_id="test", overwrite=False):
    rp = get_run_paths(run_id, base_dir=tmpdir)
    plan = _make_plan()
    return write_rating_artifacts(
        plan, run_paths=rp, run_id=run_id,
        protocol_name="p", sampling_seed=1, blinding_seed=2,
        overwrite=overwrite,
    ), rp


# === Path separation tests ===

def test_same_directory_rejected():
    q = Path("/tmp/run/shared/queue.json")
    k = Path("/tmp/run/shared/key.json")
    try:
        validate_artifact_paths(q, k)
        assert False
    except ValueError as e:
        assert "separate directories" in str(e)
    print("  PASS: test_same_directory_rejected")


def test_nested_rater_inside_private_rejected():
    q = Path("/tmp/run/private/rater/queue.json")
    k = Path("/tmp/run/private/key.json")
    try:
        validate_artifact_paths(q, k)
        assert False
    except ValueError as e:
        assert "nested" in str(e).lower()
    print("  PASS: test_nested_rater_inside_private_rejected")


def test_nested_private_inside_rater_rejected():
    q = Path("/tmp/run/rater/queue.json")
    k = Path("/tmp/run/rater/private/key.json")
    try:
        validate_artifact_paths(q, k)
        assert False
    except ValueError as e:
        assert "nested" in str(e).lower()
    print("  PASS: test_nested_private_inside_rater_rejected")


# === One-sided existing output ===

def test_queue_exists_key_missing_writes_nothing():
    with tempfile.TemporaryDirectory() as tmpdir:
        rp = get_run_paths("test", base_dir=tmpdir)
        # Pre-create only the queue
        os.makedirs(os.path.dirname(rp.rating_queue_path), exist_ok=True)
        Path(rp.rating_queue_path).write_text('{"existing": true}')
        original_content = Path(rp.rating_queue_path).read_text()

        plan = _make_plan()
        try:
            write_rating_artifacts(
                plan, run_paths=rp, run_id="test",
                protocol_name="p", sampling_seed=1, blinding_seed=2,
            )
            assert False, "Should raise FileExistsError"
        except FileExistsError:
            pass

        # Queue unchanged
        assert Path(rp.rating_queue_path).read_text() == original_content
        # Key not created
        assert not Path(rp.answer_key_path).exists()
    print("  PASS: test_queue_exists_key_missing_writes_nothing")


def test_key_exists_queue_missing_writes_nothing():
    with tempfile.TemporaryDirectory() as tmpdir:
        rp = get_run_paths("test", base_dir=tmpdir)
        # Pre-create only the answer key
        os.makedirs(os.path.dirname(rp.answer_key_path), exist_ok=True)
        Path(rp.answer_key_path).write_text('{"existing": true}')
        original_content = Path(rp.answer_key_path).read_text()

        plan = _make_plan()
        try:
            write_rating_artifacts(
                plan, run_paths=rp, run_id="test",
                protocol_name="p", sampling_seed=1, blinding_seed=2,
            )
            assert False, "Should raise FileExistsError"
        except FileExistsError:
            pass

        # Key unchanged
        assert Path(rp.answer_key_path).read_text() == original_content
        # Queue not created
        assert not Path(rp.rating_queue_path).exists()
    print("  PASS: test_key_exists_queue_missing_writes_nothing")


# === Failure during second replace (paired rollback) ===

def test_second_replace_failure_rollback():
    """If the key replace fails after queue was placed, queue is removed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rp = get_run_paths("test", base_dir=tmpdir)
        plan = _make_plan()

        original_replace = os.replace
        call_count = [0]

        def failing_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second replace (answer key)
                raise OSError("Simulated disk failure")
            return original_replace(src, dst)

        with patch("benchmarks.human_rating.artifact_writer.os.replace", side_effect=failing_replace):
            try:
                write_rating_artifacts(
                    plan, run_paths=rp, run_id="test",
                    protocol_name="p", sampling_seed=1, blinding_seed=2,
                )
                assert False, "Should raise OSError"
            except OSError as e:
                assert "Simulated" in str(e)

        # Neither final file should exist (queue rolled back)
        assert not Path(rp.rating_queue_path).exists(), "Queue should be rolled back"
        assert not Path(rp.answer_key_path).exists(), "Key should not exist"
    print("  PASS: test_second_replace_failure_rollback")


# === Temporary file cleanup ===

def test_temporary_files_removed_after_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        _std_write(tmpdir)
        # Check no .tmp or .backup files remain
        for pattern in ("**/*.tmp", "**/*.backup"):
            found = glob.glob(os.path.join(tmpdir, pattern), recursive=True)
            assert found == [], f"Leftover files: {found}"
    print("  PASS: test_temporary_files_removed_after_success")


def test_temporary_files_removed_after_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        rp = get_run_paths("test", base_dir=tmpdir)
        plan = _make_plan()

        with patch("benchmarks.human_rating.artifact_writer.os.replace", side_effect=OSError("fail")):
            try:
                write_rating_artifacts(
                    plan, run_paths=rp, run_id="test",
                    protocol_name="p", sampling_seed=1, blinding_seed=2,
                )
            except OSError:
                pass

        for pattern in ("**/*.tmp", "**/*.backup"):
            found = glob.glob(os.path.join(tmpdir, pattern), recursive=True)
            assert found == [], f"Leftover files after failure: {found}"
    print("  PASS: test_temporary_files_removed_after_failure")


# === Preview modes write nothing ===

def test_preview_selection_writes_no_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--preview-selection", "--runs-dir", tmpdir],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert result.returncode == 0
        # Runs dir should not exist or be empty
        contents = list(Path(tmpdir).rglob("*"))
        json_files = [p for p in contents if p.suffix == ".json"]
        assert json_files == [], f"Preview wrote files: {json_files}"
    print("  PASS: test_preview_selection_writes_no_files")


def test_preview_blinded_plan_writes_no_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--preview-blinded-plan", "--runs-dir", tmpdir],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert result.returncode == 0
        contents = list(Path(tmpdir).rglob("*"))
        json_files = [p for p in contents if p.suffix == ".json"]
        assert json_files == [], f"Preview wrote files: {json_files}"
    print("  PASS: test_preview_blinded_plan_writes_no_files")


# === Rating CLI surface ===

def test_rate_help_has_no_answer_key_option():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.human_rating.rate", "--help"],
        capture_output=True, text=True, cwd=_project_root,
    )
    assert result.returncode == 0
    help_text = result.stdout.lower()
    for forbidden in ("answer-key", "answer_key", "private", "manifest"):
        assert forbidden not in help_text, f"rate --help contains '{forbidden}'"
    print("  PASS: test_rate_help_has_no_answer_key_option")


def test_rate_no_private_imports():
    rate_path = os.path.join(_project_root, "benchmarks", "human_rating", "rate.py")
    with open(rate_path) as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "answer_key" not in module.lower()
            assert "artifact_writer" not in module.lower()
            assert "private" not in module.lower()
    print("  PASS: test_rate_no_private_imports")


# === Git ignore ===

def test_gitignore_covers_runs():
    result = subprocess.run(
        ["git", "check-ignore", "benchmarks/human_rating/runs/example/rater/rating_queue.json"],
        capture_output=True, text=True, cwd=_project_root,
    )
    assert result.returncode == 0, "runs/ should be ignored by .gitignore"
    result2 = subprocess.run(
        ["git", "check-ignore", "benchmarks/human_rating/runs/example/private/answer_key.json"],
        capture_output=True, text=True, cwd=_project_root,
    )
    assert result2.returncode == 0, "private/ should be ignored by .gitignore"
    print("  PASS: test_gitignore_covers_runs")


# === Existing tests (preserved) ===

def test_queue_no_private_leakage():
    plan = _make_plan()
    queue = build_rating_queue(plan, run_id="t")
    errors = validate_queue_document(queue)
    assert errors == [], f"Leakage: {errors}"
    print("  PASS: test_queue_no_private_leakage")


def test_overwrite_protection():
    with tempfile.TemporaryDirectory() as tmpdir:
        _std_write(tmpdir)
        rp = get_run_paths("test", base_dir=tmpdir)
        plan = _make_plan()
        try:
            write_rating_artifacts(
                plan, run_paths=rp, run_id="test",
                protocol_name="p", sampling_seed=1, blinding_seed=2,
            )
            assert False
        except FileExistsError:
            pass
    print("  PASS: test_overwrite_protection")


def test_successful_overwrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        _std_write(tmpdir)
        result, _ = _std_write(tmpdir, overwrite=True)
        assert result.item_count == 30
    print("  PASS: test_successful_overwrite")


def test_answer_key_permissions_posix():
    if os.name != "posix":
        print("  SKIP: test_answer_key_permissions_posix")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        result, _ = _std_write(tmpdir)
        mode = stat.S_IMODE(os.stat(result.answer_key_path).st_mode)
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"
    print("  PASS: test_answer_key_permissions_posix")


def test_real_results_generate():
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(os.path.join(results_dir, "gpt4o_naive_concat")):
        print("  SKIP: test_real_results_generate")
        return
    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials
    trials, _ = load_eligible_trials(results_root=results_dir, min_per_method=6)
    selected, _ = sample_unique_trials(trials, seed=20260715)
    plan = build_blinded_plan(selected, seed=20260716)
    with tempfile.TemporaryDirectory() as tmpdir:
        rp = get_run_paths("real-test", base_dir=tmpdir)
        result = write_rating_artifacts(
            plan, run_paths=rp, run_id="real-test",
            protocol_name="mixed_personalization_source_grounded_v1",
            sampling_seed=20260715, blinding_seed=20260716,
        )
        assert result.item_count == 30
        with open(result.rating_queue_path) as f:
            queue = json.load(f)
        assert validate_queue_document(queue) == []
    print("  PASS: test_real_results_generate")


def main():
    print("=== Artifact Writer Tests ===\n")

    print("Path separation:")
    test_same_directory_rejected()
    test_nested_rater_inside_private_rejected()
    test_nested_private_inside_rater_rejected()

    print("\nOne-sided existing output:")
    test_queue_exists_key_missing_writes_nothing()
    test_key_exists_queue_missing_writes_nothing()

    print("\nPaired rollback:")
    test_second_replace_failure_rollback()

    print("\nTemp file cleanup:")
    test_temporary_files_removed_after_success()
    test_temporary_files_removed_after_failure()

    print("\nPreview write-free:")
    test_preview_selection_writes_no_files()
    test_preview_blinded_plan_writes_no_files()

    print("\nRating CLI surface:")
    test_rate_help_has_no_answer_key_option()
    test_rate_no_private_imports()

    print("\nGit ignore:")
    test_gitignore_covers_runs()

    print("\nCore validation:")
    test_queue_no_private_leakage()
    test_overwrite_protection()
    test_successful_overwrite()
    test_answer_key_permissions_posix()
    test_real_results_generate()

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
