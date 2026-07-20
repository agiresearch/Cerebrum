"""End-to-end lifecycle and regression tests for the human-rating workflow.

Run:
    python benchmarks/human_rating/tests/test_end_to_end.py
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.rate import _queue_fingerprint


# ---------------------------------------------------------------------------
# Frozen expected values from the approved protocol
# ---------------------------------------------------------------------------

EXPECTED_SELECTED_TRIALS = {
    "kernel_shared": {"126", "15", "26", "52", "64", "96"},
    "mem0_default": {"106", "108", "121", "21", "42", "63"},
    "naive_concat": {"10", "119", "30", "41", "59", "98"},
    "vanilla_rag": {"133", "145", "26", "34", "55", "80"},
}

EXPECTED_FIRST_BLINDED_ID = "HR-63DE05"
EXPECTED_LAST_BLINDED_ID = "HR-A0CEC8"

EXPECTED_BLINDED_ID_ORDER = (
    "HR-63DE05", "HR-D7E20F", "HR-228EA7", "HR-430CD6", "HR-7FA711",
    "HR-0B8F46", "HR-880837", "HR-0DA6F8", "HR-2C9D04", "HR-ED3910",
    "HR-38F34E", "HR-113F9A", "HR-406269", "HR-8E8F6C", "HR-8405DD",
    "HR-70902A", "HR-30E37B", "HR-40B036", "HR-878841", "HR-433896",
    "HR-8E4F8D", "HR-F146A1", "HR-0AA75F", "HR-C81B50", "HR-C7EACB",
    "HR-34037C", "HR-01D54A", "HR-698FA6", "HR-1CA71E", "HR-A0CEC8",
)

EXPECTED_DUPLICATED_SOURCES = {
    ("kernel_shared", "26"),
    ("kernel_shared", "52"),
    ("mem0_default", "108"),
    ("naive_concat", "10"),
    ("naive_concat", "59"),
    ("vanilla_rag", "26"),
}

EXPECTED_REPORT_FILES = {
    "summary.json", "primary_items.csv", "all_appearances.csv",
    "method_summary.csv", "duplicate_consistency.csv", "confusion_matrix.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_real_results():
    return os.path.isdir(os.path.join(_project_root, "results", "gpt4o_naive_concat"))


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Complete lifecycle integration
# ---------------------------------------------------------------------------

def test_complete_lifecycle():
    """Full lifecycle: generate → rate (partial + resume) → compile."""
    if not _has_real_results():
        print("  SKIP: test_complete_lifecycle (results not present)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = os.path.join(tmpdir, "runs")

        # Generate
        r = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--generate", "--run-id", "e2e-test", "--runs-dir", runs_dir],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert r.returncode == 0, f"Generate failed: {r.stderr}"

        queue_path = os.path.join(runs_dir, "e2e-test", "rater", "rating_queue.json")
        key_path = os.path.join(runs_dir, "e2e-test", "private", "answer_key.json")
        ratings_path = os.path.join(runs_dir, "e2e-test", "rater", "ratings.jsonl")
        session_path = os.path.join(runs_dir, "e2e-test", "rater", "rating_session.json")
        assert os.path.exists(queue_path)
        assert os.path.exists(key_path)

        # Rate: partial (15 ratings then quit)
        partial_input = "\n".join(["3"] * 15 + ["q"]) + "\n"
        r = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", queue_path, "--ratings", ratings_path, "--rater-id", "e2e-rater"],
            input=partial_input, capture_output=True, text=True, cwd=_project_root, timeout=30,
        )
        assert r.returncode == 0
        assert os.path.exists(ratings_path)
        with open(ratings_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 15

        # Resume and complete
        remaining_input = "\n".join(["4"] * 15) + "\n"
        r = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", queue_path, "--ratings", ratings_path, "--rater-id", "e2e-rater"],
            input=remaining_input, capture_output=True, text=True, cwd=_project_root, timeout=30,
        )
        assert r.returncode == 0
        with open(ratings_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 30

        # Compile
        compiled_dir = os.path.join(runs_dir, "e2e-test", "private", "compiled")
        r = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.compile_ratings",
             "--queue", queue_path, "--ratings", ratings_path,
             "--session", session_path, "--answer-key", key_path,
             "--output-dir", compiled_dir],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert r.returncode == 0, f"Compile failed: {r.stderr}"
        assert os.path.exists(os.path.join(compiled_dir, "summary.json"))
        assert os.path.exists(os.path.join(compiled_dir, "primary_items.csv"))
        assert os.path.exists(os.path.join(compiled_dir, "all_appearances.csv"))
        assert os.path.exists(os.path.join(compiled_dir, "method_summary.csv"))
        assert os.path.exists(os.path.join(compiled_dir, "duplicate_consistency.csv"))
        assert os.path.exists(os.path.join(compiled_dir, "confusion_matrix.csv"))

    print("  PASS: test_complete_lifecycle")


# ---------------------------------------------------------------------------
# 2. Frozen selection regression
# ---------------------------------------------------------------------------

def test_frozen_selected_trial_identities():
    if not _has_real_results():
        print("  SKIP: test_frozen_selected_trial_identities")
        return

    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials

    trials, _ = load_eligible_trials(
        results_root=os.path.join(_project_root, "results"), min_per_method=6)
    selected, summary = sample_unique_trials(trials, seed=20260715)

    for method, expected_ids in EXPECTED_SELECTED_TRIALS.items():
        actual_ids = set(summary.selected_trial_ids_by_method[method])
        assert actual_ids == expected_ids, (
            f"{method}: expected {expected_ids}, got {actual_ids}"
        )
    print("  PASS: test_frozen_selected_trial_identities")


# ---------------------------------------------------------------------------
# 3. Frozen blinded plan regression
# ---------------------------------------------------------------------------

def test_frozen_blinded_plan():
    """Complete 30-ID order and duplicate identities match frozen protocol."""
    if not _has_real_results():
        print("  SKIP: test_frozen_blinded_plan")
        return

    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials
    from benchmarks.human_rating.blinding import build_blinded_plan, validate_blinded_plan

    trials, _ = load_eligible_trials(
        results_root=os.path.join(_project_root, "results"), min_per_method=6)
    selected, _ = sample_unique_trials(trials, seed=20260715)
    plan = build_blinded_plan(selected, seed=20260716)
    validate_blinded_plan(plan)

    ordered = sorted(plan.items, key=lambda x: x.appearance_index)

    # Full 30-ID order
    actual_ids = tuple(item.blinded_id for item in ordered)
    assert actual_ids == EXPECTED_BLINDED_ID_ORDER, (
        f"Blinded ID order mismatch.\nExpected: {EXPECTED_BLINDED_ID_ORDER}\nActual:   {actual_ids}"
    )

    # Exact six duplicated source identities
    assert set(plan.duplicated_source_identities) == EXPECTED_DUPLICATED_SOURCES

    # Each duplicated identity appears exactly twice; others exactly once
    from collections import Counter
    identity_counts = Counter(
        (item.source_trial.source_method, item.source_trial.original_trial_id)
        for item in ordered
    )
    for identity, count in identity_counts.items():
        if identity in EXPECTED_DUPLICATED_SOURCES:
            assert count == 2, f"{identity} should appear twice, got {count}"
        else:
            assert count == 1, f"{identity} should appear once, got {count}"

    # Duplicate separation
    for item in ordered:
        if item.duplicate_of_blinded_id:
            orig = next(i for i in ordered if i.blinded_id == item.duplicate_of_blinded_id)
            assert abs(item.appearance_index - orig.appearance_index) >= 3

    print("  PASS: test_frozen_blinded_plan")


# ---------------------------------------------------------------------------
# 4. Deterministic regeneration
# ---------------------------------------------------------------------------

def test_deterministic_regeneration():
    """Full file-level determinism: queue and answer-key bytes match across runs."""
    if not _has_real_results():
        print("  SKIP: test_deterministic_regeneration")
        return

    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials
    from benchmarks.human_rating.blinding import build_blinded_plan
    from benchmarks.human_rating.artifact_writer import write_rating_artifacts
    from benchmarks.human_rating.paths import get_run_paths

    trials, _ = load_eligible_trials(
        results_root=os.path.join(_project_root, "results"), min_per_method=6)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        for d in (d1, d2):
            s, _ = sample_unique_trials(trials, seed=20260715)
            p = build_blinded_plan(s, seed=20260716)
            rp = get_run_paths("det", base_dir=d)
            write_rating_artifacts(p, run_paths=rp, run_id="det",
                                   protocol_name="t", sampling_seed=20260715, blinding_seed=20260716)

        q1 = Path(d1) / "det" / "rater" / "rating_queue.json"
        q2 = Path(d2) / "det" / "rater" / "rating_queue.json"
        k1 = Path(d1) / "det" / "private" / "answer_key.json"
        k2 = Path(d2) / "det" / "private" / "answer_key.json"

        assert q1.read_bytes() == q2.read_bytes(), "Queue files differ"
        assert k1.read_bytes() == k2.read_bytes(), "Answer key files differ"

    print("  PASS: test_deterministic_regeneration")


# ---------------------------------------------------------------------------
# 5. Isolation regression
# ---------------------------------------------------------------------------

def test_rate_module_isolation():
    """rate.py does not import compiler, artifact_writer, or reference answer_key."""
    rate_path = os.path.join(_project_root, "benchmarks", "human_rating", "rate.py")
    with open(rate_path) as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "compile" not in module.lower(), f"rate.py imports {module}"
            assert "artifact_writer" not in module.lower(), f"rate.py imports {module}"
            assert "answer_key" not in module.lower(), f"rate.py imports {module}"
    assert "answer_key" not in source.lower().replace("answer-key", "").replace("# answer", "")
    print("  PASS: test_rate_module_isolation")


def test_queue_contains_no_private_fields():
    """Queue JSON at no depth contains private metadata."""
    if not _has_real_results():
        print("  SKIP: test_queue_contains_no_private_fields")
        return

    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials
    from benchmarks.human_rating.blinding import build_blinded_plan
    from benchmarks.human_rating.artifact_writer import build_rating_queue, validate_queue_document

    trials, _ = load_eligible_trials(
        results_root=os.path.join(_project_root, "results"), min_per_method=6)
    selected, _ = sample_unique_trials(trials, seed=20260715)
    plan = build_blinded_plan(selected, seed=20260716)
    queue = build_rating_queue(plan, run_id="isolation-test")

    errors = validate_queue_document(queue)
    assert errors == [], f"Queue leakage: {errors}"

    # Deep string search
    queue_str = json.dumps(queue)
    for forbidden in ("source_method", "original_trial_id", "judge_score",
                      "integration_score", "duplicate_of_blinded_id",
                      "sampling_seed", "blinding_seed"):
        assert forbidden not in queue_str, f"Queue contains '{forbidden}'"

    print("  PASS: test_queue_contains_no_private_fields")


# ---------------------------------------------------------------------------
# 6. Source results unchanged
# ---------------------------------------------------------------------------

def test_workflow_does_not_modify_source_results():
    if not _has_real_results():
        print("  SKIP: test_workflow_does_not_modify_source_results")
        return

    results_dir = os.path.join(_project_root, "results")
    files_to_check = [
        "gpt4o_naive_concat/results_naive_concat.json",
        "gpt4o_vanilla_rag/results_vanilla_rag.json",
        "gpt4o_mem0_default/results_mem0_default.json",
        "gpt4o_kernel_shared/results_kernel_shared.json",
    ]
    hashes_before = {f: _hash_file(os.path.join(results_dir, f)) for f in files_to_check}

    # Run the full generation pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--generate", "--run-id", "hash-test", "--runs-dir", tmpdir],
            capture_output=True, text=True, cwd=_project_root,
        )

    hashes_after = {f: _hash_file(os.path.join(results_dir, f)) for f in files_to_check}
    assert hashes_before == hashes_after, "Source results modified!"
    print("  PASS: test_workflow_does_not_modify_source_results")


def test_generated_runs_are_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "benchmarks/human_rating/runs/test/rater/queue.json"],
        capture_output=True, text=True, cwd=_project_root,
    )
    assert result.returncode == 0, "runs/ should be gitignored"
    print("  PASS: test_generated_runs_are_gitignored")


# ---------------------------------------------------------------------------
# 7. Private artifacts never enter rater directory
# ---------------------------------------------------------------------------

def test_private_artifacts_never_enter_rater_directory():
    """Answer key and compiled reports are never under rater/."""
    if not _has_real_results():
        print("  SKIP: test_private_artifacts_never_enter_rater_directory")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = os.path.join(tmpdir, "runs")
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--generate", "--run-id", "placement-test", "--runs-dir", runs_dir],
            capture_output=True, text=True, cwd=_project_root,
        )
        rater_dir = Path(runs_dir) / "placement-test" / "rater"
        private_dir = Path(runs_dir) / "placement-test" / "private"

        # Answer key in private/
        assert (private_dir / "answer_key.json").exists()
        # Not in rater/
        assert not any(p.name == "answer_key.json" for p in rater_dir.rglob("*"))
        # No compiled reports in rater/
        compiled_names = {"summary.json", "primary_items.csv", "all_appearances.csv",
                          "method_summary.csv", "duplicate_consistency.csv", "confusion_matrix.csv"}
        rater_files = {p.name for p in rater_dir.rglob("*") if p.is_file()}
        assert rater_files & compiled_names == set()

    print("  PASS: test_private_artifacts_never_enter_rater_directory")


# ---------------------------------------------------------------------------
# 8. Shared-memory benchmark boundary
# ---------------------------------------------------------------------------

def test_workflow_does_not_modify_shared_memory_benchmark():
    """The entire workflow doesn't touch benchmarks/shared_memory/."""
    if not _has_real_results():
        print("  SKIP: test_workflow_does_not_modify_shared_memory_benchmark")
        return

    sm_dir = os.path.join(_project_root, "benchmarks", "shared_memory")
    hashes_before = {}
    for root, _, files in os.walk(sm_dir):
        for f in files:
            if f.endswith(".py") or f.endswith(".json"):
                fp = os.path.join(root, f)
                hashes_before[fp] = _hash_file(fp)

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--generate", "--run-id", "boundary-test", "--runs-dir", tmpdir],
            capture_output=True, text=True, cwd=_project_root,
        )

    hashes_after = {}
    for fp in hashes_before:
        hashes_after[fp] = _hash_file(fp)

    assert hashes_before == hashes_after, "shared_memory benchmark files were modified!"
    print("  PASS: test_workflow_does_not_modify_shared_memory_benchmark")


# ---------------------------------------------------------------------------
# 9. Complete compiled artifact set and population counts
# ---------------------------------------------------------------------------

def test_compiled_artifact_set_and_counts():
    """Compiled directory has exactly the expected files with correct populations."""
    if not _has_real_results():
        print("  SKIP: test_compiled_artifact_set_and_counts")
        return

    import csv as csv_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = os.path.join(tmpdir, "runs")

        # Generate
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.sample_and_blind",
             "--generate", "--run-id", "count-test", "--runs-dir", runs_dir],
            capture_output=True, text=True, cwd=_project_root,
        )

        queue_path = os.path.join(runs_dir, "count-test", "rater", "rating_queue.json")
        key_path = os.path.join(runs_dir, "count-test", "private", "answer_key.json")
        ratings_path = os.path.join(runs_dir, "count-test", "rater", "ratings.jsonl")
        session_path = os.path.join(runs_dir, "count-test", "rater", "rating_session.json")

        # Rate all 30
        rating_input = "\n".join(["3"] * 30) + "\n"
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.rate",
             "--queue", queue_path, "--ratings", ratings_path, "--rater-id", "count-rater"],
            input=rating_input, capture_output=True, text=True, cwd=_project_root, timeout=30,
        )

        # Compile
        compiled_dir = os.path.join(runs_dir, "count-test", "private", "compiled")
        subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.compile_ratings",
             "--queue", queue_path, "--ratings", ratings_path,
             "--session", session_path, "--answer-key", key_path,
             "--output-dir", compiled_dir],
            capture_output=True, text=True, cwd=_project_root,
        )

        # Exact file set
        actual_files = {p.name for p in Path(compiled_dir).iterdir() if p.is_file()}
        assert actual_files == EXPECTED_REPORT_FILES, f"Got: {actual_files}"

        # Row counts
        with open(os.path.join(compiled_dir, "primary_items.csv")) as f:
            rows = list(csv_mod.reader(f))
        assert len(rows) == 25  # header + 24

        with open(os.path.join(compiled_dir, "all_appearances.csv")) as f:
            rows = list(csv_mod.reader(f))
        assert len(rows) == 31  # header + 30

        with open(os.path.join(compiled_dir, "method_summary.csv")) as f:
            rows = list(csv_mod.reader(f))
        assert len(rows) == 5  # header + 4

        with open(os.path.join(compiled_dir, "duplicate_consistency.csv")) as f:
            rows = list(csv_mod.reader(f))
        assert len(rows) == 7  # header + 6

        with open(os.path.join(compiled_dir, "confusion_matrix.csv")) as f:
            rows = list(csv_mod.reader(f))
        assert len(rows) == 6  # header + 5

        # summary.json method summaries
        with open(os.path.join(compiled_dir, "summary.json")) as f:
            summary = json.load(f)
        assert len(summary["primary_method_summaries"]) == 4
        assert len(summary["appearance_method_summaries"]) == 4
        assert sum(ms["item_count"] for ms in summary["primary_method_summaries"]) == 24
        assert sum(ms["item_count"] for ms in summary["appearance_method_summaries"]) == 30

        # Ratings file has 30 records
        with open(ratings_path) as f:
            rating_lines = [l for l in f if l.strip()]
        assert len(rating_lines) == 30

    print("  PASS: test_compiled_artifact_set_and_counts")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== End-to-End Tests ===\n")

    print("Lifecycle:")
    test_complete_lifecycle()

    print("\nFrozen protocol regression:")
    test_frozen_selected_trial_identities()
    test_frozen_blinded_plan()
    test_deterministic_regeneration()

    print("\nIsolation:")
    test_rate_module_isolation()
    test_queue_contains_no_private_fields()

    print("\nRepository boundary:")
    test_workflow_does_not_modify_source_results()
    test_workflow_does_not_modify_shared_memory_benchmark()
    test_generated_runs_are_gitignored()
    test_private_artifacts_never_enter_rater_directory()
    test_compiled_artifact_set_and_counts()

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
