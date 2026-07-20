"""Comprehensive compiler tests — validation, metrics, atomicity, CLI.

Run:
    python benchmarks/human_rating/tests/test_compile_ratings.py
"""

import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.compile_ratings import (
    compile_rating_run, write_compilation_outputs,
)
from benchmarks.human_rating.rate import _queue_fingerprint

METHODS = ["kernel_shared", "mem0_default", "naive_concat", "vanilla_rag"]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _build_run(tmpdir, human_ratings=None, judge_scores=None, **overrides):
    """Build a complete valid 30-item run."""
    if human_ratings is None:
        human_ratings = [3] * 30
    if judge_scores is None:
        judge_scores = [3] * 30

    q_items = []
    for i in range(24):
        q_items.append({"blinded_id": f"HR-{i:04d}", "reference_context": f"Ctx {i}",
                        "question": f"Q{i}", "response": f"R{i}"})
    for i in range(6):
        orig = q_items[i]
        q_items.append({"blinded_id": f"HR-{24+i:04d}",
                        "reference_context": orig["reference_context"],
                        "question": orig["question"], "response": orig["response"]})
    queue = {"schema_version": 1, "run_id": "test-run", "item_count": 30, "items": q_items}

    k_items = []
    for i in range(24):
        k_items.append({
            "blinded_id": f"HR-{i:04d}", "queue_position": i + 1,
            "source_method": METHODS[i % 4], "model": "gpt-4o",
            "original_trial_id": str(i), "evaluation_dimension": "integration",
            "judge_score": judge_scores[i], "profile_usage_score": judge_scores[i],
            "task_usage_score": judge_scores[i], "integration_score": judge_scores[i],
            "reference_context_provenance": "synthetic_source_context",
            "is_exact_model_visible_context": False, "duplicate_of_blinded_id": None,
        })
    for i in range(6):
        orig = k_items[i]
        k_items.append({
            "blinded_id": f"HR-{24+i:04d}", "queue_position": 25 + i,
            "source_method": orig["source_method"], "model": "gpt-4o",
            "original_trial_id": orig["original_trial_id"],
            "evaluation_dimension": "integration",
            "judge_score": orig["judge_score"], "profile_usage_score": orig["profile_usage_score"],
            "task_usage_score": orig["task_usage_score"], "integration_score": orig["integration_score"],
            "reference_context_provenance": "synthetic_source_context",
            "is_exact_model_visible_context": False,
            "duplicate_of_blinded_id": orig["blinded_id"],
        })

    answer_key = {"schema_version": 1, "run_id": "test-run", "protocol_name": "test",
                  "sampling_seed": 1, "blinding_seed": 2,
                  "item_count": 30, "unique_source_count": 24, "duplicate_count": 6,
                  "items": k_items}

    ratings = []
    for i in range(30):
        ratings.append({"schema_version": 1, "run_id": "test-run", "rater_id": "rater-01",
                        "blinded_id": f"HR-{i:04d}", "rating": human_ratings[i],
                        "note": overrides.get(f"note_{i}"),
                        "flagged": overrides.get(f"flag_{i}", False),
                        "rated_at": "2026-07-15T10:00:00Z"})

    session = {"schema_version": 1, "run_id": "test-run", "rater_id": "rater-01",
               "queue_fingerprint": _queue_fingerprint(queue),
               "queue_item_count": 30, "ratings_file": "ratings.jsonl",
               "started_at": "2026-07-15T10:00:00Z"}

    rater_dir = Path(tmpdir) / "rater"
    private_dir = Path(tmpdir) / "private"
    rater_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    qp = rater_dir / "rating_queue.json"; qp.write_text(json.dumps(queue))
    sp = rater_dir / "rating_session.json"; sp.write_text(json.dumps(session))
    kp = private_dir / "answer_key.json"; kp.write_text(json.dumps(answer_key))
    rp = rater_dir / "ratings.jsonl"
    with open(rp, "w") as f:
        for r in ratings:
            f.write(json.dumps(r) + "\n")
    return qp, rp, sp, kp


def _compile(tmpdir, **kw):
    paths = _build_run(tmpdir, **kw)
    return compile_rating_run(queue_path=paths[0], ratings_path=paths[1],
                              session_path=paths[2], answer_key_path=paths[3])


def _expect_error(tmpdir, mutator, substring):
    """Build a run, mutate it, expect ValueError containing substring."""
    paths = _build_run(tmpdir)
    mutator(paths)
    try:
        compile_rating_run(queue_path=paths[0], ratings_path=paths[1],
                           session_path=paths[2], answer_key_path=paths[3])
        assert False, f"Should have raised ValueError with '{substring}'"
    except ValueError as e:
        assert substring.lower() in str(e).lower(), f"Expected '{substring}' in: {e}"


# ===========================================================================
# 1. Malformed ratings
# ===========================================================================

def test_malformed_jsonl_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            lines = paths[1].read_text().strip().split("\n")
            lines[4] = '{"blinded_id":"HR-0004","rating":3'  # truncated
            paths[1].write_text("\n".join(lines) + "\n")
        _expect_error(d, m, "line 5")
    print("  PASS: test_malformed_jsonl_rejected")

def test_non_object_jsonl_record_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            lines = paths[1].read_text().strip().split("\n")
            lines[2] = '["array"]'
            paths[1].write_text("\n".join(lines) + "\n")
        _expect_error(d, m, "line 3")
    print("  PASS: test_non_object_jsonl_record_rejected")

def test_malformed_trailing_jsonl_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            with open(paths[1], "a") as f:
                f.write("garbage{not json\n")
        _expect_error(d, m, "line 31")
    print("  PASS: test_malformed_trailing_jsonl_rejected")


# ===========================================================================
# 2. Session metadata
# ===========================================================================

def test_session_item_count_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            s = json.loads(paths[2].read_text()); s["queue_item_count"] = 25
            paths[2].write_text(json.dumps(s))
        _expect_error(d, m, "item count")
    print("  PASS: test_session_item_count_mismatch_rejected")

def test_session_ratings_filename_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            s = json.loads(paths[2].read_text()); s["ratings_file"] = "other.jsonl"
            paths[2].write_text(json.dumps(s))
        _expect_error(d, m, "ratings file")
    print("  PASS: test_session_ratings_filename_mismatch_rejected")

def test_session_missing_required_field_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            s = json.loads(paths[2].read_text()); del s["queue_fingerprint"]
            paths[2].write_text(json.dumps(s))
        _expect_error(d, m, "missing")
    print("  PASS: test_session_missing_required_field_rejected")


# ===========================================================================
# 3. Answer-key structural
# ===========================================================================

def test_answer_key_order_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][0], k["items"][1] = k["items"][1], k["items"][0]
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "order")
    print("  PASS: test_answer_key_order_mismatch_rejected")

def test_answer_key_position_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][5]["queue_position"] = 99
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "position")
    print("  PASS: test_answer_key_position_mismatch_rejected")

def test_broken_duplicate_reference_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][24]["duplicate_of_blinded_id"] = "HR-NONEXIST"
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "duplicate")
    print("  PASS: test_broken_duplicate_reference_rejected")

def test_duplicate_points_to_duplicate_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            # Point dup 25 to dup 24 (which is itself a duplicate)
            k["items"][25]["duplicate_of_blinded_id"] = "HR-0024"
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "duplicate")
    print("  PASS: test_duplicate_points_to_duplicate_rejected")

def test_duplicate_source_metadata_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][24]["source_method"] = "vanilla_rag"  # != original
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "mismatch")
    print("  PASS: test_duplicate_source_metadata_mismatch_rejected")

def test_duplicate_queue_content_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            q = json.loads(paths[0].read_text())
            q["items"][24]["reference_context"] = "DIFFERENT"
            paths[0].write_text(json.dumps(q))
            # Recompute session fingerprint
            s = json.loads(paths[2].read_text())
            s["queue_fingerprint"] = _queue_fingerprint(q)
            paths[2].write_text(json.dumps(s))
        _expect_error(d, m, "queue")
    print("  PASS: test_duplicate_queue_content_mismatch_rejected")


# ===========================================================================
# 4. Protocol constraints
# ===========================================================================

def test_wrong_evaluation_dimension_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][0]["evaluation_dimension"] = "profile_usage"
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "evaluation_dimension")
    print("  PASS: test_wrong_evaluation_dimension_rejected")

def test_wrong_context_provenance_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][0]["reference_context_provenance"] = "stored"
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "provenance")
    print("  PASS: test_wrong_context_provenance_rejected")

def test_exact_model_visible_context_true_rejected():
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            k["items"][0]["is_exact_model_visible_context"] = True
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "exact")
    print("  PASS: test_exact_model_visible_context_true_rejected")


# ===========================================================================
# 5. Overall summary with nontrivial known values
# ===========================================================================

def test_overall_summary_all_fields():
    """Mixed ratings with exact expected values for every field."""
    # 24 primary: [5,4,3,2,1, 4,3,2,1,5, 3,2,1,5,4, 2,1,5,4,3, 1,5,4,3]
    # Judge all=3. Diffs: [+2,+1,0,-1,-2, +1,0,-1,-2,+2, 0,-1,-2,+2,+1, -1,-2,+2,+1,0, -2,+2,+1,0]
    h_primary = [5,4,3,2,1, 4,3,2,1,5, 3,2,1,5,4, 2,1,5,4,3, 1,5,4,3]
    h = h_primary + [3]*6  # dups match judge
    j = [3]*30
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d, human_ratings=h, judge_scores=j)
    s = result.overall_summary
    assert s.item_count == 24
    assert abs(s.mean_human_rating - sum(h_primary)/24) < 0.001
    assert s.mean_judge_score == 3.0
    diffs = [hp - 3 for hp in h_primary]
    assert abs(s.mean_signed_difference - sum(diffs)/24) < 0.001
    abs_diffs = [abs(d) for d in diffs]
    assert abs(s.mean_absolute_error - sum(abs_diffs)/24) < 0.001
    exact_count = sum(1 for hp in h_primary if hp == 3)
    assert s.exact_agreement_count == exact_count
    assert abs(s.exact_agreement_rate - exact_count/24) < 0.001
    w1_count = sum(1 for hp in h_primary if abs(hp-3) <= 1)
    assert s.within_one_count == w1_count
    assert abs(s.within_one_rate - w1_count/24) < 0.001
    print("  PASS: test_overall_summary_all_fields")


# ===========================================================================
# 6. Method summaries
# ===========================================================================

def test_method_flag_and_note_counts():
    with tempfile.TemporaryDirectory() as d:
        # Flag items 0,4 (kernel_shared indices); note item 1 (mem0_default)
        result = _compile(d, flag_0=True, flag_4=True, note_1="note")
    ks = next(ms for ms in result.method_summaries if ms.method == "kernel_shared")
    md = next(ms for ms in result.method_summaries if ms.method == "mem0_default")
    assert ks.flagged_count == 2
    assert md.noted_count == 1
    print("  PASS: test_method_flag_and_note_counts")


# ===========================================================================
# 7. Sensitivity independence
# ===========================================================================

def test_primary_metrics_unchanged_when_duplicates_change():
    """Changing duplicate ratings doesn't affect primary."""
    j = [3]*30
    h1 = [3]*24 + [3]*6  # All match
    h2 = [3]*24 + [1]*6  # Dups differ
    with tempfile.TemporaryDirectory() as d1:
        r1 = _compile(d1, human_ratings=h1, judge_scores=j)
    with tempfile.TemporaryDirectory() as d2:
        r2 = _compile(d2, human_ratings=h2, judge_scores=j)
    # Primary identical
    assert r1.overall_summary == r2.overall_summary
    # Sensitivity differs
    assert r1.appearance_sensitivity_summary != r2.appearance_sensitivity_summary
    print("  PASS: test_primary_metrics_unchanged_when_duplicates_change")


# ===========================================================================
# 8. Duplicate metrics
# ===========================================================================

def test_duplicate_positions_known():
    """Duplicate positions are queue_position values from answer key."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
    for dp in result.duplicate_pairs:
        # Originals at positions 1-6, duplicates at 25-30
        assert dp.original_queue_position <= 24
        assert dp.duplicate_queue_position >= 25
        assert dp.positional_distance == dp.duplicate_queue_position - dp.original_queue_position
    print("  PASS: test_duplicate_positions_known")

def test_duplicate_metrics_ignore_judge_scores():
    """Changing judge scores doesn't affect duplicate consistency."""
    h = [3]*30
    j1 = [3]*30
    j2 = [5]*30  # Different judge scores
    with tempfile.TemporaryDirectory() as d1:
        r1 = _compile(d1, human_ratings=h, judge_scores=j1)
    with tempfile.TemporaryDirectory() as d2:
        r2 = _compile(d2, human_ratings=h, judge_scores=j2)
    # Duplicate consistency depends only on human ratings
    assert r1.duplicate_summary == r2.duplicate_summary
    print("  PASS: test_duplicate_metrics_ignore_judge_scores")


# ===========================================================================
# 9. Confusion matrix
# ===========================================================================

def test_confusion_matrix_labels_and_shape():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
    m = result.confusion_matrix
    assert set(m.keys()) == {1, 2, 3, 4, 5}
    for row in m.values():
        assert set(row.keys()) == {1, 2, 3, 4, 5}
        assert all(isinstance(v, int) for v in row.values())
    print("  PASS: test_confusion_matrix_labels_and_shape")

def test_confusion_matrix_csv_shape():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        with open(out / "confusion_matrix.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 6  # header + 5
        assert rows[0] == ["human\\judge", "1", "2", "3", "4", "5"]
    print("  PASS: test_confusion_matrix_csv_shape")


# ===========================================================================
# 10. Output completeness
# ===========================================================================

def test_output_row_counts():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        for fname, expected in [("primary_items.csv", 25), ("all_appearances.csv", 31),
                                ("method_summary.csv", 5), ("duplicate_consistency.csv", 7),
                                ("confusion_matrix.csv", 6)]:
            with open(out / fname) as f:
                assert len(f.readlines()) == expected, f"{fname} wrong count"
    print("  PASS: test_output_row_counts")

def test_summary_protocol_limitations():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        with open(out / "summary.json") as f:
            s = json.load(f)
        lim = s["limitations"]
        assert lim == {"question_stratification": None,
                       "display_context_provenance": "synthetic_source_context",
                       "is_exact_model_visible_context": False,
                       "primary_judge_dimension": "integration"}
    print("  PASS: test_summary_protocol_limitations")

def test_deterministic_outputs():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        o1 = Path(d) / "c1"; o2 = Path(d) / "c2"
        write_compilation_outputs(result, output_dir=o1, run_id="t", rater_id="r", protocol_name="p")
        write_compilation_outputs(result, output_dir=o2, run_id="t", rater_id="r", protocol_name="p")
        for fname in ("summary.json", "primary_items.csv", "confusion_matrix.csv"):
            assert (o1/fname).read_text() == (o2/fname).read_text(), f"{fname} not deterministic"
    print("  PASS: test_deterministic_outputs")


# ===========================================================================
# 11. Atomic publication
# ===========================================================================

def test_failure_leaves_no_output():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        with patch("benchmarks.human_rating.compile_ratings.os.rename", side_effect=OSError("fail")):
            try:
                write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
            except OSError:
                pass
        assert not out.exists()
        # No staging dirs remain
        staging = [p for p in Path(d).iterdir() if ".compile_staging" in p.name]
        assert staging == []
    print("  PASS: test_failure_leaves_no_output")

def test_overwrite_protection():
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        try:
            write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
            assert False
        except FileExistsError:
            pass
    print("  PASS: test_overwrite_protection")

def test_compiled_permissions_posix():
    if os.name != "posix":
        print("  SKIP: test_compiled_permissions_posix")
        return
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        for f in out.iterdir():
            assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    print("  PASS: test_compiled_permissions_posix")


# ===========================================================================
# 12. CLI
# ===========================================================================

def test_compile_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.human_rating.compile_ratings", "--help"],
        capture_output=True, text=True, cwd=_project_root,
    )
    assert result.returncode == 0
    assert "--queue" in result.stdout
    assert "--answer-key" in result.stdout
    print("  PASS: test_compile_cli_help")

def test_compile_cli_success():
    with tempfile.TemporaryDirectory() as d:
        paths = _build_run(d)
        out = Path(d) / "compiled"
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.compile_ratings",
             "--queue", str(paths[0]), "--ratings", str(paths[1]),
             "--session", str(paths[2]), "--answer-key", str(paths[3]),
             "--output-dir", str(out)],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert result.returncode == 0
        assert "Compilation complete" in result.stdout
        assert (out / "summary.json").exists()
    print("  PASS: test_compile_cli_success")


# ===========================================================================
# 13. Real artifact integration
# ===========================================================================

def test_real_artifacts_compile():
    """Generate real artifacts and compile with synthetic ratings."""
    results_dir = os.path.join(_project_root, "results")
    if not os.path.isdir(os.path.join(results_dir, "gpt4o_naive_concat")):
        print("  SKIP: test_real_artifacts_compile")
        return
    from benchmarks.human_rating.trial_loader import load_eligible_trials
    from benchmarks.human_rating.sampling import sample_unique_trials
    from benchmarks.human_rating.blinding import build_blinded_plan
    from benchmarks.human_rating.artifact_writer import write_rating_artifacts
    from benchmarks.human_rating.paths import get_run_paths

    with tempfile.TemporaryDirectory() as d:
        trials, _ = load_eligible_trials(results_root=results_dir, min_per_method=6)
        selected, _ = sample_unique_trials(trials, seed=20260715)
        plan = build_blinded_plan(selected, seed=20260716)
        rp = get_run_paths("real-test", base_dir=d)
        write_rating_artifacts(plan, run_paths=rp, run_id="real-test",
                               protocol_name="test", sampling_seed=20260715, blinding_seed=20260716)
        # Create synthetic complete ratings
        with open(rp.rating_queue_path) as f:
            queue = json.load(f)
        session = {"schema_version": 1, "run_id": "real-test", "rater_id": "rater-01",
                   "queue_fingerprint": _queue_fingerprint(queue),
                   "queue_item_count": 30, "ratings_file": "ratings.jsonl",
                   "started_at": "t"}
        session_path = Path(rp.rater_dir) / "rating_session.json"
        session_path.write_text(json.dumps(session))
        ratings_path = Path(rp.ratings_path)
        with open(ratings_path, "w") as f:
            for item in queue["items"]:
                f.write(json.dumps({"schema_version": 1, "run_id": "real-test",
                        "rater_id": "rater-01", "blinded_id": item["blinded_id"],
                        "rating": 3, "note": None, "flagged": False, "rated_at": "t"}) + "\n")
        # Compile
        result = compile_rating_run(
            queue_path=Path(rp.rating_queue_path),
            ratings_path=ratings_path,
            session_path=session_path,
            answer_key_path=Path(rp.answer_key_path),
        )
        assert len(result.primary_items) == 24
        assert len(result.all_appearances) == 30
        assert len(result.method_summaries) == 4
        assert len(result.duplicate_pairs) == 6
    print("  PASS: test_real_artifacts_compile")


# ===========================================================================
# Additional: Method summaries, duplicate details, privacy, rollback, CLI
# ===========================================================================

def test_method_summary_all_fields_known_values():
    """Each method has different ratings so grouping bugs are visible."""
    # Methods assigned round-robin: i%4 → ks=0,4,8,12,16,20 m0=1,5,9,13,17,21 nc=2,6,10,14,18,22 vr=3,7,11,15,19,23
    # kernel_shared (indices 0,4,8,12,16,20): human=[1,2,3,4,5,5], judge=3
    # mem0_default (indices 1,5,9,13,17,21): human=[2,2,2,2,2,2], judge=3
    # naive_concat (indices 2,6,10,14,18,22): human=[5,5,5,5,5,5], judge=3
    # vanilla_rag (indices 3,7,11,15,19,23): human=[3,4,3,4,3,4], judge=3
    import statistics as st
    h = [0]*24
    ks_h = [1,2,3,4,5,5]; m0_h = [2,2,2,2,2,2]; nc_h = [5,5,5,5,5,5]; vr_h = [3,4,3,4,3,4]
    for i, v in enumerate(ks_h): h[i*4] = v
    for i, v in enumerate(m0_h): h[i*4+1] = v
    for i, v in enumerate(nc_h): h[i*4+2] = v
    for i, v in enumerate(vr_h): h[i*4+3] = v
    h += [3]*6
    j = [3]*30
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d, human_ratings=h, judge_scores=j)

    ks = next(ms for ms in result.method_summaries if ms.method == "kernel_shared")
    assert ks.item_count == 6
    assert abs(ks.mean_human_rating - st.mean(ks_h)) < 0.001
    assert ks.median_human_rating == float(st.median(ks_h))
    assert abs(ks.human_rating_std - st.stdev(ks_h)) < 0.001
    assert ks.mean_judge_score == 3.0
    diffs = [v-3 for v in ks_h]
    assert abs(ks.mean_signed_difference - st.mean(diffs)) < 0.001
    assert abs(ks.mean_absolute_error - st.mean([abs(d) for d in diffs])) < 0.001
    exact = sum(1 for v in ks_h if v == 3)
    assert abs(ks.exact_agreement_rate - exact/6) < 0.001
    w1 = sum(1 for v in ks_h if abs(v-3) <= 1)
    assert abs(ks.within_one_rate - w1/6) < 0.001

    nc = next(ms for ms in result.method_summaries if ms.method == "naive_concat")
    assert nc.mean_human_rating == 5.0
    assert nc.exact_agreement_rate == 0.0  # all 5 vs judge 3
    assert nc.within_one_rate == 0.0

    m0 = next(ms for ms in result.method_summaries if ms.method == "mem0_default")
    assert m0.mean_human_rating == 2.0
    assert m0.within_one_rate == 1.0  # all |2-3|=1

    vr = next(ms for ms in result.method_summaries if ms.method == "vanilla_rag")
    assert abs(vr.mean_human_rating - st.mean(vr_h)) < 0.001
    print("  PASS: test_method_summary_all_fields_known_values")


def test_duplicate_summary_all_fields_known():
    """Known dup ratings: originals[0-5]=[1,2,3,4,5,3], dups=[3,3,3,3,3,3]."""
    h = [1,2,3,4,5,3] + [3]*18 + [3,3,3,3,3,3]  # dups at 24-29
    j = [3]*30
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d, human_ratings=h, judge_scores=j)
    ds = result.duplicate_summary
    # Pairs: (1,3),(2,3),(3,3),(4,3),(5,3),(3,3)
    # Diffs: 2,1,0,-1,-2,0; abs: 2,1,0,1,2,0
    assert ds.pair_count == 6
    assert ds.exact_match_count == 2  # items 2,5 (3==3)
    assert abs(ds.exact_match_rate - 2/6) < 0.001
    w1 = sum(1 for ad in [2,1,0,1,2,0] if ad <= 1)  # 1,0,1,0 → 4
    assert ds.within_one_count == w1
    assert abs(ds.within_one_rate - w1/6) < 0.001
    assert abs(ds.mean_absolute_difference - sum([2,1,0,1,2,0])/6) < 0.001
    assert ds.max_absolute_difference == 2
    # Positions: originals at 1-6, dups at 25-30; distances = 24 each
    assert abs(ds.mean_positional_distance - 24.0) < 0.001
    # Verify individual records
    for dp in result.duplicate_pairs:
        assert dp.original_queue_position <= 6
        assert dp.duplicate_queue_position >= 25
    print("  PASS: test_duplicate_summary_all_fields_known")


def test_confusion_matrix_integer_cells_and_known():
    """Multiple nonzero cells verified."""
    # human=[5,4,3,2,1]*4 + [3,3,3,3], judge=3 for all
    h_p = [5,4,3,2,1]*4 + [3,3,3,3]  # 24 items
    h = h_p + [3]*6
    j = [3]*30
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d, human_ratings=h, judge_scores=j)
    m = result.confusion_matrix
    total = sum(m[hr][js] for hr in range(1,6) for js in range(1,6))
    assert total == 24
    # All judge=3, so only column 3 has values
    for hr in range(1, 6):
        for js in range(1, 6):
            assert isinstance(m[hr][js], int)
            if js != 3:
                assert m[hr][js] == 0
    # Column 3: counts of each human rating
    # 5 appears 4 times, 4 appears 4, 3 appears 8 (4+4 from extra), 2 appears 4, 1 appears 4
    assert m[5][3] == 4
    assert m[4][3] == 4
    assert m[3][3] == 8
    assert m[2][3] == 4
    assert m[1][3] == 4
    print("  PASS: test_confusion_matrix_integer_cells_and_known")


def test_output_headers():
    """CSV files have expected headers."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        with open(out / "primary_items.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert "blinded_id" in headers
        assert "human_rating" in headers
        assert "judge_score" in headers
        assert "source_method" in headers
        with open(out / "confusion_matrix.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["human\\judge", "1", "2", "3", "4", "5"]
    print("  PASS: test_output_headers")


def test_notes_not_in_summary_json():
    """summary.json must not contain note text."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d, note_0="PRIVATE-NOTE-SENTINEL-9274")
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        text = (out / "summary.json").read_text()
        assert "PRIVATE-NOTE-SENTINEL" not in text
    print("  PASS: test_notes_not_in_summary_json")


def test_compile_cli_does_not_print_notes():
    """CLI terminal output must not contain note text."""
    with tempfile.TemporaryDirectory() as d:
        paths = _build_run(d, note_5="PRIVATE-NOTE-CLI-CHECK")
        out = Path(d) / "compiled"
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.human_rating.compile_ratings",
             "--queue", str(paths[0]), "--ratings", str(paths[1]),
             "--session", str(paths[2]), "--answer-key", str(paths[3]),
             "--output-dir", str(out)],
            capture_output=True, text=True, cwd=_project_root,
        )
        assert "PRIVATE-NOTE-CLI-CHECK" not in result.stdout
        assert "PRIVATE-NOTE-CLI-CHECK" not in result.stderr
    print("  PASS: test_compile_cli_does_not_print_notes")


def test_failed_overwrite_restores_existing():
    """Failed overwrite restores original compiled directory."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        # Save original bytes
        original_files = {}
        for f in out.iterdir():
            original_files[f.name] = f.read_bytes()

        # Inject failure during overwrite (after backup, before new publish)
        orig_rename = os.rename
        call_count = [0]
        def failing_rename(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:  # Second rename = staging→final
                raise OSError("Injected failure")
            return orig_rename(src, dst)

        with patch("benchmarks.human_rating.compile_ratings.os.rename", side_effect=failing_rename):
            try:
                write_compilation_outputs(result, output_dir=out, run_id="t",
                                          rater_id="r", protocol_name="p", overwrite=True)
            except OSError:
                pass

        # Original directory restored
        assert out.exists()
        for name, content in original_files.items():
            assert (out / name).read_bytes() == content
        # No backup or staging remains
        leftovers = [p for p in Path(d).iterdir()
                     if "backup" in p.name or "staging" in p.name]
        assert leftovers == []
    print("  PASS: test_failed_overwrite_restores_existing")


def test_no_staging_after_success():
    """No staging or backup directories after successful write."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        leftovers = [p for p in Path(d).iterdir()
                     if "staging" in p.name or "backup" in p.name]
        assert leftovers == []
    print("  PASS: test_no_staging_after_success")


def test_compile_cli_requires_all_args():
    """Missing required args fail."""
    base_args = [sys.executable, "-m", "benchmarks.human_rating.compile_ratings"]
    for missing in ["--queue", "--ratings", "--session", "--answer-key", "--output-dir"]:
        # Build args with one missing
        all_flags = {"--queue": "q", "--ratings": "r", "--session": "s",
                     "--answer-key": "k", "--output-dir": "o"}
        args = list(base_args)
        for flag, val in all_flags.items():
            if flag != missing:
                args.extend([flag, val])
        result = subprocess.run(args, capture_output=True, text=True, cwd=_project_root)
        assert result.returncode != 0, f"Should fail without {missing}"
    print("  PASS: test_compile_cli_requires_all_args")


def test_compile_cli_overwrite_flag():
    """--overwrite allows re-compilation."""
    with tempfile.TemporaryDirectory() as d:
        paths = _build_run(d)
        out = Path(d) / "compiled"
        base = [sys.executable, "-m", "benchmarks.human_rating.compile_ratings",
                "--queue", str(paths[0]), "--ratings", str(paths[1]),
                "--session", str(paths[2]), "--answer-key", str(paths[3]),
                "--output-dir", str(out)]
        # First run
        r1 = subprocess.run(base, capture_output=True, text=True, cwd=_project_root)
        assert r1.returncode == 0
        # Second without overwrite → fail
        r2 = subprocess.run(base, capture_output=True, text=True, cwd=_project_root)
        assert r2.returncode != 0
        # Third with --overwrite → success
        r3 = subprocess.run(base + ["--overwrite"], capture_output=True, text=True, cwd=_project_root)
        assert r3.returncode == 0
    print("  PASS: test_compile_cli_overwrite_flag")


def test_method_primary_count_violation_rejected():
    """Uneven primary method distribution is rejected."""
    with tempfile.TemporaryDirectory() as d:
        def m(paths):
            k = json.loads(paths[3].read_text())
            # Change item 7's method from vanilla_rag to naive_concat
            # (item 7 is not referenced by any duplicate, safe to modify)
            # This makes naive_concat=7, vanilla_rag=5
            k["items"][7]["source_method"] = "naive_concat"
            paths[3].write_text(json.dumps(k))
        _expect_error(d, m, "6")
    print("  PASS: test_method_primary_count_violation_rejected")


def test_sensitivity_method_summaries_include_duplicates():
    """Appearance method summaries include duplicate appearances."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
    # Appearance method counts should total 30
    total_appearances = sum(ms.item_count for ms in result.appearance_method_summaries)
    assert total_appearances == 30
    # Primary counts total 24
    total_primary = sum(ms.item_count for ms in result.method_summaries)
    assert total_primary == 24
    # Each primary method has 6
    for ms in result.method_summaries:
        assert ms.item_count == 6
    # Appearance counts are >= 6 for each method (6 + duplicates)
    for ms in result.appearance_method_summaries:
        assert ms.item_count >= 6
    print("  PASS: test_sensitivity_method_summaries_include_duplicates")


def test_duplicate_ratings_affect_sensitivity_not_primary():
    """Changing duplicate ratings changes sensitivity but not primary method summaries."""
    j = [3] * 30
    h1 = [3] * 24 + [3] * 6  # All match
    h2 = [3] * 24 + [1] * 6  # Dups differ
    with tempfile.TemporaryDirectory() as d1:
        r1 = _compile(d1, human_ratings=h1, judge_scores=j)
    with tempfile.TemporaryDirectory() as d2:
        r2 = _compile(d2, human_ratings=h2, judge_scores=j)

    # Primary method summaries unchanged
    for ms1, ms2 in zip(r1.method_summaries, r2.method_summaries):
        assert ms1 == ms2

    # Overall primary unchanged
    assert r1.overall_summary == r2.overall_summary

    # Appearance method summaries differ (at least for methods with duplicates)
    assert r1.appearance_method_summaries != r2.appearance_method_summaries
    print("  PASS: test_duplicate_ratings_affect_sensitivity_not_primary")


def test_summary_json_has_both_method_lists():
    """summary.json contains both primary and appearance method summaries."""
    with tempfile.TemporaryDirectory() as d:
        result = _compile(d)
        out = Path(d) / "compiled"
        write_compilation_outputs(result, output_dir=out, run_id="t", rater_id="r", protocol_name="p")
        with open(out / "summary.json") as f:
            s = json.load(f)
        assert "primary_method_summaries" in s
        assert "appearance_method_summaries" in s
        assert len(s["primary_method_summaries"]) == 4
        assert len(s["appearance_method_summaries"]) == 4
        # Primary counts all 6
        for ms in s["primary_method_summaries"]:
            assert ms["item_count"] == 6
        # Appearance counts total 30
        assert sum(ms["item_count"] for ms in s["appearance_method_summaries"]) == 30
    print("  PASS: test_summary_json_has_both_method_lists")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=== Compile Ratings Tests ===\n")

    print("Malformed ratings:")
    test_malformed_jsonl_rejected()
    test_non_object_jsonl_record_rejected()
    test_malformed_trailing_jsonl_rejected()

    print("\nSession metadata:")
    test_session_item_count_mismatch_rejected()
    test_session_ratings_filename_mismatch_rejected()
    test_session_missing_required_field_rejected()

    print("\nAnswer-key structural:")
    test_answer_key_order_mismatch_rejected()
    test_answer_key_position_mismatch_rejected()
    test_broken_duplicate_reference_rejected()
    test_duplicate_points_to_duplicate_rejected()
    test_duplicate_source_metadata_mismatch_rejected()
    test_duplicate_queue_content_mismatch_rejected()

    print("\nProtocol constraints:")
    test_wrong_evaluation_dimension_rejected()
    test_wrong_context_provenance_rejected()
    test_exact_model_visible_context_true_rejected()

    print("\nMetrics (known values):")
    test_overall_summary_all_fields()
    test_primary_metrics_unchanged_when_duplicates_change()
    test_method_flag_and_note_counts()
    test_method_summary_all_fields_known_values()

    print("\nDuplicate/sensitivity:")
    test_duplicate_positions_known()
    test_duplicate_metrics_ignore_judge_scores()
    test_duplicate_summary_all_fields_known()
    test_confusion_matrix_labels_and_shape()
    test_confusion_matrix_csv_shape()
    test_confusion_matrix_integer_cells_and_known()

    print("\nOutputs/privacy:")
    test_output_row_counts()
    test_summary_protocol_limitations()
    test_deterministic_outputs()
    test_output_headers()
    test_notes_not_in_summary_json()
    test_compile_cli_does_not_print_notes()

    print("\nMethod population integrity:")
    test_method_primary_count_violation_rejected()
    test_sensitivity_method_summaries_include_duplicates()
    test_duplicate_ratings_affect_sensitivity_not_primary()
    test_summary_json_has_both_method_lists()

    print("\nAtomicity/permissions:")
    test_failure_leaves_no_output()
    test_overwrite_protection()
    test_compiled_permissions_posix()
    test_failed_overwrite_restores_existing()
    test_no_staging_after_success()

    print("\nCLI:")
    test_compile_cli_help()
    test_compile_cli_success()
    test_compile_cli_requires_all_args()
    test_compile_cli_overwrite_flag()

    print("\nIntegration:")
    test_real_artifacts_compile()

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
