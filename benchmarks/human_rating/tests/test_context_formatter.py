"""Tests for the deterministic reference-context formatter.

Run:
    python benchmarks/human_rating/tests/test_context_formatter.py
"""

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.human_rating.context_formatter import format_reference_context


def test_full_context():
    """Full profile + task context formats correctly."""
    profile = {
        "user_name": "Alice Smith",
        "preferred_tools": ["VS Code", "Git", "Docker"],
        "preferred_language": "Python",
        "response_style": "concise",
    }
    task = {
        "current_project": "Recommendation Engine",
        "active_experiment": "Testing collaborative filtering",
        "goals": ["Improve accuracy", "Reduce latency"],
        "blockers": ["Limited cold-start data"],
        "next_steps": ["Profile query performance", "Run A/B test"],
    }
    result = format_reference_context(profile, task)
    assert "USER PROFILE" in result
    assert "Name: Alice Smith" in result
    assert "VS Code, Git, Docker" in result
    assert "Python" in result
    assert "concise" in result
    assert "TASK CONTEXT" in result
    assert "Recommendation Engine" in result
    assert "Testing collaborative filtering" in result
    assert "- Improve accuracy" in result
    assert "- Limited cold-start data" in result
    assert "- Run A/B test" in result
    print("  PASS: test_full_context")


def test_deterministic():
    """Same input produces identical output."""
    profile = {"user_name": "Bob", "preferred_tools": ["Git"], "preferred_language": "Rust", "response_style": "detailed"}
    task = {"current_project": "P", "active_experiment": "E", "goals": ["G"], "blockers": [], "next_steps": ["S"]}
    r1 = format_reference_context(profile, task)
    r2 = format_reference_context(profile, task)
    assert r1 == r2
    print("  PASS: test_deterministic")


def test_empty_lists():
    """Empty blockers/goals/next_steps handled gracefully."""
    profile = {"user_name": "Eve", "preferred_tools": [], "preferred_language": "Go", "response_style": "casual"}
    task = {"current_project": "API", "active_experiment": "Load testing", "goals": [], "blockers": [], "next_steps": []}
    result = format_reference_context(profile, task)
    assert "(none)" in result  # empty tools
    assert "Goals:" not in result  # empty goals not shown
    assert "Blockers:" not in result
    assert "Next steps:" not in result
    print("  PASS: test_empty_lists")


def test_missing_profile():
    """None profile with valid task still formats."""
    task = {"current_project": "X", "active_experiment": "Y", "goals": ["G"], "blockers": [], "next_steps": ["S"]}
    result = format_reference_context(None, task)
    assert "USER PROFILE" not in result
    assert "TASK CONTEXT" in result
    assert "X" in result
    print("  PASS: test_missing_profile")


def test_missing_task():
    """None task with valid profile still formats."""
    profile = {"user_name": "Sam", "preferred_tools": ["Vim"], "preferred_language": "C", "response_style": "formal"}
    result = format_reference_context(profile, None)
    assert "USER PROFILE" in result
    assert "TASK CONTEXT" not in result
    assert "Sam" in result
    print("  PASS: test_missing_task")


def test_both_none_raises():
    """Both None raises ValueError."""
    try:
        format_reference_context(None, None)
        assert False
    except ValueError as e:
        assert "None" in str(e)
    print("  PASS: test_both_none_raises")


def test_preserves_list_ordering():
    """Goals/blockers/next_steps maintain insertion order."""
    task = {"current_project": "P", "active_experiment": "E",
            "goals": ["First", "Second", "Third"], "blockers": ["B1", "B2"], "next_steps": ["S1"]}
    result = format_reference_context(None, task)
    first_idx = result.index("First")
    second_idx = result.index("Second")
    third_idx = result.index("Third")
    assert first_idx < second_idx < third_idx
    print("  PASS: test_preserves_list_ordering")


def main():
    print("=== Context Formatter Tests ===\n")
    test_full_context()
    test_deterministic()
    test_empty_lists()
    test_missing_profile()
    test_missing_task()
    test_both_none_raises()
    test_preserves_list_ordering()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
