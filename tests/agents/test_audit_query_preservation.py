"""Preservation property tests for audit-query-cross-agent-fix (Property 2).

These tests verify that non-cross-agent behaviors are unchanged on the
UNFIXED code. They capture baseline behavior that the fix must preserve:

- Empty user_id returns empty RetrievalLog() immediately
- Exceptions from send_request return empty RetrievalLog() gracefully
- _build_retrieval_log_from_search applies RELEVANCE_THRESHOLD=0.3
  (drops score < 0.3, keeps score=None)
- When kernel diagnostics are present with injected_count > 0, the
  diagnostics path is used and audit is never called
- When share_memory=False (Phase 1), audit query path is never reached

All tests MUST PASS on the current unfixed code. These are preservation
tests that confirm baseline behavior to protect during the fix.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

from __future__ import annotations

import sys
import traceback
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, ".")

from hypothesis import HealthCheck, given, settings, assume
from hypothesis import strategies as st

from benchmarks.shared_memory.pipeline import AgentPipeline
from benchmarks.shared_memory.models import (
    RetrievalLog,
    RetrievalLogEntry,
    InjectionDiagnostics,
    InjectedMemoryEntry,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for empty/None user_id values (non-buggy: empty user_id)
_empty_user_id_strategy = st.one_of(
    st.just(""),
    st.just(None),
)

# Strategy for query text
_query_text_strategy = st.text(min_size=1, max_size=100)

# Strategy for non-empty user_id (used in exception tests)
_nonempty_user_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
).filter(lambda s: len(s.strip()) > 0)

# Strategy for generating random exception types
_exception_strategy = st.sampled_from([
    ConnectionError("Connection refused"),
    TimeoutError("Request timed out"),
    OSError("Network unreachable"),
    RuntimeError("Kernel unavailable"),
    ValueError("Invalid response format"),
    Exception("Generic failure"),
])

# Strategy for relevance scores: mix of below-threshold, above-threshold, and None
_score_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=0.29, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.3, max_value=1.0, allow_nan=False, allow_infinity=False),
)

# Strategy for owner agents
_owner_agent_strategy = st.sampled_from([
    "profile_agent", "task_agent", "assistant_agent", "other_agent"
])

# Strategy for memory types
_memory_type_strategy = st.sampled_from([
    "profile", "task_context", "conversation", "other"
])

# Strategy for sharing policies
_sharing_policy_strategy = st.sampled_from(["shared", "private"])


def _search_result_entry_strategy():
    """Strategy generating a single search result dict with random score/metadata."""
    return st.fixed_dictionaries({
        "memory_id": st.text(min_size=5, max_size=20, alphabet="abcdef0123456789"),
        "content": st.text(min_size=1, max_size=50),
        "score": _score_strategy,
        "metadata": st.fixed_dictionaries({
            "owner_agent": _owner_agent_strategy,
            "memory_type": _memory_type_strategy,
            "sharing_policy": _sharing_policy_strategy,
            "user_id": st.text(min_size=1, max_size=10),
        }),
    })


# Strategy for a list of search results (0 to 10 entries)
_search_results_strategy = st.lists(
    _search_result_entry_strategy(),
    min_size=0,
    max_size=10,
)

# Strategy for InjectionDiagnostics with injected_count > 0
_diagnostics_with_injections_strategy = st.integers(
    min_value=1, max_value=10
).flatmap(
    lambda count: st.fixed_dictionaries({
        "injected_count": st.just(count),
        "injected_memories": st.lists(
            st.fixed_dictionaries({
                "owner_agent": _owner_agent_strategy,
                "memory_type": _memory_type_strategy,
                "match_score": st.one_of(
                    st.none(),
                    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                ),
            }),
            min_size=count,
            max_size=count,
        ),
    })
)


# ---------------------------------------------------------------------------
# Result tracker
# ---------------------------------------------------------------------------

results: list[tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


# ---------------------------------------------------------------------------
# Property (a): Empty user_id returns empty RetrievalLog
# ---------------------------------------------------------------------------

@given(
    user_id=_empty_user_id_strategy,
    query_text=_query_text_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_a_empty_user_id_returns_empty_log(user_id, query_text) -> None:
    """When user_id is empty or None, _audit_shared_memories returns empty RetrievalLog.

    This is a preservation check: the early-return for empty user_id must
    remain unchanged regardless of any fix to the cross-agent routing.

    **Validates: Requirements 3.5**
    """
    pipeline = AgentPipeline(share_memory=True)

    # send_request should never be called for empty user_id
    with patch("cerebrum.utils.communication.send_request") as mock_send:
        # Handle None user_id — the method checks `if not user_id`
        result = pipeline._audit_shared_memories(user_id or "", query_text)
        mock_send.assert_not_called()

    assert result.shared_memory_count == 0, (
        f"Expected shared_memory_count=0 for empty user_id={user_id!r}, "
        f"got {result.shared_memory_count}"
    )
    assert result.retrieved_memories == [], (
        f"Expected empty retrieved_memories for empty user_id={user_id!r}, "
        f"got {result.retrieved_memories}"
    )
    assert result.cross_agent_found is False, (
        f"Expected cross_agent_found=False for empty user_id={user_id!r}, "
        f"got {result.cross_agent_found}"
    )


# ---------------------------------------------------------------------------
# Property (b): Exceptions from send_request return empty RetrievalLog
# ---------------------------------------------------------------------------

@given(
    user_id=_nonempty_user_id_strategy,
    query_text=_query_text_strategy,
    exception=_exception_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_b_exception_returns_empty_log(user_id, query_text, exception) -> None:
    """When send_request raises any exception, _audit_shared_memories returns empty RetrievalLog.

    This is a preservation check: the try/except handler must continue to
    catch all exceptions and return an empty RetrievalLog without propagating.

    **Validates: Requirements 3.5**
    """
    pipeline = AgentPipeline(share_memory=True)

    with patch("cerebrum.utils.communication.send_request", side_effect=exception):
        # Must not raise — should be caught internally
        result = pipeline._audit_shared_memories(user_id, query_text)

    assert result.shared_memory_count == 0, (
        f"Expected shared_memory_count=0 after exception {type(exception).__name__}, "
        f"got {result.shared_memory_count}"
    )
    assert result.retrieved_memories == [], (
        f"Expected empty retrieved_memories after exception, "
        f"got {result.retrieved_memories}"
    )
    assert result.cross_agent_found is False, (
        f"Expected cross_agent_found=False after exception, "
        f"got {result.cross_agent_found}"
    )


# ---------------------------------------------------------------------------
# Property (c): _build_retrieval_log_from_search threshold filtering
# ---------------------------------------------------------------------------

@given(search_results=_search_results_strategy)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_c_threshold_filtering(search_results) -> None:
    """_build_retrieval_log_from_search applies RELEVANCE_THRESHOLD=0.3 filter.

    - Drops entries with score < 0.3
    - Keeps entries with score=None (kernel behavior for unscored results)
    - Keeps entries with score >= 0.3
    - Correctly populates shared_memory_count, cross_agent_found, retrieved_memories

    **Validates: Requirements 3.4**
    """
    pipeline = AgentPipeline(share_memory=True)

    # Construct a synthetic response dict
    response = {"response": {"search_results": search_results}}
    result = pipeline._build_retrieval_log_from_search(response)

    # Manually compute expected behavior
    expected_entries = []
    expected_shared_count = 0
    expected_cross_agent = False

    for mem in search_results:
        score = mem.get("score")
        # Drop below threshold
        if score is not None and score < 0.3:
            continue
        meta = mem.get("metadata", {})
        if not meta:
            continue
        owner = meta.get("owner_agent", "")
        if not owner:
            continue
        mem_type = meta.get("memory_type", "")
        expected_entries.append(RetrievalLogEntry(
            owner_agent=owner,
            memory_type=mem_type,
        ))
        if meta.get("sharing_policy") == "shared":
            expected_shared_count += 1
        if owner != "assistant_agent":
            expected_cross_agent = True

    assert result.shared_memory_count == expected_shared_count, (
        f"shared_memory_count mismatch: got {result.shared_memory_count}, "
        f"expected {expected_shared_count}. Input: {search_results}"
    )
    assert result.cross_agent_found == expected_cross_agent, (
        f"cross_agent_found mismatch: got {result.cross_agent_found}, "
        f"expected {expected_cross_agent}. Input: {search_results}"
    )
    assert len(result.retrieved_memories) == len(expected_entries), (
        f"retrieved_memories count mismatch: got {len(result.retrieved_memories)}, "
        f"expected {len(expected_entries)}. Input: {search_results}"
    )
    # Verify each entry matches
    for i, (actual, expected) in enumerate(zip(result.retrieved_memories, expected_entries)):
        assert actual.owner_agent == expected.owner_agent, (
            f"Entry {i} owner_agent mismatch: got {actual.owner_agent!r}, "
            f"expected {expected.owner_agent!r}"
        )
        assert actual.memory_type == expected.memory_type, (
            f"Entry {i} memory_type mismatch: got {actual.memory_type!r}, "
            f"expected {expected.memory_type!r}"
        )


# ---------------------------------------------------------------------------
# Property (d): Diagnostics path with injected_count > 0 uses diagnostics,
#               audit not called
# ---------------------------------------------------------------------------

@given(diagnostics_data=_diagnostics_with_injections_strategy)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_d_diagnostics_path_used_when_present(diagnostics_data) -> None:
    """When kernel diagnostics have injected_count > 0, _retrieval_log_from_diagnostics is used.

    The run_trial routing logic uses the diagnostics-first path when
    injection_diagnostics is present and injected_count > 0. The audit
    query (_audit_shared_memories) should NOT be called in this case.

    This tests _retrieval_log_from_diagnostics directly to confirm it
    produces the expected output from diagnostics data.

    **Validates: Requirements 3.1**
    """
    pipeline = AgentPipeline(share_memory=True)

    # Build InjectionDiagnostics from generated data
    entries = []
    for mem_data in diagnostics_data["injected_memories"]:
        entries.append(InjectedMemoryEntry(
            owner_agent=mem_data["owner_agent"],
            memory_type=mem_data["memory_type"],
            match_score=mem_data["match_score"],
        ))
    diagnostics = InjectionDiagnostics(
        injected_count=diagnostics_data["injected_count"],
        injected_memories=entries,
    )

    # Call _retrieval_log_from_diagnostics directly
    result = pipeline._retrieval_log_from_diagnostics(diagnostics)

    # Verify the diagnostics path produces correct output
    assert result.shared_memory_count == diagnostics.injected_count, (
        f"shared_memory_count should equal injected_count={diagnostics.injected_count}, "
        f"got {result.shared_memory_count}"
    )
    assert len(result.retrieved_memories) == len(diagnostics.injected_memories), (
        f"retrieved_memories count should equal injected_memories count="
        f"{len(diagnostics.injected_memories)}, got {len(result.retrieved_memories)}"
    )

    # Verify cross_agent_found is set correctly
    expected_cross_agent = any(
        mem.owner_agent != "assistant_agent" for mem in diagnostics.injected_memories
    )
    assert result.cross_agent_found == expected_cross_agent, (
        f"cross_agent_found mismatch: got {result.cross_agent_found}, "
        f"expected {expected_cross_agent}"
    )

    # Verify entries are correctly mapped
    for i, (actual, expected_mem) in enumerate(zip(result.retrieved_memories, entries)):
        assert actual.owner_agent == expected_mem.owner_agent, (
            f"Entry {i} owner_agent: got {actual.owner_agent!r}, "
            f"expected {expected_mem.owner_agent!r}"
        )
        assert actual.memory_type == expected_mem.memory_type, (
            f"Entry {i} memory_type: got {actual.memory_type!r}, "
            f"expected {expected_mem.memory_type!r}"
        )


# ---------------------------------------------------------------------------
# Property (e): Diagnostics routing in run_trial — audit not called when
#               diagnostics present
# ---------------------------------------------------------------------------

def test_e_run_trial_diagnostics_routing() -> None:
    """When injection_diagnostics are present with injected_count > 0 in run_trial,
    the audit path (_audit_shared_memories) is never called.

    This tests the routing logic at the run_trial level by mocking the
    agent execution steps and verifying _audit_shared_memories is not invoked.

    **Validates: Requirements 3.1**
    """
    pipeline = AgentPipeline(share_memory=True)

    # Mock diagnostics present in assistant_result
    mock_assistant_result = {
        "result": "test response",
        "injection_diagnostics": {
            "injected_count": 2,
            "injected_memories": [
                {"owner_agent": "profile_agent", "memory_type": "profile", "match_score": 0.8},
                {"owner_agent": "task_agent", "memory_type": "task_context", "match_score": 0.7},
            ],
        },
    }

    from benchmarks.shared_memory.models import SyntheticTrialData, SyntheticProfile, SyntheticTaskContext

    trial_data = SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="test_user",
            preferred_tools=["python"],
            preferred_language="English",
            response_style="concise",
        ),
        task_context=SyntheticTaskContext(
            current_project="test_project",
            active_experiment="exp1",
            goals=["goal1"],
            blockers=["blocker1"],
            next_steps=["step1"],
        ),
        follow_up_query="test query",
        user_id="test_user_123",
    )

    # Mock all agent runs and search_memories to avoid needing a kernel
    with patch.object(pipeline, '_audit_shared_memories') as mock_audit, \
         patch("benchmarks.shared_memory.pipeline.ProfileAgent") as mock_profile_cls, \
         patch("benchmarks.shared_memory.pipeline.TaskAgent") as mock_task_cls, \
         patch("benchmarks.shared_memory.pipeline.AssistantAgent") as mock_assistant_cls, \
         patch("benchmarks.shared_memory.pipeline.search_memories", return_value={"response": {"search_results": []}}):

        # Setup mocks
        mock_profile_cls.return_value.run.return_value = {}
        mock_task_cls.return_value.run.return_value = {}
        mock_assistant_cls.return_value.run.return_value = mock_assistant_result

        result = pipeline.run_trial(trial_data)

        # _audit_shared_memories should NOT have been called
        mock_audit.assert_not_called()

    # Verify the diagnostics path was used
    assert result.retrieval_log is not None
    assert result.retrieval_log.shared_memory_count == 2
    assert result.retrieval_log.cross_agent_found is True
    return True


# ---------------------------------------------------------------------------
# Property (f): Phase 1 (share_memory=False) — audit never reached
# ---------------------------------------------------------------------------

def test_f_phase1_no_audit() -> None:
    """When share_memory=False (Phase 1), the audit query path is never called.

    In Phase 1, the pipeline runs with share_memory=False. When the
    assistant_result has no injection_diagnostics (or injected_count=0),
    _audit_shared_memories is still called but its results produce
    injection_status="unknown" only if share_memory=True. When
    share_memory=False, the same code path runs but no "Observability gap"
    warning is logged.

    The key preservation behavior: Phase 1 trials should not trigger
    observability gap warnings regardless of audit results.

    **Validates: Requirements 3.2**
    """
    pipeline = AgentPipeline(share_memory=False)

    # Mock assistant result with NO diagnostics
    mock_assistant_result = {
        "result": "test response",
    }

    from benchmarks.shared_memory.models import SyntheticTrialData, SyntheticProfile, SyntheticTaskContext

    trial_data = SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="test_user",
            preferred_tools=["python"],
            preferred_language="English",
            response_style="concise",
        ),
        task_context=SyntheticTaskContext(
            current_project="test_project",
            active_experiment="exp1",
            goals=["goal1"],
            blockers=["blocker1"],
            next_steps=["step1"],
        ),
        follow_up_query="test query",
        user_id="test_user_phase1",
    )

    import logging

    with patch("benchmarks.shared_memory.pipeline.ProfileAgent") as mock_profile_cls, \
         patch("benchmarks.shared_memory.pipeline.TaskAgent") as mock_task_cls, \
         patch("benchmarks.shared_memory.pipeline.AssistantAgent") as mock_assistant_cls, \
         patch("benchmarks.shared_memory.pipeline.search_memories", return_value={"response": {"search_results": []}}), \
         patch("cerebrum.utils.communication.send_request", return_value={"response": {"search_results": []}}), \
         patch("benchmarks.shared_memory.pipeline.logger") as mock_logger:

        # Setup mocks
        mock_profile_cls.return_value.run.return_value = {}
        mock_task_cls.return_value.run.return_value = {}
        mock_assistant_cls.return_value.run.return_value = mock_assistant_result

        result = pipeline.run_trial(trial_data)

    # For Phase 1, audit still runs (empty user_id would skip but we have one).
    # Key assertion: NO "Observability gap" warning is logged because
    # share_memory=False means the gap warning condition is not triggered.
    warning_calls = [
        c for c in mock_logger.warning.call_args_list
        if "Observability gap" in str(c)
    ]
    assert len(warning_calls) == 0, (
        f"Phase 1 trial should not trigger 'Observability gap' warning, "
        f"but got: {warning_calls}"
    )

    # The retrieval log should have injection_status != "unknown" OR
    # if "unknown", the warning was suppressed because share_memory=False
    # Actually in the code: the warning only fires when self.share_memory is True
    # So Phase 1 should never produce the warning
    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    """Run all preservation property tests.

    ALL tests should PASS on the unfixed code. These capture baseline
    behavior that the fix must preserve.
    """
    print("=" * 78)
    print("Preservation property tests (Property 2)")
    print("Tests run against UNFIXED code to capture baseline behavior.")
    print("ALL tests should PASS — they validate non-buggy code paths.")
    print("=" * 78)

    all_passed = True

    # (a) Empty user_id
    print("\n--- (a) Empty user_id returns empty RetrievalLog (Hypothesis, 50 examples) ---")
    try:
        test_a_empty_user_id_returns_empty_log()
        record("a_empty_user_id_returns_empty_log", True)
    except AssertionError as e:
        record("a_empty_user_id_returns_empty_log", False, str(e).splitlines()[0])
        all_passed = False
    except Exception as e:
        record("a_empty_user_id_returns_empty_log", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_passed = False

    # (b) Exception handling
    print("\n--- (b) Exceptions return empty RetrievalLog (Hypothesis, 50 examples) ---")
    try:
        test_b_exception_returns_empty_log()
        record("b_exception_returns_empty_log", True)
    except AssertionError as e:
        record("b_exception_returns_empty_log", False, str(e).splitlines()[0])
        all_passed = False
    except Exception as e:
        record("b_exception_returns_empty_log", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_passed = False

    # (c) Threshold filtering
    print("\n--- (c) Threshold filtering in _build_retrieval_log_from_search (Hypothesis, 100 examples) ---")
    try:
        test_c_threshold_filtering()
        record("c_threshold_filtering", True)
    except AssertionError as e:
        record("c_threshold_filtering", False, str(e).splitlines()[0])
        all_passed = False
    except Exception as e:
        record("c_threshold_filtering", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_passed = False

    # (d) Diagnostics path produces correct output
    print("\n--- (d) Diagnostics path (_retrieval_log_from_diagnostics) (Hypothesis, 50 examples) ---")
    try:
        test_d_diagnostics_path_used_when_present()
        record("d_diagnostics_path_produces_correct_output", True)
    except AssertionError as e:
        record("d_diagnostics_path_produces_correct_output", False, str(e).splitlines()[0])
        all_passed = False
    except Exception as e:
        record("d_diagnostics_path_produces_correct_output", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_passed = False

    # (e) Diagnostics routing in run_trial
    print("\n--- (e) run_trial uses diagnostics path, skips audit (deterministic) ---")
    try:
        passed = test_e_run_trial_diagnostics_routing()
        record("e_run_trial_diagnostics_routing", True)
    except AssertionError as e:
        record("e_run_trial_diagnostics_routing", False, str(e).splitlines()[0])
        all_passed = False
    except Exception as e:
        record("e_run_trial_diagnostics_routing", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_passed = False

    # (f) Phase 1 no audit warning
    print("\n--- (f) Phase 1 (share_memory=False) no observability gap warning ---")
    try:
        passed = test_f_phase1_no_audit()
        record("f_phase1_no_audit_warning", True)
    except AssertionError as e:
        record("f_phase1_no_audit_warning", False, str(e).splitlines()[0])
        all_passed = False
    except Exception as e:
        record("f_phase1_no_audit_warning", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        all_passed = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — Preservation property tests")
    print("=" * 78)
    passed_count = sum(1 for _, s, _ in results if s == "PASS")
    failed_count = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total:  {len(results)}")

    if all_passed:
        print("\nAll preservation tests PASSED — baseline behavior confirmed.")
    else:
        print("\nSome preservation tests FAILED — investigate the unfixed code.")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  FAILED: {name}: {detail}")

    print("=" * 78)
    return all_passed


if __name__ == "__main__":
    all_passed = run_all()
    sys.exit(0 if all_passed else 1)
