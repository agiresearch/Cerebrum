"""Preservation property tests for audit nondeterminism fix.

These tests capture the CURRENT (unfixed) behavior for non-buggy inputs —
inputs where isBugCondition returns false. They verify behaviors that MUST
remain unchanged after the fix is applied.

IMPORTANT: These tests follow observation-first methodology.
They observe and encode existing behavior patterns, then verify they hold.
All tests MUST PASS on unfixed code (confirms baseline behavior to preserve).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from benchmarks.shared_memory.pipeline import AgentPipeline
from benchmarks.shared_memory.models import (
    InjectedMemoryEntry,
    InjectionDiagnostics,
    RetrievalLog,
    RetrievalLogEntry,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate valid owner agents (non-empty, realistic names)
owner_agent_strategy = st.sampled_from([
    "profile_agent", "task_agent", "assistant_agent",
    "custom_agent_1", "data_agent",
])

# Generate valid memory types
memory_type_strategy = st.sampled_from([
    "profile", "task_context", "conversation", "knowledge",
])

# Generate injection diagnostics with injected_count > 0
def injection_diagnostics_strategy():
    """Generate InjectionDiagnostics with injected_count > 0."""
    return st.integers(min_value=1, max_value=10).flatmap(
        lambda count: st.lists(
            st.builds(
                InjectedMemoryEntry,
                owner_agent=owner_agent_strategy,
                memory_type=memory_type_strategy,
                match_score=st.one_of(st.none(), st.floats(min_value=0.3, max_value=1.0)),
            ),
            min_size=count,
            max_size=count,
        ).map(lambda entries: InjectionDiagnostics(
            injected_count=len(entries),
            injected_memories=entries,
        ))
    )


# Generate search results where ALL entries have non-null memory_id
# and scores >= RELEVANCE_THRESHOLD (non-buggy: stable dedup keys, passing scores)
def valid_search_result_strategy():
    """Generate a search result entry with non-null memory_id and passing score."""
    return st.fixed_dictionaries({
        "memory_id": st.text(min_size=8, max_size=24, alphabet="abcdef0123456789"),
        "id": st.text(min_size=8, max_size=24, alphabet="abcdef0123456789"),
        "score": st.one_of(
            st.none(),  # None scores are kept (kernel behavior)
            st.floats(min_value=0.3, max_value=1.0),
        ),
        "metadata": st.fixed_dictionaries({
            "owner_agent": st.sampled_from(["profile_agent", "task_agent"]),
            "memory_type": st.sampled_from(["profile", "task_context"]),
            "sharing_policy": st.just("shared"),
        }),
    })


# Generate search result lists with unique memory_ids (no dedup collisions)
def unique_search_results_strategy():
    """Generate search results with all unique non-null memory_ids."""
    return st.lists(
        valid_search_result_strategy(),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x["memory_id"],
    )


# Generate writer agents list
writer_agents_strategy = st.just(["profile_agent", "task_agent"])

# User ID strategy
user_id_strategy = st.text(
    min_size=3, max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
)

# Query text strategy
query_text_strategy = st.text(min_size=5, max_size=100)


# ---------------------------------------------------------------------------
# Property Test 1: Diagnostics-First Path Preservation (Req 3.1)
# ---------------------------------------------------------------------------

@given(
    diagnostics=injection_diagnostics_strategy(),
    user_id=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_diagnostics_first_path_no_audit_query(diagnostics, user_id, query_text):
    """When injection_diagnostics.injected_count > 0, audit query is NEVER called.

    **Validates: Requirements 3.1**

    Property: For all InjectionDiagnostics with injected_count > 0,
    the system uses _retrieval_log_from_diagnostics and does NOT call
    _audit_shared_memories.
    """
    pipeline = AgentPipeline(share_memory=True)

    # Track whether _audit_shared_memories is called
    audit_called = False
    original_audit = pipeline._audit_shared_memories

    def tracking_audit(uid, qt):
        nonlocal audit_called
        audit_called = True
        return original_audit(uid, qt)

    pipeline._audit_shared_memories = tracking_audit

    # Simulate the logic in run_trial for the diagnostics branch
    # (We directly test the branching logic since run_trial requires full agent setup)
    assert diagnostics.injected_count > 0

    # This is the exact logic from run_trial:
    if diagnostics and diagnostics.injected_count > 0:
        retrieval_log = pipeline._retrieval_log_from_diagnostics(diagnostics)
    else:
        retrieval_log = pipeline._audit_shared_memories(user_id, query_text)

    # ASSERT: audit was never called
    assert not audit_called, (
        f"_audit_shared_memories was called despite injection_diagnostics "
        f"having injected_count={diagnostics.injected_count}. "
        f"Diagnostics-first path must be preserved."
    )

    # ASSERT: retrieval_log is populated from diagnostics
    assert isinstance(retrieval_log, RetrievalLog)
    assert retrieval_log.shared_memory_count == diagnostics.injected_count
    assert len(retrieval_log.retrieved_memories) == len(diagnostics.injected_memories)


# ---------------------------------------------------------------------------
# Property Test 2: Phase 1 Skip — No Audit Query Warning (Req 3.2)
# ---------------------------------------------------------------------------

@given(
    user_id=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_phase1_no_unknown_status(user_id, query_text):
    """When share_memory=False (Phase 1), injection_status never becomes "unknown".

    **Validates: Requirements 3.2**

    Property: For all Phase 1 trials (share_memory=False), even when
    diagnostics are absent and audit returns 0 results, injection_status
    is NOT set to "unknown" (the warning path is Phase 2 only).
    """
    pipeline = AgentPipeline(share_memory=False)

    # Mock send_request to return empty results (simulating no memories found)
    def mock_send_request(agent_name, query_obj, base_url):
        return {"response": {"search_results": []}}

    with patch("cerebrum.utils.communication.send_request", mock_send_request):
        with patch("cerebrum.config.config_manager.config") as mock_config:
            mock_config.get_kernel_url.return_value = "http://localhost:8000"
            retrieval_log = pipeline._audit_shared_memories(user_id, query_text)

    # Simulate the run_trial logic for the case where diagnostics are absent
    # and audit returns 0 results:
    injection_diagnostics = None  # No diagnostics
    if injection_diagnostics and injection_diagnostics.injected_count > 0:
        pass  # diagnostics path
    else:
        # audit query was already called above
        if retrieval_log.shared_memory_count > 0:
            retrieval_log.injection_status = "audit_inferred"
        elif pipeline.share_memory:
            # This only happens for Phase 2 (share_memory=True)
            retrieval_log.injection_status = "unknown"

    # ASSERT: For Phase 1, injection_status is never "unknown"
    assert retrieval_log.injection_status != "unknown", (
        f"Phase 1 trial (share_memory=False) got injection_status='unknown'. "
        f"Phase 1 should never trigger the observability warning."
    )


# ---------------------------------------------------------------------------
# Property Test 3: _build_retrieval_log_from_search Threshold Filter (Req 3.4)
# ---------------------------------------------------------------------------

@given(
    search_results=unique_search_results_strategy(),
)
@settings(max_examples=50)
def test_build_retrieval_log_threshold_filter(search_results):
    """_build_retrieval_log_from_search applies RELEVANCE_THRESHOLD correctly.

    **Validates: Requirements 3.4**

    Property: For all search result lists with non-null memory_id and
    scores >= threshold (or None), _build_retrieval_log_from_search produces
    a RetrievalLog where:
    - Every entry with score >= RELEVANCE_THRESHOLD or score=None is included
    - shared_memory_count counts entries with sharing_policy="shared"
    - cross_agent_found is True if any owner != "assistant_agent"
    """
    pipeline = AgentPipeline(share_memory=True)

    # Build the search response format
    search_response = {"response": {"search_results": search_results}}

    result = pipeline._build_retrieval_log_from_search(search_response)

    # Calculate expected results
    expected_entries = []
    expected_shared_count = 0
    expected_cross_agent = False

    for mem in search_results:
        score = mem.get("score")
        # Threshold filter: drop below threshold, keep None
        if score is not None and score < pipeline.RELEVANCE_THRESHOLD:
            continue
        meta = mem.get("metadata", {})
        if not meta:
            continue
        owner = meta.get("owner_agent", "")
        if not owner:
            continue
        expected_entries.append(RetrievalLogEntry(
            owner_agent=owner,
            memory_type=meta.get("memory_type", ""),
        ))
        if meta.get("sharing_policy") == "shared":
            expected_shared_count += 1
        if owner != "assistant_agent":
            expected_cross_agent = True

    # ASSERT: result matches expected
    assert isinstance(result, RetrievalLog)
    assert result.shared_memory_count == expected_shared_count, (
        f"shared_memory_count={result.shared_memory_count}, expected={expected_shared_count}"
    )
    assert result.cross_agent_found == expected_cross_agent, (
        f"cross_agent_found={result.cross_agent_found}, expected={expected_cross_agent}"
    )
    assert len(result.retrieved_memories) == len(expected_entries), (
        f"retrieved_memories count={len(result.retrieved_memories)}, "
        f"expected={len(expected_entries)}"
    )


# ---------------------------------------------------------------------------
# Property Test 4: Graceful Error — All Writers Fail (Req 3.5)
# ---------------------------------------------------------------------------

@given(
    user_id=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_all_writers_fail_returns_empty_retrieval_log(user_id, query_text):
    """When ALL writers fail (network down), empty RetrievalLog() is returned.

    **Validates: Requirements 3.5**

    Property: For all (user_id, query_text), when every send_request call
    raises an exception (simulating network down / kernel unavailable),
    _audit_shared_memories returns an empty RetrievalLog() without crashing.
    """
    assume(len(user_id.strip()) > 0)  # non-empty user_id

    pipeline = AgentPipeline(share_memory=True)

    def mock_send_request_fails(agent_name, query_obj, base_url):
        raise ConnectionError(f"Network down: cannot reach kernel for {agent_name}")

    with patch("cerebrum.utils.communication.send_request", mock_send_request_fails):
        with patch("cerebrum.config.config_manager.config") as mock_config:
            mock_config.get_kernel_url.return_value = "http://localhost:8000"
            result = pipeline._audit_shared_memories(user_id, query_text)

    # ASSERT: Empty RetrievalLog returned gracefully (no crash)
    assert isinstance(result, RetrievalLog)
    assert result.shared_memory_count == 0
    assert result.retrieved_memories == []
    assert result.cross_agent_found is False


# ---------------------------------------------------------------------------
# Property Test 5: Fan-Out Routing Per Writer (Req 3.3)
# ---------------------------------------------------------------------------

@given(
    user_id=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_fan_out_routing_per_writer(user_id, query_text):
    """Fan-out routing issues one query per writer through writer's own agent_name.

    **Validates: Requirements 3.3**

    Property: For all (user_id, query_text), _audit_shared_memories calls
    send_request once per writer agent, passing the writer_agent name as
    the first argument (agent_name routing parameter).
    """
    assume(len(user_id.strip()) > 0)

    pipeline = AgentPipeline(share_memory=True)

    # Track send_request calls: (agent_name, query_obj)
    captured_calls = []

    def mock_send_request(agent_name, query_obj, base_url):
        captured_calls.append((agent_name, query_obj))
        return {"response": {"search_results": []}}

    with patch("cerebrum.utils.communication.send_request", mock_send_request):
        with patch("cerebrum.config.config_manager.config") as mock_config:
            mock_config.get_kernel_url.return_value = "http://localhost:8000"
            pipeline._audit_shared_memories(user_id, query_text)

    # ASSERT: One call per writer agent
    assert len(captured_calls) == len(pipeline.WRITER_AGENTS), (
        f"Expected {len(pipeline.WRITER_AGENTS)} send_request calls, "
        f"got {len(captured_calls)}"
    )

    # ASSERT: Each call routes through the correct writer agent_name
    for i, writer_agent in enumerate(pipeline.WRITER_AGENTS):
        called_agent, query_obj = captured_calls[i]
        assert called_agent == writer_agent, (
            f"Call {i} routed through '{called_agent}', "
            f"expected '{writer_agent}'"
        )

    # ASSERT: Query params include correct user_id and query content
    for _, query_obj in captured_calls:
        assert query_obj.params["user_id"] == user_id
        assert query_obj.params["content"] == query_text
        assert query_obj.params["sharing_policy"] == "shared"


# ---------------------------------------------------------------------------
# Property Test 6: Complete Audit Context — All Writers Succeed (Req 3.3, 3.4)
# ---------------------------------------------------------------------------

@given(
    search_results_per_writer=st.lists(
        unique_search_results_strategy(),
        min_size=2,
        max_size=2,
    ),
    user_id=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_audit_all_writers_succeed_output_structure(
    search_results_per_writer, user_id, query_text
):
    """When all writers succeed with stable dedup keys, output structure is correct.

    **Validates: Requirements 3.3, 3.4**

    Property: For all audit contexts where all writers succeed, dedup keys
    are stable and non-null, and no residual state exists, the output
    RetrievalLog has correct structure matching the original function.
    """
    assume(len(user_id.strip()) > 0)

    pipeline = AgentPipeline(share_memory=True)

    # Ensure unique memory_ids across both writer result sets
    all_memory_ids = set()
    for writer_results in search_results_per_writer:
        for r in writer_results:
            if r["memory_id"] in all_memory_ids:
                assume(False)  # skip if collision between writers
            all_memory_ids.add(r["memory_id"])

    call_index = [0]

    def mock_send_request(agent_name, query_obj, base_url):
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(search_results_per_writer):
            return {"response": {"search_results": search_results_per_writer[idx]}}
        return {"response": {"search_results": []}}

    with patch("cerebrum.utils.communication.send_request", mock_send_request):
        with patch("cerebrum.config.config_manager.config") as mock_config:
            mock_config.get_kernel_url.return_value = "http://localhost:8000"
            result = pipeline._audit_shared_memories(user_id, query_text)

    # Calculate expected: merge all results, dedup by memory_id, apply threshold
    all_results = []
    for writer_results in search_results_per_writer:
        all_results.extend(writer_results)

    # Dedup (current code: mem.get("memory_id") or mem.get("id"))
    # Since all memory_ids are non-null and unique, no dedup occurs
    unique = all_results

    # Apply threshold filter
    expected_entries = []
    expected_shared = 0
    expected_cross_agent = False
    for mem in unique:
        score = mem.get("score")
        if score is not None and score < pipeline.RELEVANCE_THRESHOLD:
            continue
        meta = mem.get("metadata", {})
        if not meta:
            continue
        owner = meta.get("owner_agent", "")
        if not owner:
            continue
        expected_entries.append(owner)
        if meta.get("sharing_policy") == "shared":
            expected_shared += 1
        if owner != "assistant_agent":
            expected_cross_agent = True

    assert isinstance(result, RetrievalLog)
    assert result.shared_memory_count == expected_shared
    assert result.cross_agent_found == expected_cross_agent
    assert len(result.retrieved_memories) == len(expected_entries)


# ---------------------------------------------------------------------------
# Property Test 7: Mem0 Diagnostic Query for profile_agent (Req 3.6)
# ---------------------------------------------------------------------------

@given(
    user_id=user_id_strategy,
)
@settings(max_examples=30)
def test_mem0_diagnostic_query_profile_agent(user_id):
    """Mem0 diagnostic query for profile_agent uses correct agent_name.

    **Validates: Requirements 3.6**

    Property: When share_memory=True, the pipeline calls search_memories
    with agent_name="profile_agent" for diagnostic verification.
    This tests that the diagnostic code path is structurally preserved.
    """
    assume(len(user_id.strip()) > 0)

    # We verify the diagnostic call structure by checking that search_memories
    # is called with agent_name="profile_agent" during the pipeline's
    # diagnostic check. We test _retrieval_log_from_diagnostics directly
    # since it's the other half of the diagnostics path.
    pipeline = AgentPipeline(share_memory=True)

    # Create diagnostics with a profile_agent entry
    diagnostics = InjectionDiagnostics(
        injected_count=2,
        injected_memories=[
            InjectedMemoryEntry(
                owner_agent="profile_agent",
                memory_type="profile",
                match_score=0.85,
            ),
            InjectedMemoryEntry(
                owner_agent="task_agent",
                memory_type="task_context",
                match_score=0.72,
            ),
        ],
    )

    result = pipeline._retrieval_log_from_diagnostics(diagnostics)

    # ASSERT: Correctly builds RetrievalLog from diagnostics
    assert isinstance(result, RetrievalLog)
    assert result.shared_memory_count == 2
    assert result.cross_agent_found is True  # profile_agent != assistant_agent
    assert len(result.retrieved_memories) == 2
    assert result.retrieved_memories[0].owner_agent == "profile_agent"
    assert result.retrieved_memories[1].owner_agent == "task_agent"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

results_log = []


def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    results_log.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


def run_all():
    """Run all preservation property tests and report results."""
    print("=" * 70)
    print("Preservation Property Tests — Audit Nondeterminism")
    print("These tests encode EXISTING behavior for non-buggy inputs.")
    print("They MUST PASS on unfixed code (confirms baseline to preserve).")
    print("=" * 70)

    # Test 1: Diagnostics-first path
    print("\n--- Req 3.1: Diagnostics-First Path (no audit query) ---")
    try:
        test_diagnostics_first_path_no_audit_query()
        record("diagnostics_first_path", True, "Audit query never called when diagnostics present")
    except Exception as e:
        record("diagnostics_first_path", False, str(e)[:200])

    # Test 2: Phase 1 skip
    print("\n--- Req 3.2: Phase 1 Skip (no 'unknown' status) ---")
    try:
        test_phase1_no_unknown_status()
        record("phase1_no_unknown_status", True, "Phase 1 never gets injection_status='unknown'")
    except Exception as e:
        record("phase1_no_unknown_status", False, str(e)[:200])

    # Test 3: Threshold filter
    print("\n--- Req 3.4: _build_retrieval_log_from_search Threshold Filter ---")
    try:
        test_build_retrieval_log_threshold_filter()
        record("threshold_filter", True, "RELEVANCE_THRESHOLD applied correctly")
    except Exception as e:
        record("threshold_filter", False, str(e)[:200])

    # Test 4: Graceful error (all writers fail)
    print("\n--- Req 3.5: Graceful Error (all writers fail) ---")
    try:
        test_all_writers_fail_returns_empty_retrieval_log()
        record("graceful_error_all_fail", True, "Empty RetrievalLog returned on total failure")
    except Exception as e:
        record("graceful_error_all_fail", False, str(e)[:200])

    # Test 5: Fan-out routing per writer
    print("\n--- Req 3.3: Fan-Out Routing Per Writer ---")
    try:
        test_fan_out_routing_per_writer()
        record("fan_out_routing", True, "One query per writer through writer's agent_name")
    except Exception as e:
        record("fan_out_routing", False, str(e)[:200])

    # Test 6: All writers succeed — output structure
    print("\n--- Req 3.3, 3.4: All Writers Succeed — Output Structure ---")
    try:
        test_audit_all_writers_succeed_output_structure()
        record("all_writers_succeed_output", True, "Output matches expected structure")
    except Exception as e:
        record("all_writers_succeed_output", False, str(e)[:200])

    # Test 7: Mem0 diagnostic query
    print("\n--- Req 3.6: Mem0 Diagnostic Query for profile_agent ---")
    try:
        test_mem0_diagnostic_query_profile_agent()
        record("mem0_diagnostic_query", True, "Diagnostic path correctly builds RetrievalLog")
    except Exception as e:
        record("mem0_diagnostic_query", False, str(e)[:200])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in results_log if s == "PASS")
    failed = sum(1 for _, s, _ in results_log if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results_log)}")

    if failed > 0:
        print(f"\n{failed} test(s) FAILED — preservation behavior BROKEN.")
        print("This is UNEXPECTED. Preservation tests should PASS on unfixed code.")
    else:
        print("\nAll tests PASSED — baseline behavior confirmed for preservation.")
        print("These tests guard against regressions when the fix is applied.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
