"""Bug-condition exploration test for audit-query-cross-agent-fix (Property 1).

This test confirms that `_audit_shared_memories` in the UNFIXED code routes
a single `search_memories` query through `"assistant_agent"`, and the kernel's
syscall handler scopes results to the routing agent's owned memories. Since
shared memories were written by `profile_agent` and `task_agent`, the query
returns zero results.

The test mocks `send_request` to simulate kernel agent-scoping behavior:
results are returned only when the routing `agent_name` matches the memory
owner agent. This reproduces the bug condition described in:
  `.kiro/specs/audit-query-cross-agent-fix/bugfix.md`

This test is EXPECTED TO FAIL on unfixed code. Failure confirms the bug exists:
- `_audit_shared_memories` routes through `"assistant_agent"`
- Mocked kernel returns 0 results for `assistant_agent` (no memories owned)
- The method returns `RetrievalLog(shared_memory_count=0)`
- The assertion `result.shared_memory_count > 0` fails

After the fix is applied (fan-out queries per writer agent), this test will PASS.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

from __future__ import annotations

import sys
import traceback
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.shared_memory.pipeline import AgentPipeline
from benchmarks.shared_memory.models import RetrievalLog


# ---------------------------------------------------------------------------
# Mock kernel agent-scoping behavior
# ---------------------------------------------------------------------------

# Simulated memories owned by each agent.
# The kernel returns results only when routing agent_name matches the owner.
MOCKED_MEMORIES = {
    "profile_agent": [
        {
            "memory_id": "mem-profile-001",
            "id": "mem-profile-001",
            "content": "User prefers dark mode and concise answers",
            "score": 0.85,
            "metadata": {
                "owner_agent": "profile_agent",
                "memory_type": "profile",
                "sharing_policy": "shared",
                "user_id": None,  # will be filled per-call
            },
        },
        {
            "memory_id": "mem-profile-002",
            "id": "mem-profile-002",
            "content": "User works primarily with Python and TypeScript",
            "score": 0.72,
            "metadata": {
                "owner_agent": "profile_agent",
                "memory_type": "profile",
                "sharing_policy": "shared",
                "user_id": None,
            },
        },
    ],
    "task_agent": [
        {
            "memory_id": "mem-task-001",
            "id": "mem-task-001",
            "content": "Currently debugging authentication flow in AIOS kernel",
            "score": 0.91,
            "metadata": {
                "owner_agent": "task_agent",
                "memory_type": "task_context",
                "sharing_policy": "shared",
                "user_id": None,
            },
        },
    ],
    "assistant_agent": [],  # assistant_agent owns no shared memories
}


def make_mock_send_request(user_id: str):
    """Create a mock send_request that simulates kernel agent-scoping.

    The kernel's syscall handler uses the top-level routing `agent_name`
    (first arg to send_request) to scope search results. Only memories
    owned by that agent are returned.

    Args:
        user_id: The user_id to stamp into returned memory metadata.

    Returns:
        A mock function with the same signature as send_request.
    """
    def mock_send_request(agent_name, query_obj, base_url):
        """Return memories only when routing agent_name matches owner."""
        memories = MOCKED_MEMORIES.get(agent_name, [])
        # Deep copy and stamp user_id into metadata
        results = []
        for mem in memories:
            mem_copy = {
                "memory_id": mem["memory_id"],
                "id": mem["id"],
                "content": mem["content"],
                "score": mem["score"],
                "metadata": {
                    **mem["metadata"],
                    "user_id": user_id,
                },
            }
            results.append(mem_copy)
        return {
            "response": {
                "search_results": results,
            }
        }
    return mock_send_request


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
# Sub-case (a) — Deterministic: single user, mocked memories exist
# ---------------------------------------------------------------------------

def test_a_single_user_audit_returns_zero() -> tuple[bool, str]:
    """(a) Single user audit query routes through assistant_agent, gets 0 results.

    EXPECTED on UNFIXED code: FAIL — _audit_shared_memories routes through
    "assistant_agent" which owns no shared memories, so result is 0.

    Validates: Requirements 1.1, 1.3
    """
    pipeline = AgentPipeline(share_memory=True)
    user_id = "user_alice"
    query_text = "What are my preferences?"

    mock_fn = make_mock_send_request(user_id)

    with patch("cerebrum.utils.communication.send_request", side_effect=mock_fn):
        result = pipeline._audit_shared_memories(user_id, query_text)

    passed = result.shared_memory_count > 0 and result.cross_agent_found is True
    detail = (
        f"user_id={user_id!r}, query={query_text!r}, "
        f"shared_memory_count={result.shared_memory_count}, "
        f"cross_agent_found={result.cross_agent_found}, "
        f"expected: shared_memory_count > 0, cross_agent_found=True"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (b) — Multiple queries same user, consistent zero results
# ---------------------------------------------------------------------------

def test_b_consistent_zero_results() -> tuple[bool, str]:
    """(b) Multiple queries for same user always return 0 on unfixed code.

    EXPECTED on UNFIXED code: FAIL — regardless of query_text, routing
    through assistant_agent always yields 0 results.

    Validates: Requirements 1.1, 1.2
    """
    pipeline = AgentPipeline(share_memory=True)
    user_id = "user_bob"
    queries = [
        "Tell me about my goals",
        "What project am I working on?",
        "Summarize my task context",
    ]

    all_zero = True
    for q in queries:
        mock_fn = make_mock_send_request(user_id)
        with patch("cerebrum.utils.communication.send_request", side_effect=mock_fn):
            result = pipeline._audit_shared_memories(user_id, q)
        if result.shared_memory_count > 0:
            all_zero = False

    # The property: at least one query should return results
    passed = not all_zero
    detail = (
        f"user_id={user_id!r}, queries={len(queries)}, "
        f"all returned shared_memory_count=0 (bug confirmed)"
        if all_zero else
        f"user_id={user_id!r}, some queries returned results (unexpected)"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (c) — Params-level agent_name is ignored by kernel scoping
# ---------------------------------------------------------------------------

def test_c_params_agent_name_ignored() -> tuple[bool, str]:
    """(c) The agent_name in params dict does not affect kernel scoping.

    EXPECTED on UNFIXED code: FAIL — the method includes
    "agent_name": "assistant_agent" in params, but the kernel uses the
    top-level routing agent_name (first arg to send_request), not the
    params-level field. Either way, assistant_agent returns 0.

    Validates: Requirements 1.3
    """
    pipeline = AgentPipeline(share_memory=True)
    user_id = "user_carol"
    query_text = "my preferences"

    call_log = []

    def logging_mock_send_request(agent_name, query_obj, base_url):
        """Log calls and simulate kernel agent-scoping."""
        call_log.append({
            "routing_agent": agent_name,
            "params_agent_name": getattr(query_obj, 'params', {}).get("agent_name"),
        })
        # Kernel uses routing agent_name for scoping
        memories = MOCKED_MEMORIES.get(agent_name, [])
        results = []
        for mem in memories:
            results.append({
                "memory_id": mem["memory_id"],
                "id": mem["id"],
                "content": mem["content"],
                "score": mem["score"],
                "metadata": {**mem["metadata"], "user_id": user_id},
            })
        return {"response": {"search_results": results}}

    with patch("cerebrum.utils.communication.send_request", side_effect=logging_mock_send_request):
        result = pipeline._audit_shared_memories(user_id, query_text)

    # Verify the unfixed code routes through assistant_agent
    routed_through_assistant = any(
        c["routing_agent"] == "assistant_agent" for c in call_log
    )
    # The property: despite params containing agent_name, results should be found
    passed = result.shared_memory_count > 0 and result.cross_agent_found is True
    detail = (
        f"routed_through_assistant={routed_through_assistant}, "
        f"call_log={call_log}, "
        f"shared_memory_count={result.shared_memory_count}, "
        f"cross_agent_found={result.cross_agent_found}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (d) — Hypothesis PBT: random user_id and query_text
# ---------------------------------------------------------------------------

_user_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
).filter(lambda s: len(s.strip()) > 0)

_query_text_strategy = st.text(min_size=1, max_size=100)


@given(
    user_id=_user_id_strategy,
    query_text=_query_text_strategy,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_d_hypothesis_cross_agent_audit(user_id: str, query_text: str) -> None:
    """(d) Hypothesis PBT — Property 1: Cross-Agent Audit Returns Shared Memories.

    For ANY user_id and query_text where shared memories exist for
    profile_agent and task_agent, calling _audit_shared_memories on the
    UNFIXED code SHALL return shared_memory_count > 0 and
    cross_agent_found == True.

    EXPECTED on UNFIXED code: FAILURE — the unfixed code always routes
    through assistant_agent which owns no shared memories, resulting in
    shared_memory_count=0 for all inputs.

    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    pipeline = AgentPipeline(share_memory=True)
    mock_fn = make_mock_send_request(user_id)

    with patch("cerebrum.utils.communication.send_request", side_effect=mock_fn):
        result = pipeline._audit_shared_memories(user_id, query_text)

    assert result.shared_memory_count > 0, (
        f"_audit_shared_memories({user_id!r}, {query_text!r}) returned "
        f"shared_memory_count={result.shared_memory_count}, expected > 0. "
        f"Bug: unfixed code routes through 'assistant_agent' which owns no "
        f"shared memories. Memories exist for profile_agent and task_agent."
    )
    assert result.cross_agent_found is True, (
        f"_audit_shared_memories({user_id!r}, {query_text!r}) returned "
        f"cross_agent_found={result.cross_agent_found}, expected True. "
        f"Bug: no cross-agent memories found because routing mismatch."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    """Run every sub-case. Each FAIL is a counterexample confirming the bug.

    EXPECTED OUTCOME on UNFIXED code: all sub-cases FAIL. Failure
    confirms the bug exists — _audit_shared_memories routes through
    assistant_agent and the kernel returns 0 results despite memories
    existing for profile_agent and task_agent.
    """
    print("=" * 78)
    print("Audit query cross-agent bug-condition exploration (Property 1)")
    print("Tests run against UNFIXED _audit_shared_memories code.")
    print("FAILURE on the unfixed code IS the expected outcome — it")
    print("confirms the agent-scoping mismatch documented in bugfix.md.")
    print("=" * 78)

    print("\n--- (a) Single user audit returns zero ---")
    passed_a, detail_a = test_a_single_user_audit_returns_zero()
    record("a_single_user_audit_returns_zero", passed_a, detail_a)

    print("\n--- (b) Consistent zero results across queries ---")
    passed_b, detail_b = test_b_consistent_zero_results()
    record("b_consistent_zero_results", passed_b, detail_b)

    print("\n--- (c) Params-level agent_name ignored ---")
    passed_c, detail_c = test_c_params_agent_name_ignored()
    record("c_params_agent_name_ignored", passed_c, detail_c)

    print("\n--- (d) Hypothesis PBT (100 examples) ---")
    try:
        test_d_hypothesis_cross_agent_audit()
        record(
            "d_hypothesis_cross_agent_audit",
            True,
            "All 100 hypothesis examples passed (unexpected — bug not reproduced)",
        )
    except AssertionError as e:
        msg = str(e).splitlines()[0]
        record(
            "d_hypothesis_cross_agent_audit",
            False,
            f"Hypothesis counterexample: {msg}",
        )
    except Exception as e:
        record(
            "d_hypothesis_cross_agent_audit",
            False,
            f"Hypothesis raised non-assertion error: {type(e).__name__}: {e}",
        )
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — Bug-condition exploration")
    print("=" * 78)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed} (expected — each failure is a bug counterexample)")
    print(f"  Total:  {len(results)}")

    print("\nCounterexamples (each one demonstrates the routing mismatch bug):")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")

    print("=" * 78)

    # Exploration semantics:
    # - exit code 0 when failures exist (the bug was reproduced — success)
    # - exit code 1 when EVERY sub-case passed (unexpected — bug not found)
    return failed > 0


if __name__ == "__main__":
    bug_reproduced = run_all()
    if bug_reproduced:
        print("\nEXPLORATION RESULT: bug reproduced on unfixed code.")
        sys.exit(0)
    else:
        print(
            "\nEXPLORATION RESULT: bug NOT reproduced — investigate the test or code."
        )
        sys.exit(1)
