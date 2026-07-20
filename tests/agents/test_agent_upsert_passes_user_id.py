"""Regression tests: agent upsert methods pass user_id to search_memories.

Verifies that ProfileAgent._upsert_profile_memory and
TaskAgent._upsert_task_memory pass user_id as a keyword argument to
search_memories(...), so the kernel can scope the lookup correctly.

These tests would FAIL if user_id=user_id were removed from either call.
No running kernel required — all external calls are mocked.
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock


def test_profile_agent_upsert_passes_user_id():
    """ProfileAgent._upsert_profile_memory passes user_id=user_id to search_memories."""
    from cerebrum.example.agents.profile_agent.agent import ProfileAgent

    user_id = "test_user_alice_99"
    profile_data = {
        "user_name": "Alice",
        "preferred_tools": ["vim"],
        "preferred_language": "python",
        "response_style": "concise",
    }

    # Mock search_memories at the agent's import location
    with patch(
        "cerebrum.example.agents.profile_agent.agent.search_memories"
    ) as mock_search:
        mock_search.return_value = {"response": {"search_results": []}}

        # Mock create_memory so the write path doesn't hit the kernel
        with patch(
            "cerebrum.example.agents.profile_agent.agent.create_memory"
        ) as mock_create:
            mock_create.return_value = {
                "response": {"memory_id": "mem_new", "success": True}
            }

            agent = ProfileAgent("profile_agent")
            agent.user_id = user_id
            agent._upsert_profile_memory(user_id, profile_data)

    # ASSERT: search_memories was called with user_id as a keyword argument
    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args
    # call_args is (args, kwargs) — search_memories uses keyword args
    all_args = call_kwargs.kwargs if call_kwargs.kwargs else {}
    # Also check positional-as-keyword via call_args[1] for older mock styles
    if not all_args:
        all_args = call_kwargs[1] if len(call_kwargs) > 1 else {}

    # The function may be called with positional + keyword mix, so inspect both
    actual_call_kwargs = mock_search.call_args.kwargs
    actual_call_args = mock_search.call_args.args

    # user_id must be present as a keyword argument
    assert "user_id" in actual_call_kwargs, (
        f"search_memories was called without user_id keyword arg. "
        f"kwargs={actual_call_kwargs}, args={actual_call_args}"
    )
    assert actual_call_kwargs["user_id"] == user_id, (
        f"Expected user_id={user_id!r}, got {actual_call_kwargs['user_id']!r}"
    )

    # Verify other args are still present
    assert actual_call_kwargs.get("agent_name") == "profile_agent" or (
        len(actual_call_args) > 0 and actual_call_args[0] == "profile_agent"
    )


def test_task_agent_upsert_passes_user_id():
    """TaskAgent._upsert_task_memory passes user_id=user_id to search_memories."""
    from cerebrum.example.agents.task_agent.agent import TaskAgent

    user_id = "test_user_bob_42"
    context_data = {
        "current_project": "Project X",
        "active_experiment": "exp1",
        "goals": ["ship feature"],
        "blockers": ["ci broken"],
        "next_steps": ["fix ci"],
    }

    # Mock search_memories at the agent's import location
    with patch(
        "cerebrum.example.agents.task_agent.agent.search_memories"
    ) as mock_search:
        mock_search.return_value = {"response": {"search_results": []}}

        # Mock create_memory so the write path doesn't hit the kernel
        with patch(
            "cerebrum.example.agents.task_agent.agent.create_memory"
        ) as mock_create:
            mock_create.return_value = {
                "response": {"memory_id": "mem_new", "success": True}
            }

            agent = TaskAgent("task_agent")
            agent.user_id = user_id
            agent._upsert_task_memory(user_id, context_data)

    # ASSERT: search_memories was called with user_id as a keyword argument
    mock_search.assert_called_once()
    actual_call_kwargs = mock_search.call_args.kwargs
    actual_call_args = mock_search.call_args.args

    # user_id must be present as a keyword argument
    assert "user_id" in actual_call_kwargs, (
        f"search_memories was called without user_id keyword arg. "
        f"kwargs={actual_call_kwargs}, args={actual_call_args}"
    )
    assert actual_call_kwargs["user_id"] == user_id, (
        f"Expected user_id={user_id!r}, got {actual_call_kwargs['user_id']!r}"
    )

    # Verify other args are still present
    assert actual_call_kwargs.get("agent_name") == "task_agent" or (
        len(actual_call_args) > 0 and actual_call_args[0] == "task_agent"
    )


# ---------------------------------------------------------------------------
# Main runner (matches project convention)
# ---------------------------------------------------------------------------

results_log = []


def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results_log.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


def run_all():
    print("=" * 70)
    print("Agent Upsert user_id Regression Tests")
    print("Confirms search_memories receives user_id as a keyword argument.")
    print("=" * 70)

    print("\n--- ProfileAgent._upsert_profile_memory ---")
    try:
        test_profile_agent_upsert_passes_user_id()
        record("profile_agent_upsert_user_id", True)
    except Exception as e:
        record("profile_agent_upsert_user_id", False, str(e)[:200])

    print("\n--- TaskAgent._upsert_task_memory ---")
    try:
        test_task_agent_upsert_passes_user_id()
        record("task_agent_upsert_user_id", True)
    except Exception as e:
        record("task_agent_upsert_user_id", False, str(e)[:200])

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
        print(f"\n{failed} test(s) FAILED.")
    else:
        print("\nAll tests PASSED.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
