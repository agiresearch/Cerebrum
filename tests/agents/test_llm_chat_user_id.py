"""Regression tests: user_id flows through AssistantAgent → llm_chat → LLMQuery.

Verifies that:
1. llm_chat(user_id="alice") creates an LLMQuery with user_id="alice".
2. llm_chat() without user_id keeps user_id=None (backward compat).
3. AssistantAgent.user_id is forwarded into llm_chat().
4. AssistantAgent without user_id passes None (backward compat).

These tests would FAIL if user_id forwarding were removed.
No running kernel required — all external calls are mocked.
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch


def test_llm_chat_forwards_user_id_to_query():
    """llm_chat(user_id='alice_123') creates LLMQuery with user_id='alice_123'."""
    from cerebrum.llm.apis import llm_chat

    captured = {}

    def fake_send_request(agent_name, query, base_url):
        captured["agent_name"] = agent_name
        captured["query"] = query
        captured["base_url"] = base_url
        return {"ok": True}

    with patch("cerebrum.llm.apis.send_request", side_effect=fake_send_request):
        llm_chat(
            agent_name="assistant_agent",
            messages=[{"role": "user", "content": "hello"}],
            base_url="http://kernel",
            user_id="alice_123",
        )

    assert captured["agent_name"] == "assistant_agent"
    assert captured["query"].user_id == "alice_123", (
        f"Expected user_id='alice_123', got {captured['query'].user_id!r}"
    )


def test_llm_chat_without_user_id_keeps_none():
    """llm_chat() without user_id keeps LLMQuery.user_id=None."""
    from cerebrum.llm.apis import llm_chat

    captured = {}

    def fake_send_request(agent_name, query, base_url):
        captured["query"] = query
        return {"ok": True}

    with patch("cerebrum.llm.apis.send_request", side_effect=fake_send_request):
        llm_chat(
            agent_name="assistant_agent",
            messages=[{"role": "user", "content": "hello"}],
            base_url="http://kernel",
        )

    assert captured["query"].user_id is None, (
        f"Expected user_id=None, got {captured['query'].user_id!r}"
    )


def test_assistant_agent_passes_user_id_to_llm_chat():
    """AssistantAgent.user_id='alice_123' is forwarded into llm_chat()."""
    from cerebrum.example.agents.assistant_agent.agent import AssistantAgent

    captured = {}

    def fake_llm_chat(**kwargs):
        captured.update(kwargs)
        return {"response": {"response_message": "mocked"}}

    with patch(
        "cerebrum.example.agents.assistant_agent.agent.llm_chat",
        side_effect=fake_llm_chat,
    ):
        agent = AssistantAgent("assistant_agent")
        agent.user_id = "alice_123"
        agent.run("hello")

    assert "user_id" in captured, "llm_chat was not called with user_id kwarg"
    assert captured["user_id"] == "alice_123", (
        f"Expected user_id='alice_123', got {captured['user_id']!r}"
    )


def test_assistant_agent_without_user_id_passes_none():
    """AssistantAgent without user_id set passes None to llm_chat()."""
    from cerebrum.example.agents.assistant_agent.agent import AssistantAgent

    captured = {}

    def fake_llm_chat(**kwargs):
        captured.update(kwargs)
        return {"response": {"response_message": "mocked"}}

    with patch(
        "cerebrum.example.agents.assistant_agent.agent.llm_chat",
        side_effect=fake_llm_chat,
    ):
        agent = AssistantAgent("assistant_agent")
        # Deliberately do NOT set agent.user_id
        agent.run("hello")

    assert "user_id" in captured, "llm_chat was not called with user_id kwarg"
    assert captured["user_id"] is None, (
        f"Expected user_id=None, got {captured['user_id']!r}"
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
    print("LLM Chat user_id Forwarding — Regression Tests")
    print("Confirms user_id flows: AssistantAgent → llm_chat → LLMQuery")
    print("=" * 70)

    print("\n--- llm_chat forwards user_id ---")
    try:
        test_llm_chat_forwards_user_id_to_query()
        record("llm_chat_forwards_user_id", True)
    except Exception as e:
        record("llm_chat_forwards_user_id", False, str(e)[:200])

    print("\n--- llm_chat without user_id keeps None ---")
    try:
        test_llm_chat_without_user_id_keeps_none()
        record("llm_chat_without_user_id_none", True)
    except Exception as e:
        record("llm_chat_without_user_id_none", False, str(e)[:200])

    print("\n--- AssistantAgent passes user_id ---")
    try:
        test_assistant_agent_passes_user_id_to_llm_chat()
        record("assistant_agent_passes_user_id", True)
    except Exception as e:
        record("assistant_agent_passes_user_id", False, str(e)[:200])

    print("\n--- AssistantAgent without user_id passes None ---")
    try:
        test_assistant_agent_without_user_id_passes_none()
        record("assistant_agent_without_user_id_none", True)
    except Exception as e:
        record("assistant_agent_without_user_id_none", False, str(e)[:200])

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
