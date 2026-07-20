"""Regression tests for the explicit user_id contract.

These tests lock in the invariant that Cerebrum NEVER retrieves user-scoped
memories unless an explicit request-scoped user_id is provided. They protect
against the "latest writer wins" identity leak where the kernel's
process-global `latest_user_id` state could cause cross-user contamination.

Test scenarios:
  1. Retrieval without explicit user_id returns no user-scoped memories.
  2. Retrieval with explicit user_id returns only that user's memories.
  3. Sequential writes for user_a then user_b do NOT make missing-user
     retrieval return user_b (no "latest writer wins" fallback).
  4. Context injection with missing request user_id skips user-scoped
     memory injection.
  5. Context injection with explicit request user_id injects only that
     user's memories.

These tests are unit-level (mock-based) and do NOT require a running kernel.

Expected behavior: These tests define the CORRECT contract. They may FAIL
on current code if the production behavior has not yet been updated to
enforce explicit user_id requirements. Failure confirms the contract
violation exists and motivates the production fix.
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock, call
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from cerebrum.memory.apis import search_memories, MemoryQuery
from cerebrum.utils.communication import send_request


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

user_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="_"),
    min_size=5,
    max_size=30,
)

query_text_strategy = st.text(min_size=5, max_size=80)


def make_memory_result(user_id, owner_agent="profile_agent", memory_type="profile"):
    """Build a mock search result entry for a given user_id."""
    return {
        "memory_id": f"mem_{user_id}_{owner_agent}",
        "score": 0.85,
        "content": f"Profile data for {user_id}",
        "metadata": {
            "owner_agent": owner_agent,
            "memory_type": memory_type,
            "sharing_policy": "shared",
            "user_id": user_id,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Retrieval without explicit user_id returns no user-scoped memories
# ---------------------------------------------------------------------------

@given(
    user_a=user_id_strategy,
    user_b=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_retrieval_without_user_id_returns_empty(user_a, user_b, query_text):
    """Retrieval without explicit user_id must return no user-scoped memories.

    Scenario:
    - Memories exist for user_a and user_b in the store.
    - search_memories is called WITHOUT user_id parameter.
    - The SDK must NOT return user-scoped memories via fallback.

    This ensures the SDK does not silently use a cached/global user_id.
    """
    assume(user_a != user_b)

    # Capture the MemoryQuery sent to the kernel
    captured_queries = []

    def mock_send_request(agent_name, query_obj, base_url):
        captured_queries.append(query_obj)
        # Kernel returns empty when no user_id scoping is provided
        # (agent-scoped search only)
        return {"response": {"search_results": []}}

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        result = search_memories(
            agent_name="assistant_agent",
            query=query_text,
            k=5,
        )

    # ASSERT: The query params must NOT contain a user_id
    assert len(captured_queries) == 1
    params = captured_queries[0].params
    assert "user_id" not in params, (
        f"search_memories called without user_id kwarg, but params contains "
        f"user_id={params.get('user_id')!r}. The SDK must not inject a "
        f"user_id from process-global state."
    )

    # ASSERT: Result is empty (no user-scoped memories leaked)
    resp = result.get("response", {})
    search_results = resp.get("search_results", [])
    assert search_results == [], (
        f"Expected empty results without user_id, got {len(search_results)} results"
    )


# ---------------------------------------------------------------------------
# Test 2: Retrieval with explicit user_id returns only that user's memories
# ---------------------------------------------------------------------------

@given(
    user_a=user_id_strategy,
    user_b=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_retrieval_with_explicit_user_id_scopes_correctly(user_a, user_b, query_text):
    """Retrieval with explicit user_id must return ONLY that user's memories.

    Scenario:
    - Memories exist for user_a and user_b.
    - search_memories called with user_id=user_a.
    - Only user_a memories are returned; user_b memories are excluded.
    """
    assume(user_a != user_b)

    captured_queries = []

    def mock_send_request(agent_name, query_obj, base_url):
        captured_queries.append(query_obj)
        # Kernel correctly scopes to user_id in params
        requested_user_id = query_obj.params.get("user_id")
        if requested_user_id == user_a:
            return {
                "response": {
                    "search_results": [make_memory_result(user_a)]
                }
            }
        elif requested_user_id == user_b:
            return {
                "response": {
                    "search_results": [make_memory_result(user_b)]
                }
            }
        else:
            return {"response": {"search_results": []}}

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        result = search_memories(
            agent_name="assistant_agent",
            query=query_text,
            k=5,
            user_id=user_a,
        )

    # ASSERT: The query params contain exactly user_a
    assert len(captured_queries) == 1
    params = captured_queries[0].params
    assert params.get("user_id") == user_a, (
        f"Expected user_id={user_a!r} in params, got {params.get('user_id')!r}"
    )

    # ASSERT: Only user_a memories returned
    resp = result.get("response", {})
    search_results = resp.get("search_results", [])
    assert len(search_results) == 1
    assert search_results[0]["metadata"]["user_id"] == user_a

    # ASSERT: user_b memories NOT present
    for mem in search_results:
        assert mem["metadata"]["user_id"] != user_b, (
            f"user_b memory leaked into user_a scoped query"
        )


# ---------------------------------------------------------------------------
# Test 3: No fallback to "latest writer wins"
# ---------------------------------------------------------------------------

@given(
    user_a=user_id_strategy,
    user_b=user_id_strategy,
    query_text=query_text_strategy,
)
@settings(max_examples=50)
def test_no_fallback_to_latest_writer(user_a, user_b, query_text):
    """Sequential writes for user_a then user_b must NOT make retrieval
    without user_id return user_b's memories.

    This specifically protects against the "latest writer wins" identity
    leak where kernel-side process-global `latest_user_id` state from a
    prior write could pollute a subsequent read.

    Scenario:
    - create_memory called for user_a (simulated write)
    - create_memory called for user_b (simulated write — now "latest")
    - search_memories called WITHOUT user_id
    - Neither user_a nor user_b memories should be returned
    """
    assume(user_a != user_b)

    captured_queries = []

    def mock_send_request(agent_name, query_obj, base_url):
        captured_queries.append(query_obj)
        query_class = query_obj.query_class

        if query_class == "memory" and query_obj.operation_type == "add_memory":
            # Write succeeds
            return {"response": {"memory_id": f"mem_{agent_name}_new", "success": True}}
        elif query_class == "memory" and query_obj.operation_type == "retrieve_memory":
            # Read: if no user_id in params, return empty (correct behavior)
            # If user_id is present (leaked), return that user's data (bug)
            params = query_obj.params
            if "user_id" in params:
                # This is the bug: SDK should NOT inject user_id without
                # the caller explicitly providing it
                leaked_uid = params["user_id"]
                return {
                    "response": {
                        "search_results": [make_memory_result(leaked_uid)]
                    }
                }
            return {"response": {"search_results": []}}

        return {"response": {}}

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        from cerebrum.memory.apis import create_memory

        # Simulate sequential writes (user_a then user_b)
        create_memory(
            agent_name="profile_agent",
            content="Profile for user_a",
            metadata={"user_id": user_a, "owner_agent": "profile_agent",
                      "memory_type": "profile", "sharing_policy": "shared"},
        )
        create_memory(
            agent_name="task_agent",
            content="Task context for user_b",
            metadata={"user_id": user_b, "owner_agent": "task_agent",
                      "memory_type": "task_context", "sharing_policy": "shared"},
        )

        # Now retrieve WITHOUT specifying user_id
        result = search_memories(
            agent_name="assistant_agent",
            query=query_text,
            k=5,
        )

    # Find the retrieve_memory query in captured_queries
    retrieve_queries = [
        q for q in captured_queries
        if q.query_class == "memory" and q.operation_type == "retrieve_memory"
    ]
    assert len(retrieve_queries) == 1

    # ASSERT: No user_id in the retrieval params
    retrieve_params = retrieve_queries[0].params
    assert "user_id" not in retrieve_params, (
        f"search_memories without user_id kwarg injected user_id="
        f"{retrieve_params.get('user_id')!r} into params. "
        f"This is the 'latest writer wins' identity leak."
    )

    # ASSERT: No user-scoped memories returned
    resp = result.get("response", {})
    search_results = resp.get("search_results", [])
    assert search_results == [], (
        f"Expected empty results (no user_id specified), but got "
        f"{len(search_results)} results. Latest writer leak detected."
    )


# ---------------------------------------------------------------------------
# Test 4: Context injection with missing request user_id skips injection
# ---------------------------------------------------------------------------

@given(
    user_a=user_id_strategy,
    user_b=user_id_strategy,
)
@settings(max_examples=50)
def test_context_injection_skips_without_user_id(user_a, user_b):
    """Context injection must skip user-scoped memory retrieval when no
    request-scoped user_id is provided.

    Scenario:
    - Memories exist for user_a and user_b in the store.
    - ContextInjector.inject() called with empty/missing user_id.
    - No user-scoped memories should be injected.
    """
    assume(user_a != user_b)

    from aios.memory.context_injector import ContextInjector, InjectionResult

    # Create a mock Mem0Provider that would return memories if called
    mock_provider = MagicMock()
    mock_provider.get_all_shared_cross_agent.return_value = [
        {
            "content": f"Profile for {user_a}",
            "metadata": {"user_id": user_a, "owner_agent": "profile_agent",
                         "sharing_policy": "shared", "memory_type": "profile"},
        },
        {
            "content": f"Task for {user_b}",
            "metadata": {"user_id": user_b, "owner_agent": "task_agent",
                         "sharing_policy": "shared", "memory_type": "task_context"},
        },
    ]

    # Mock config with auto_inject=True
    mock_config = MagicMock()
    mock_config.memory.auto_inject = True
    mock_config.memory.write_barrier.enabled = False
    mock_config.memory.relevance_threshold = 0.3
    mock_config.memory.max_injected_memories = 10
    mock_config.memory.max_memory_tokens = 2000

    # Mock pending tracker (no pending writes)
    mock_tracker = MagicMock()
    mock_tracker.pending_for.return_value = []

    injector = ContextInjector(
        mem0_provider=mock_provider,
        config=mock_config,
        pending_tracker=mock_tracker,
    )

    # Call inject with empty user_id (no request-scoped identity)
    result = injector.inject(user_id="", calling_agent="assistant_agent")

    # ASSERT: When user_id is empty, no memories should be injected.
    # The injector should either:
    # (a) not call get_all_shared_cross_agent at all, OR
    # (b) call it with empty user_id and get no results back
    #
    # The correct behavior is that empty user_id means "no user context"
    # and injection should be skipped or return empty.
    assert isinstance(result, InjectionResult)
    assert len(result.memories) == 0, (
        f"Expected 0 injected memories with empty user_id, "
        f"got {len(result.memories)}. Context injection must not inject "
        f"user-scoped memories without an explicit request-scoped user_id."
    )


# ---------------------------------------------------------------------------
# Test 5: Context injection with explicit user_id injects only that user
# ---------------------------------------------------------------------------

@given(
    user_a=user_id_strategy,
    user_b=user_id_strategy,
)
@settings(max_examples=50)
def test_context_injection_scopes_to_explicit_user_id(user_a, user_b):
    """Context injection with explicit user_id must inject ONLY that user's
    memories and exclude other users.

    Scenario:
    - Memories exist for user_a and user_b.
    - ContextInjector.inject() called with user_id=user_a.
    - Only user_a memories appear in the injected result.
    - user_b memories do NOT appear.
    """
    assume(user_a != user_b)

    from aios.memory.context_injector import ContextInjector, InjectionResult

    # The provider should only return user_a memories when called with user_a
    def mock_get_all_shared(user_id, calling_agent):
        if user_id == user_a:
            return [
                {
                    "content": f"Profile for {user_a}",
                    "metadata": {"user_id": user_a, "owner_agent": "profile_agent",
                                 "sharing_policy": "shared", "memory_type": "profile"},
                },
            ]
        elif user_id == user_b:
            return [
                {
                    "content": f"Task for {user_b}",
                    "metadata": {"user_id": user_b, "owner_agent": "task_agent",
                                 "sharing_policy": "shared", "memory_type": "task_context"},
                },
            ]
        return []

    mock_provider = MagicMock()
    mock_provider.get_all_shared_cross_agent.side_effect = mock_get_all_shared

    # Mock config
    mock_config = MagicMock()
    mock_config.memory.auto_inject = True
    mock_config.memory.write_barrier.enabled = False
    mock_config.memory.relevance_threshold = 0.3
    mock_config.memory.max_injected_memories = 10
    mock_config.memory.max_memory_tokens = 2000

    # Mock pending tracker
    mock_tracker = MagicMock()
    mock_tracker.pending_for.return_value = []

    injector = ContextInjector(
        mem0_provider=mock_provider,
        config=mock_config,
        pending_tracker=mock_tracker,
    )

    # Call inject with explicit user_id = user_a
    result = injector.inject(user_id=user_a, calling_agent="assistant_agent")

    # ASSERT: Provider was called with user_a
    mock_provider.get_all_shared_cross_agent.assert_called_once_with(
        user_id=user_a,
        calling_agent="assistant_agent",
    )

    # ASSERT: Only user_a memories in result
    assert isinstance(result, InjectionResult)
    assert len(result.memories) >= 1, (
        f"Expected at least 1 memory for user_a, got {len(result.memories)}"
    )

    for mem in result.memories:
        mem_user_id = mem.get("metadata", {}).get("user_id", "")
        assert mem_user_id == user_a, (
            f"Injected memory belongs to {mem_user_id!r}, expected {user_a!r}. "
            f"Cross-user contamination detected."
        )
        assert mem_user_id != user_b, (
            f"user_b memory leaked into user_a injection context"
        )


# ---------------------------------------------------------------------------
# Test 6: Agent fallback chain does not use agent_name as user_id
# ---------------------------------------------------------------------------

@given(
    query_text=query_text_strategy,
)
@settings(max_examples=30)
def test_profile_agent_fallback_does_not_use_agent_name_for_retrieval(query_text):
    """ProfileAgent's user_id fallback chain must not use agent_name as user_id
    for memory retrieval/write operations when no explicit user_id is set.

    The current code does:
        user_id = getattr(self, 'user_id', profile_data.get("user_name", self.agent_name))

    This means if self.user_id is not set AND the LLM doesn't return a
    user_name, the agent uses "profile_agent" as user_id — which is wrong.
    Memories should not be scoped to the agent's own name.

    This test verifies that when self.user_id IS set, it takes precedence
    over all fallbacks (current correct behavior when pipeline sets it).
    """
    from cerebrum.example.agents.profile_agent.agent import ProfileAgent

    captured_queries = []
    explicit_user_id = "alice_explicit_123"

    def mock_send_request(agent_name, query_obj, base_url):
        captured_queries.append((agent_name, query_obj))
        if hasattr(query_obj, 'query_class'):
            if query_obj.query_class == "llm":
                # Return a valid JSON profile extraction
                return {
                    "response": {
                        "response_message": '{"user_name": "should_not_use_this", "preferred_tools": ["vim"], "preferred_language": "python", "response_style": "concise"}'
                    }
                }
            elif query_obj.query_class == "memory":
                if query_obj.operation_type == "retrieve_memory":
                    return {"response": {"search_results": []}}
                elif query_obj.operation_type == "add_memory":
                    return {"response": {"memory_id": "mem_new_1", "success": True}}
        return {"response": {}}

    agent = ProfileAgent("profile_agent")
    agent.user_id = explicit_user_id  # Explicitly set by pipeline

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        with patch("cerebrum.llm.apis.send_request", mock_send_request):
            result = agent.run('{"user_name": "should_not_use_this", "preferred_tools": ["vim"], "preferred_language": "python", "response_style": "concise"}')

    # Find memory operations in captured queries
    memory_queries = [
        (name, q) for name, q in captured_queries
        if hasattr(q, 'query_class') and q.query_class == "memory"
    ]

    # ASSERT: Memory operations use the explicit user_id, not fallback
    for agent_name, query_obj in memory_queries:
        if query_obj.operation_type == "add_memory":
            metadata = query_obj.params.get("metadata", {})
            if "user_id" in metadata:
                assert metadata["user_id"] == explicit_user_id, (
                    f"create_memory used user_id={metadata['user_id']!r} "
                    f"instead of explicit {explicit_user_id!r}. "
                    f"Fallback chain overrode the explicit user_id."
                )
                # Must NOT be the agent_name
                assert metadata["user_id"] != "profile_agent", (
                    "create_memory used agent_name as user_id — identity leak"
                )


# ---------------------------------------------------------------------------
# Test 6b: search_memories with user_id=None does not send user_id to kernel
# ---------------------------------------------------------------------------

def test_search_memories_none_user_id_no_kernel_user_scope():
    """search_memories(user_id=None) must not include user_id in kernel params.

    This proves the SDK API layer does not inject a user_id from any
    fallback source (global state, latest writer, etc.) when the caller
    does not provide one.
    """
    captured_queries = []

    def mock_send_request(agent_name, query_obj, base_url):
        captured_queries.append(query_obj)
        return {"response": {"search_results": []}}

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        result = search_memories(
            agent_name="assistant_agent",
            query="some query",
            k=5,
            # user_id deliberately omitted (defaults to None)
        )

    assert len(captured_queries) == 1
    params = captured_queries[0].params

    # ASSERT: No user_id in params — SDK did not inject one
    assert "user_id" not in params, (
        f"search_memories(user_id=None) injected user_id={params.get('user_id')!r} "
        f"into kernel query params. No fallback identity should be used."
    )

    # ASSERT: Result is the standard empty response shape
    resp = result.get("response", {})
    assert resp.get("search_results") == []


def test_search_memories_whitespace_user_id_raises():
    """search_memories with whitespace-only user_id must raise ValueError.

    This ensures callers cannot accidentally pass empty/whitespace user_id
    and get a silent default behavior.
    """
    try:
        search_memories(
            agent_name="assistant_agent",
            query="test",
            user_id="   ",  # whitespace-only
        )
        assert False, "Expected ValueError for whitespace-only user_id"
    except ValueError as e:
        assert "non-empty" in str(e).lower() or "user_id" in str(e).lower()


def test_search_memories_strips_user_id_whitespace():
    """search_memories must strip whitespace from a valid user_id before
    passing it to the kernel.

    This ensures " alice " becomes "alice" in the kernel query params.
    """
    captured_queries = []

    def mock_send_request(agent_name, query_obj, base_url):
        captured_queries.append(query_obj)
        return {"response": {"search_results": []}}

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        search_memories(
            agent_name="assistant_agent",
            query="test",
            user_id="  alice_123  ",  # has leading/trailing whitespace
        )

    assert len(captured_queries) == 1
    params = captured_queries[0].params
    assert params["user_id"] == "alice_123", (
        f"Expected stripped user_id='alice_123', got {params['user_id']!r}. "
        f"SDK must normalize user_id before sending to kernel."
    )


# ---------------------------------------------------------------------------
# Test 7: ProfileAgent without self.user_id skips memory write entirely
# ---------------------------------------------------------------------------

def test_profile_agent_no_user_id_skips_memory_write():
    """ProfileAgent without self.user_id must skip memory write.

    When self.user_id is not set (or is None), ProfileAgent must NOT:
    - Use profile_data["user_name"] as user_id
    - Use self.agent_name ("profile_agent") as user_id
    - Call create_memory or update_memory at all

    The agent should still extract profile data (LLM call) but skip
    the memory upsert step entirely.
    """
    from cerebrum.example.agents.profile_agent.agent import ProfileAgent

    memory_calls = []

    def mock_send_request(agent_name, query_obj, base_url):
        if hasattr(query_obj, 'query_class'):
            if query_obj.query_class == "llm":
                return {
                    "response": {
                        "response_message": '{"user_name": "alice_from_llm", "preferred_tools": ["vim", "tmux"], "preferred_language": "rust", "response_style": "detailed"}'
                    }
                }
            elif query_obj.query_class == "memory":
                memory_calls.append((agent_name, query_obj.operation_type, query_obj.params))
                if query_obj.operation_type == "retrieve_memory":
                    return {"response": {"search_results": []}}
                elif query_obj.operation_type == "add_memory":
                    return {"response": {"memory_id": "mem_should_not_exist", "success": True}}
        return {"response": {}}

    # Instantiate WITHOUT setting self.user_id
    agent = ProfileAgent("profile_agent")
    # Deliberately do NOT set agent.user_id

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        with patch("cerebrum.llm.apis.send_request", mock_send_request):
            result = agent.run('{"user_name": "alice_from_llm", "preferred_tools": ["vim", "tmux"], "preferred_language": "rust", "response_style": "detailed"}')

    # ASSERT: No memory operations occurred
    assert len(memory_calls) == 0, (
        f"Expected 0 memory calls without self.user_id, but got {len(memory_calls)}: "
        f"{[(op, params.get('metadata', {}).get('user_id', 'N/A')) for _, op, params in memory_calls]}. "
        f"Agent derived user_id from profile data or agent_name."
    )

    # ASSERT: Agent returned a result (didn't crash)
    assert result["agent_name"] == "profile_agent"
    assert "Error" not in result["result"]


# ---------------------------------------------------------------------------
# Test 8: TaskAgent without self.user_id skips memory write entirely
# ---------------------------------------------------------------------------

def test_task_agent_no_user_id_skips_memory_write():
    """TaskAgent without self.user_id must skip memory write.

    When self.user_id is not set (or is None), TaskAgent must NOT:
    - Use context_data["current_project"] as user_id
    - Use self.agent_name ("task_agent") as user_id
    - Call create_memory or update_memory at all

    The agent should still extract task context (LLM call) but skip
    the memory upsert step entirely.
    """
    from cerebrum.example.agents.task_agent.agent import TaskAgent

    memory_calls = []

    def mock_send_request(agent_name, query_obj, base_url):
        if hasattr(query_obj, 'query_class'):
            if query_obj.query_class == "llm":
                return {
                    "response": {
                        "response_message": '{"current_project": "project_x", "active_experiment": "exp1", "goals": ["ship feature"], "blockers": ["ci broken"], "next_steps": ["fix ci"]}'
                    }
                }
            elif query_obj.query_class == "memory":
                memory_calls.append((agent_name, query_obj.operation_type, query_obj.params))
                if query_obj.operation_type == "retrieve_memory":
                    return {"response": {"search_results": []}}
                elif query_obj.operation_type == "add_memory":
                    return {"response": {"memory_id": "mem_should_not_exist", "success": True}}
        return {"response": {}}

    # Instantiate WITHOUT setting self.user_id
    agent = TaskAgent("task_agent")
    # Deliberately do NOT set agent.user_id

    with patch("cerebrum.memory.apis.send_request", mock_send_request):
        with patch("cerebrum.llm.apis.send_request", mock_send_request):
            result = agent.run('{"current_project": "project_x", "active_experiment": "exp1", "goals": ["ship feature"], "blockers": ["ci broken"], "next_steps": ["fix ci"]}')

    # ASSERT: No memory operations occurred
    assert len(memory_calls) == 0, (
        f"Expected 0 memory calls without self.user_id, but got {len(memory_calls)}: "
        f"{[(op, params.get('metadata', {}).get('user_id', 'N/A')) for _, op, params in memory_calls]}. "
        f"Agent derived user_id from task data or agent_name."
    )

    # ASSERT: Agent returned a result (didn't crash)
    assert result["agent_name"] == "task_agent"
    assert "Error" not in result["result"]


# ---------------------------------------------------------------------------
# Main runner (matches project convention)
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
    """Run all explicit user_id contract regression tests."""
    print("=" * 70)
    print("Explicit user_id Contract — Regression Tests")
    print("These tests lock in the requirement that Cerebrum NEVER retrieves")
    print("user-scoped memories without an explicit request-scoped user_id.")
    print("=" * 70)

    # Test 1
    print("\n--- Test 1: Retrieval without user_id returns empty ---")
    try:
        test_retrieval_without_user_id_returns_empty()
        record("retrieval_without_user_id_empty", True)
    except Exception as e:
        record("retrieval_without_user_id_empty", False, str(e)[:200])

    # Test 2
    print("\n--- Test 2: Retrieval with explicit user_id scopes correctly ---")
    try:
        test_retrieval_with_explicit_user_id_scopes_correctly()
        record("retrieval_with_explicit_user_id", True)
    except Exception as e:
        record("retrieval_with_explicit_user_id", False, str(e)[:200])

    # Test 3
    print("\n--- Test 3: No fallback to latest writer ---")
    try:
        test_no_fallback_to_latest_writer()
        record("no_fallback_to_latest_writer", True)
    except Exception as e:
        record("no_fallback_to_latest_writer", False, str(e)[:200])

    # Test 4
    print("\n--- Test 4: Context injection skips without user_id ---")
    try:
        test_context_injection_skips_without_user_id()
        record("context_injection_skips_without_user_id", True)
    except Exception as e:
        record("context_injection_skips_without_user_id", False, str(e)[:200])

    # Test 5
    print("\n--- Test 5: Context injection scopes to explicit user_id ---")
    try:
        test_context_injection_scopes_to_explicit_user_id()
        record("context_injection_scopes_to_user_id", True)
    except Exception as e:
        record("context_injection_scopes_to_user_id", False, str(e)[:200])

    # Test 6
    print("\n--- Test 6: ProfileAgent explicit user_id takes precedence ---")
    try:
        test_profile_agent_fallback_does_not_use_agent_name_for_retrieval()
        record("profile_agent_explicit_user_id_precedence", True)
    except Exception as e:
        record("profile_agent_explicit_user_id_precedence", False, str(e)[:200])

    # Test 6b: SDK-level guards
    print("\n--- Test 6b: search_memories(user_id=None) no kernel user scope ---")
    try:
        test_search_memories_none_user_id_no_kernel_user_scope()
        record("search_memories_none_user_id_no_scope", True)
    except Exception as e:
        record("search_memories_none_user_id_no_scope", False, str(e)[:200])

    print("\n--- Test 6c: search_memories whitespace user_id raises ---")
    try:
        test_search_memories_whitespace_user_id_raises()
        record("search_memories_whitespace_raises", True)
    except Exception as e:
        record("search_memories_whitespace_raises", False, str(e)[:200])

    print("\n--- Test 6d: search_memories strips user_id whitespace ---")
    try:
        test_search_memories_strips_user_id_whitespace()
        record("search_memories_strips_whitespace", True)
    except Exception as e:
        record("search_memories_strips_whitespace", False, str(e)[:200])

    # Test 7
    print("\n--- Test 7: ProfileAgent without user_id skips memory write ---")
    try:
        test_profile_agent_no_user_id_skips_memory_write()
        record("profile_agent_no_user_id_skips_write", True)
    except Exception as e:
        record("profile_agent_no_user_id_skips_write", False, str(e)[:200])

    # Test 8
    print("\n--- Test 8: TaskAgent without user_id skips memory write ---")
    try:
        test_task_agent_no_user_id_skips_memory_write()
        record("task_agent_no_user_id_skips_write", True)
    except Exception as e:
        record("task_agent_no_user_id_skips_write", False, str(e)[:200])

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
        print(f"\n{failed} test(s) FAILED — contract violations detected.")
        print("These failures confirm the production code needs updating")
        print("to enforce the explicit user_id contract.")
    else:
        print("\nAll tests PASSED — explicit user_id contract is enforced.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
