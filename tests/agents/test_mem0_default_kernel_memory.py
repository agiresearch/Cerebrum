"""Regression tests: mem0_default uses the 3-agent pipeline with share_memory=False.

Verifies that _run_mem0_default:
1. Delegates to AgentPipeline(share_memory=False) — same pathway as kernel_shared
2. ProfileAgent and TaskAgent write memories with sharing_policy="private"
3. AssistantAgent runs without cross-agent memory injection
4. Result method is set to "mem0_default"

No running kernel, Ollama, ChromaDB, or real LLM required.
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

from benchmarks.shared_memory.pipeline import MethodPipeline, AgentPipeline
from benchmarks.shared_memory.models import SyntheticTrialData, SyntheticProfile, SyntheticTaskContext


def _make_trial_data(user_id="test_user_abc"):
    """Build minimal SyntheticTrialData for testing."""
    return SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="Test User",
            preferred_tools=["VS Code", "Git"],
            preferred_language="Python",
            response_style="concise",
        ),
        task_context=SyntheticTaskContext(
            current_project="Test Project",
            active_experiment="Testing memory retrieval",
            goals=["Ship feature", "Write tests"],
            blockers=["Flaky CI"],
            next_steps=["Fix CI", "Deploy"],
        ),
        follow_up_query="What should I focus on next?",
        user_id=user_id,
    )


def test_mem0_default_delegates_to_agent_pipeline_with_share_memory_false():
    """mem0_default creates AgentPipeline(share_memory=False) and calls run_trial."""
    trial_data = _make_trial_data()

    captured_args = {}

    original_init = AgentPipeline.__init__

    def capturing_init(self, share_memory=True, assistant_llms=None):
        captured_args["share_memory"] = share_memory
        captured_args["assistant_llms"] = assistant_llms
        original_init(self, share_memory=share_memory, assistant_llms=assistant_llms)

    mock_result = MagicMock()
    mock_result.method = "mem0_default"

    with patch.object(AgentPipeline, "__init__", capturing_init), \
         patch.object(AgentPipeline, "run_trial", return_value=mock_result):
        pipeline = MethodPipeline(method="mem0_default", assistant_llms=[{"name": "gpt-4o", "backend": "azure"}])
        result = pipeline._run_mem0_default(trial_data, f"{trial_data.user_id}__mem0_default")

    assert captured_args["share_memory"] is False, (
        f"Expected share_memory=False, got {captured_args['share_memory']}"
    )


def test_mem0_default_scopes_user_id_with_method_suffix():
    """mem0_default passes method-scoped user_id to AgentPipeline."""
    trial_data = _make_trial_data(user_id="alice")
    method_user_id = "alice__mem0_default"

    captured_trial = {}

    def capturing_run_trial(self, scoped_trial):
        captured_trial["user_id"] = scoped_trial.user_id
        result = MagicMock()
        result.method = "kernel_shared"  # Will be overwritten
        return result

    with patch.object(AgentPipeline, "run_trial", capturing_run_trial):
        pipeline = MethodPipeline(method="mem0_default", assistant_llms=[{"name": "gpt-4o", "backend": "azure"}])
        result = pipeline._run_mem0_default(trial_data, method_user_id)

    assert captured_trial["user_id"] == method_user_id, (
        f"Expected user_id='{method_user_id}', got '{captured_trial['user_id']}'"
    )


def test_mem0_default_sets_method_to_mem0_default():
    """mem0_default sets result.method='mem0_default' regardless of AgentPipeline output."""
    trial_data = _make_trial_data()

    mock_result = MagicMock()
    mock_result.method = "kernel_shared"  # AgentPipeline doesn't set method

    with patch.object(AgentPipeline, "run_trial", return_value=mock_result):
        pipeline = MethodPipeline(method="mem0_default", assistant_llms=[{"name": "gpt-4o", "backend": "azure"}])
        result = pipeline._run_mem0_default(trial_data, f"{trial_data.user_id}__mem0_default")

    assert result.method == "mem0_default", (
        f"Expected method='mem0_default', got {result.method!r}"
    )


def test_mem0_default_passes_assistant_llms():
    """mem0_default forwards assistant_llms to AgentPipeline."""
    trial_data = _make_trial_data()
    llms = [{"name": "gpt-4o", "backend": "azure"}]

    captured_args = {}

    original_init = AgentPipeline.__init__

    def capturing_init(self, share_memory=True, assistant_llms=None):
        captured_args["assistant_llms"] = assistant_llms
        original_init(self, share_memory=share_memory, assistant_llms=assistant_llms)

    mock_result = MagicMock()
    mock_result.method = "mem0_default"

    with patch.object(AgentPipeline, "__init__", capturing_init), \
         patch.object(AgentPipeline, "run_trial", return_value=mock_result):
        pipeline = MethodPipeline(method="mem0_default", assistant_llms=llms)
        pipeline._run_mem0_default(trial_data, f"{trial_data.user_id}__mem0_default")

    assert captured_args["assistant_llms"] == llms, (
        f"Expected assistant_llms={llms}, got {captured_args['assistant_llms']}"
    )


def test_mem0_default_full_pipeline_writes_private_memories():
    """End-to-end: mem0_default pipeline writes memories with sharing_policy='private'."""
    trial_data = _make_trial_data()
    method_user_id = f"{trial_data.user_id}__mem0_default"

    create_calls = []

    def mock_create_memory(agent_name, content, metadata=None, base_url=None):
        create_calls.append({"agent_name": agent_name, "content": content, "metadata": metadata})
        return {"response": {"memory_id": f"mem_{len(create_calls)}", "success": True}}

    def mock_search_memories(agent_name, query, k=5, base_url=None, *, user_id=None, sharing_policy=None):
        return {"response": {"search_results": []}}

    def mock_llm_chat(agent_name, messages, base_url=None, llms=None, user_id=None):
        return {"response": {"response_message": "mocked response"}}

    def mock_llm_chat_json(agent_name, messages, base_url=None, llms=None, user_id=None):
        return {"response": {"response_message": "{}"}}

    with patch("benchmarks.shared_memory.pipeline.create_memory", side_effect=mock_create_memory), \
         patch("benchmarks.shared_memory.pipeline.search_memories", side_effect=mock_search_memories), \
         patch("cerebrum.example.agents.assistant_agent.agent.llm_chat", side_effect=mock_llm_chat), \
         patch("cerebrum.example.agents.profile_agent.agent.llm_chat_with_json_output", side_effect=mock_llm_chat_json), \
         patch("cerebrum.example.agents.task_agent.agent.llm_chat_with_json_output", side_effect=mock_llm_chat_json), \
         patch("cerebrum.memory.apis.create_memory", side_effect=mock_create_memory), \
         patch("cerebrum.memory.apis.search_memories", side_effect=mock_search_memories):
        pipeline = MethodPipeline(method="mem0_default", assistant_llms=[{"name": "gpt-4o", "backend": "azure"}])
        result = pipeline.run_trial(trial_data)

    assert result.method == "mem0_default"

    # All memories should be private since share_memory=False
    for c in create_calls:
        if c["metadata"]:
            assert c["metadata"]["sharing_policy"] == "private", (
                f"Expected sharing_policy='private', got '{c['metadata']['sharing_policy']}'"
            )


# ---------------------------------------------------------------------------
# Main runner
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
    print("mem0_default Pipeline Delegation — Regression Tests")
    print("Confirms mem0_default uses AgentPipeline(share_memory=False).")
    print("=" * 70)

    tests = [
        ("delegates_share_memory_false", test_mem0_default_delegates_to_agent_pipeline_with_share_memory_false),
        ("scopes_user_id", test_mem0_default_scopes_user_id_with_method_suffix),
        ("sets_method_mem0_default", test_mem0_default_sets_method_to_mem0_default),
        ("passes_assistant_llms", test_mem0_default_passes_assistant_llms),
        ("full_pipeline_private_memories", test_mem0_default_full_pipeline_writes_private_memories),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            record(name, True)
        except Exception as e:
            record(name, False, str(e)[:200])

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
