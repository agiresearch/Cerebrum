"""Unit tests for the naive_concat pipeline path isolation.

Verifies that:
- No create_memory, search_memories, or Mem0 calls are made
- retrieved_context_count == 0 in the result

Validates: Requirements 3.2, 3.3, 3.5
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock, call

from benchmarks.shared_memory.pipeline import MethodPipeline
from benchmarks.shared_memory.models import (
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
)


def _make_trial_data():
    """Return a minimal SyntheticTrialData for testing."""
    return SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="Alice Smith",
            preferred_tools=["vim", "tmux"],
            preferred_language="Python",
            response_style="concise",
        ),
        task_context=SyntheticTaskContext(
            current_project="data pipeline",
            active_experiment="batch processing",
            goals=["reduce latency", "improve throughput"],
            blockers=["memory bottleneck"],
            next_steps=["profile the code"],
        ),
        follow_up_query="How do I speed up my data pipeline?",
        user_id="alice_smith",
    )


def _mock_assistant_result():
    return {"agent_name": "assistant_agent", "result": "mocked response", "rounds": 1}


def test_no_create_memory_calls():
    """naive_concat must not call create_memory at all."""
    pipeline = MethodPipeline(method="naive_concat")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "cerebrum.memory.apis.create_memory"
    ) as mock_create:
        pipeline._run_naive_concat(trial_data, "alice_smith__naive_concat")

    mock_create.assert_not_called()
    print("PASSED: create_memory is never called in naive_concat")


def test_no_search_memories_calls():
    """naive_concat must not call search_memories at all."""
    pipeline = MethodPipeline(method="naive_concat")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "cerebrum.memory.apis.search_memories"
    ) as mock_search:
        pipeline._run_naive_concat(trial_data, "alice_smith__naive_concat")

    mock_search.assert_not_called()
    print("PASSED: search_memories is never called in naive_concat")


def test_no_mem0_memory_instantiation():
    """naive_concat must not instantiate or use any Mem0 Memory client."""
    pipeline = MethodPipeline(method="naive_concat")
    trial_data = _make_trial_data()

    # Patch mem0.Memory at the pipeline module level to detect any import/use
    mock_mem0_memory = MagicMock()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_mem0_memory)}):
        pipeline._run_naive_concat(trial_data, "alice_smith__naive_concat")

    # mem0.Memory should never be instantiated
    mock_mem0_memory.assert_not_called()
    print("PASSED: Mem0 Memory is never instantiated in naive_concat")


def test_retrieved_context_count_is_zero():
    """naive_concat must set retrieved_context_count=0 in the result."""
    pipeline = MethodPipeline(method="naive_concat")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ):
        result = pipeline._run_naive_concat(trial_data, "alice_smith__naive_concat")

    assert result.retrieved_context_count == 0, (
        f"Expected retrieved_context_count=0, got {result.retrieved_context_count}"
    )
    print("PASSED: retrieved_context_count == 0 in naive_concat result")


def test_retrieved_context_count_zero_via_run_trial():
    """MethodPipeline.run_trial with method=naive_concat also gives retrieved_context_count=0."""
    pipeline = MethodPipeline(method="naive_concat")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ):
        result = pipeline.run_trial(trial_data)

    assert result.retrieved_context_count == 0, (
        f"Expected retrieved_context_count=0 via run_trial, got {result.retrieved_context_count}"
    )
    assert result.method == "naive_concat", (
        f"Expected method='naive_concat', got {result.method!r}"
    )
    print("PASSED: run_trial with naive_concat gives retrieved_context_count=0")


def test_no_kernel_memory_api_calls_at_all():
    """naive_concat must not call any AIOS kernel memory API."""
    pipeline = MethodPipeline(method="naive_concat")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "cerebrum.memory.apis.create_memory"
    ) as mock_create, patch(
        "cerebrum.memory.apis.search_memories"
    ) as mock_search, patch(
        "cerebrum.memory.apis.get_memory"
    ) as mock_get, patch(
        "cerebrum.memory.apis.update_memory"
    ) as mock_update, patch(
        "cerebrum.memory.apis.delete_memory"
    ) as mock_delete:
        pipeline._run_naive_concat(trial_data, "alice_smith__naive_concat")

    mock_create.assert_not_called()
    mock_search.assert_not_called()
    mock_get.assert_not_called()
    mock_update.assert_not_called()
    mock_delete.assert_not_called()
    print("PASSED: no AIOS kernel memory API calls in naive_concat")


if __name__ == "__main__":
    test_no_create_memory_calls()
    test_no_search_memories_calls()
    test_no_mem0_memory_instantiation()
    test_retrieved_context_count_is_zero()
    test_retrieved_context_count_zero_via_run_trial()
    test_no_kernel_memory_api_calls_at_all()
    print("\nAll naive_concat unit tests passed.")
