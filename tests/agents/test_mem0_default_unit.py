"""Unit tests for the mem0_default pipeline path isolation.

Verifies that:
- No kernel memory calls (create_memory, search_memories) are made
- add is called before search (call order matters)
- Mem0 exception sets retrieved_context_count=0 without failing the trial

Validates: Requirements 5.5, 5.6, 5.8
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
            user_name="Eve Davis",
            preferred_tools=["jupyter", "pandas"],
            preferred_language="Python",
            response_style="analytical",
        ),
        task_context=SyntheticTaskContext(
            current_project="ML pipeline",
            active_experiment="feature engineering",
            goals=["improve model accuracy", "reduce training time"],
            blockers=["data quality issues"],
            next_steps=["clean the dataset", "run cross-validation"],
        ),
        follow_up_query="How do I handle missing values in my dataset?",
        user_id="eve_davis",
    )


def _mock_assistant_result():
    return {"agent_name": "assistant_agent", "result": "mocked response", "rounds": 1}


def test_no_create_memory_calls():
    """mem0_default must not call AIOS kernel create_memory."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    mock_memory = MagicMock()
    mock_memory.add.return_value = None
    mock_memory.search.return_value = []

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "mem0.Memory", return_value=mock_memory
    ), patch(
        "cerebrum.memory.apis.create_memory"
    ) as mock_create:
        pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    mock_create.assert_not_called()
    print("PASSED: AIOS create_memory is never called in mem0_default")


def test_no_search_memories_calls():
    """mem0_default must not call AIOS kernel search_memories."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    mock_memory = MagicMock()
    mock_memory.add.return_value = None
    mock_memory.search.return_value = []

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "mem0.Memory", return_value=mock_memory
    ), patch(
        "cerebrum.memory.apis.search_memories"
    ) as mock_search:
        pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    mock_search.assert_not_called()
    print("PASSED: AIOS search_memories is never called in mem0_default")


def test_add_called_before_search():
    """mem0_default must call memory.add before memory.search (call order matters)."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    call_order = []

    mock_memory = MagicMock()

    def tracking_add(text, user_id):
        call_order.append("add")
        return None

    def tracking_search(query, user_id):
        call_order.append("search")
        return []

    mock_memory.add.side_effect = tracking_add
    mock_memory.search.side_effect = tracking_search

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch("mem0.Memory", return_value=mock_memory):
        pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    # add must be called at least once before search
    assert "add" in call_order, "memory.add was never called"
    assert "search" in call_order, "memory.search was never called"

    first_add_idx = call_order.index("add")
    first_search_idx = call_order.index("search")
    assert first_add_idx < first_search_idx, (
        f"Expected add before search, but call order was: {call_order}"
    )
    print(f"PASSED: add called before search (order: {call_order})")


def test_add_called_twice_before_search():
    """mem0_default calls add twice (profile + task) before calling search."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    call_order = []

    mock_memory = MagicMock()

    def tracking_add(text, user_id):
        call_order.append("add")
        return None

    def tracking_search(query, user_id):
        call_order.append("search")
        return []

    mock_memory.add.side_effect = tracking_add
    mock_memory.search.side_effect = tracking_search

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch("mem0.Memory", return_value=mock_memory):
        pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    add_count = call_order.count("add")
    search_count = call_order.count("search")

    assert add_count == 2, f"Expected 2 add calls (profile + task), got {add_count}"
    assert search_count == 1, f"Expected 1 search call, got {search_count}"

    # Both adds must come before search
    add_indices = [i for i, c in enumerate(call_order) if c == "add"]
    search_idx = call_order.index("search")
    assert all(idx < search_idx for idx in add_indices), (
        f"Not all add calls precede search. Order: {call_order}"
    )
    print(f"PASSED: add called twice before search (order: {call_order})")


def test_add_exception_sets_retrieved_context_count_zero():
    """When memory.add raises, retrieved_context_count must be 0."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    mock_memory = MagicMock()
    mock_memory.add.side_effect = RuntimeError("Mem0 add failed")

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch("mem0.Memory", return_value=mock_memory):
        result = pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    assert result.retrieved_context_count == 0, (
        f"Expected retrieved_context_count=0 after add exception, "
        f"got {result.retrieved_context_count}"
    )
    print("PASSED: add exception sets retrieved_context_count=0")


def test_search_exception_sets_retrieved_context_count_zero():
    """When memory.search raises, retrieved_context_count must be 0."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    mock_memory = MagicMock()
    mock_memory.add.return_value = None
    mock_memory.search.side_effect = ConnectionError("Mem0 search failed")

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch("mem0.Memory", return_value=mock_memory):
        result = pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    assert result.retrieved_context_count == 0, (
        f"Expected retrieved_context_count=0 after search exception, "
        f"got {result.retrieved_context_count}"
    )
    print("PASSED: search exception sets retrieved_context_count=0")


def test_exception_does_not_fail_trial():
    """Mem0 exception must not cause the pipeline to raise — result is returned."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    mock_memory = MagicMock()
    mock_memory.add.side_effect = ValueError("Mem0 add error")

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch("mem0.Memory", return_value=mock_memory):
        # Must not raise — result must be returned
        result = pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    assert result is not None, "Expected a PipelineResult, got None"
    assert result.method == "mem0_default", (
        f"Expected method='mem0_default', got {result.method!r}"
    )
    assert result.assistant_response == "mocked response", (
        "AssistantAgent.run should still be called after Mem0 exception"
    )
    print("PASSED: Mem0 exception does not fail the trial — result is returned")


def test_exception_does_not_prevent_assistant_run():
    """When Mem0 raises, AssistantAgent.run must still be called."""
    pipeline = MethodPipeline(method="mem0_default")
    trial_data = _make_trial_data()

    mock_memory = MagicMock()
    mock_memory.add.side_effect = TimeoutError("Mem0 timeout")

    assistant_calls = []

    def counting_run(self_agent, task_input):
        assistant_calls.append(task_input)
        return _mock_assistant_result()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        counting_run,
    ), patch("mem0.Memory", return_value=mock_memory):
        pipeline._run_mem0_default(trial_data, "eve_davis__mem0_default")

    assert len(assistant_calls) == 1, (
        f"Expected AssistantAgent.run called once, got {len(assistant_calls)}"
    )
    print("PASSED: AssistantAgent.run called even when Mem0 raises")


if __name__ == "__main__":
    test_no_create_memory_calls()
    test_no_search_memories_calls()
    test_add_called_before_search()
    test_add_called_twice_before_search()
    test_add_exception_sets_retrieved_context_count_zero()
    test_search_exception_sets_retrieved_context_count_zero()
    test_exception_does_not_fail_trial()
    test_exception_does_not_prevent_assistant_run()
    print("\nAll mem0_default unit tests passed.")
