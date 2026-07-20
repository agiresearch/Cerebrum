"""Unit tests for the vanilla_rag pipeline path isolation.

Verifies that:
- No kernel memory or Mem0 calls are made
- Zero-overlap fallback injects all chunks when all TF-IDF scores are 0.0
- retrieved_context_count reflects the number of chunks injected

Validates: Requirements 4.5, 4.6, 4.8
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

from benchmarks.shared_memory.pipeline import MethodPipeline
from benchmarks.shared_memory.models import (
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
)


def _make_trial_data(follow_up_query="How do I speed up my data pipeline?"):
    """Return a minimal SyntheticTrialData for testing."""
    return SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="Bob Jones",
            preferred_tools=["emacs", "git"],
            preferred_language="Rust",
            response_style="detailed",
        ),
        task_context=SyntheticTaskContext(
            current_project="compiler optimization",
            active_experiment="LLVM passes",
            goals=["reduce binary size", "improve compile time"],
            blockers=["complex IR transformations"],
            next_steps=["benchmark with perf", "profile hot paths"],
        ),
        follow_up_query=follow_up_query,
        user_id="bob_jones",
    )


def _mock_assistant_result():
    return {"agent_name": "assistant_agent", "result": "mocked response", "rounds": 1}


def test_no_create_memory_calls():
    """vanilla_rag must not call create_memory at all."""
    pipeline = MethodPipeline(method="vanilla_rag")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "cerebrum.memory.apis.create_memory"
    ) as mock_create:
        pipeline._run_vanilla_rag(trial_data, "bob_jones__vanilla_rag")

    mock_create.assert_not_called()
    print("PASSED: create_memory is never called in vanilla_rag")


def test_no_search_memories_calls():
    """vanilla_rag must not call search_memories at all."""
    pipeline = MethodPipeline(method="vanilla_rag")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch(
        "cerebrum.memory.apis.search_memories"
    ) as mock_search:
        pipeline._run_vanilla_rag(trial_data, "bob_jones__vanilla_rag")

    mock_search.assert_not_called()
    print("PASSED: search_memories is never called in vanilla_rag")


def test_no_mem0_calls():
    """vanilla_rag must not instantiate or use any Mem0 Memory client."""
    pipeline = MethodPipeline(method="vanilla_rag")
    trial_data = _make_trial_data()

    mock_mem0_memory = MagicMock()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ), patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_mem0_memory)}):
        pipeline._run_vanilla_rag(trial_data, "bob_jones__vanilla_rag")

    mock_mem0_memory.assert_not_called()
    print("PASSED: Mem0 Memory is never instantiated in vanilla_rag")


def test_zero_overlap_fallback_injects_all_chunks():
    """When all TF-IDF scores are 0.0 (zero overlap), all chunks are injected.

    Uses a query that has zero lexical overlap with any chunk content.
    The query uses only characters/tokens that don't appear in the chunks.
    """
    pipeline = MethodPipeline(method="vanilla_rag")

    # Use a query with zero overlap: only digits and special chars not in chunks
    trial_data = _make_trial_data(follow_up_query="12345 67890 99999")

    # Override profile and task context with content that has no overlap with the query
    trial_data = SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="Carol White",
            preferred_tools=["bash"],
            preferred_language="Go",
            response_style="brief",
        ),
        task_context=SyntheticTaskContext(
            current_project="web server",
            active_experiment="load testing",
            goals=["handle concurrent requests"],
            blockers=["high memory usage"],
            next_steps=["add caching layer"],
        ),
        # Query with zero overlap: only numeric tokens not in any chunk
        follow_up_query="xyzxyz qqqqqq wwwwww",
        user_id="carol_white",
    )

    captured_prompts = []

    def capture_run(self_agent, task_input):
        captured_prompts.append(task_input)
        return _mock_assistant_result()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        capture_run,
    ):
        result = pipeline._run_vanilla_rag(trial_data, "carol_white__vanilla_rag")

    # Count expected chunks: 1 profile + 1 task + 1 goal + 1 blocker + 1 next_step = 5
    expected_chunk_count = (
        1  # profile block
        + 1  # task block
        + len(trial_data.task_context.goals)
        + len(trial_data.task_context.blockers)
        + len(trial_data.task_context.next_steps)
    )

    assert result.retrieved_context_count == expected_chunk_count, (
        f"Zero-overlap fallback: expected all {expected_chunk_count} chunks injected, "
        f"got retrieved_context_count={result.retrieved_context_count}"
    )
    print(f"PASSED: zero-overlap fallback injects all {expected_chunk_count} chunks")


def test_zero_overlap_fallback_with_numeric_query():
    """Numeric-only query has zero overlap with text chunks — all chunks injected."""
    pipeline = MethodPipeline(method="vanilla_rag")

    trial_data = SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="Dave Brown",
            preferred_tools=["nvim"],
            preferred_language="TypeScript",
            response_style="verbose",
        ),
        task_context=SyntheticTaskContext(
            current_project="frontend app",
            active_experiment="A/B testing",
            goals=["improve conversion rate"],
            blockers=["slow API responses"],
            next_steps=["add lazy loading"],
        ),
        # Pure numeric query — zero lexical overlap with text chunks
        follow_up_query="111 222 333 444 555",
        user_id="dave_brown",
    )

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ):
        result = pipeline._run_vanilla_rag(trial_data, "dave_brown__vanilla_rag")

    expected_chunk_count = (
        1  # profile block
        + 1  # task block
        + len(trial_data.task_context.goals)
        + len(trial_data.task_context.blockers)
        + len(trial_data.task_context.next_steps)
    )

    assert result.retrieved_context_count == expected_chunk_count, (
        f"Expected all {expected_chunk_count} chunks for zero-overlap query, "
        f"got {result.retrieved_context_count}"
    )
    print(f"PASSED: numeric query triggers zero-overlap fallback, all {expected_chunk_count} chunks injected")


def test_retrieved_context_count_at_most_top_k():
    """When there is overlap, retrieved_context_count <= top_k."""
    top_k = 2
    pipeline = MethodPipeline(method="vanilla_rag", top_k=top_k)

    # Query with overlap to trigger normal TF-IDF retrieval
    trial_data = _make_trial_data(
        follow_up_query="compiler optimization LLVM binary size"
    )

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ):
        result = pipeline._run_vanilla_rag(trial_data, "bob_jones__vanilla_rag")

    assert result.retrieved_context_count <= top_k, (
        f"Expected retrieved_context_count <= {top_k}, got {result.retrieved_context_count}"
    )
    print(f"PASSED: retrieved_context_count <= top_k={top_k}")


def test_method_is_vanilla_rag():
    """vanilla_rag path sets method='vanilla_rag' on the result."""
    pipeline = MethodPipeline(method="vanilla_rag")
    trial_data = _make_trial_data()

    with patch(
        "benchmarks.shared_memory.pipeline.AssistantAgent.run",
        return_value=_mock_assistant_result(),
    ):
        result = pipeline._run_vanilla_rag(trial_data, "bob_jones__vanilla_rag")

    assert result.method == "vanilla_rag", (
        f"Expected method='vanilla_rag', got {result.method!r}"
    )
    print("PASSED: method='vanilla_rag' set on result")


def test_no_kernel_memory_api_calls_at_all():
    """vanilla_rag must not call any AIOS kernel memory API."""
    pipeline = MethodPipeline(method="vanilla_rag")
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
        pipeline._run_vanilla_rag(trial_data, "bob_jones__vanilla_rag")

    mock_create.assert_not_called()
    mock_search.assert_not_called()
    mock_get.assert_not_called()
    mock_update.assert_not_called()
    mock_delete.assert_not_called()
    print("PASSED: no AIOS kernel memory API calls in vanilla_rag")


if __name__ == "__main__":
    test_no_create_memory_calls()
    test_no_search_memories_calls()
    test_no_mem0_calls()
    test_zero_overlap_fallback_injects_all_chunks()
    test_zero_overlap_fallback_with_numeric_query()
    test_retrieved_context_count_at_most_top_k()
    test_method_is_vanilla_rag()
    test_no_kernel_memory_api_calls_at_all()
    print("\nAll vanilla_rag unit tests passed.")
