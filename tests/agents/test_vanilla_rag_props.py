"""Property-based tests for the vanilla_rag pipeline path using Hypothesis.

Feature: personalization-baselines
Property 5: vanilla_rag retrieves at most top-k chunks
Property 6: vanilla_rag retrieved_context_count matches injected chunk count
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch

from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import (
    text,
    lists,
    integers,
    composite,
    from_regex,
)

from benchmarks.shared_memory.pipeline import MethodPipeline
from benchmarks.shared_memory.models import (
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty printable text (avoid empty strings for names/fields)
_nonempty_text = text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
    min_size=1,
    max_size=30,
)

_nonempty_list = lists(_nonempty_text, min_size=1, max_size=5)


@composite
def synthetic_profile_strategy(draw):
    """Generate a random SyntheticProfile."""
    return SyntheticProfile(
        user_name=draw(_nonempty_text),
        preferred_tools=draw(_nonempty_list),
        preferred_language=draw(_nonempty_text),
        response_style=draw(_nonempty_text),
    )


@composite
def synthetic_task_context_strategy(draw):
    """Generate a random SyntheticTaskContext."""
    return SyntheticTaskContext(
        current_project=draw(_nonempty_text),
        active_experiment=draw(_nonempty_text),
        goals=draw(_nonempty_list),
        blockers=draw(_nonempty_list),
        next_steps=draw(_nonempty_list),
    )


@composite
def synthetic_trial_data_strategy(draw):
    """Generate a random SyntheticTrialData."""
    return SyntheticTrialData(
        profile=draw(synthetic_profile_strategy()),
        task_context=draw(synthetic_task_context_strategy()),
        follow_up_query=draw(_nonempty_text),
        plausible_actions=draw(lists(_nonempty_text, min_size=0, max_size=3)),
        user_id=draw(from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)),
    )


# ---------------------------------------------------------------------------
# Property 5: vanilla_rag retrieves at most top-k chunks
# ---------------------------------------------------------------------------

class TestVanillaRagTopKBound:
    """Feature: personalization-baselines, Property 5: vanilla_rag retrieves at most top-k chunks

    **Validates: Requirements 4.3**
    """

    @given(
        trial_data=synthetic_trial_data_strategy(),
        top_k=integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_retrieved_at_most_top_k(self, trial_data, top_k):
        """For any trial data and any k, the number of injected chunks must be <= k,
        UNLESS all similarity scores are 0.0 (zero-overlap fallback), in which case
        all chunks are injected per Requirement 4.8.

        Generates random chunk lists (derived from random SyntheticTrialData) and
        queries, verifies len(retrieved) <= k when there is non-zero similarity,
        or len(retrieved) == total_chunks when all scores are 0.0.
        """
        # Feature: personalization-baselines, Property 5: vanilla_rag retrieves at most top-k chunks
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

        pipeline = MethodPipeline(method="vanilla_rag", top_k=top_k)

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "mocked response", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            method_user_id = f"{trial_data.user_id}__vanilla_rag"
            result = pipeline._run_vanilla_rag(trial_data, method_user_id)

        # Reconstruct the total chunk count to determine if zero-overlap fallback applies
        profile = trial_data.profile
        task_context = trial_data.task_context
        total_chunks = (
            2  # profile block + task block
            + len(task_context.goals)
            + len(task_context.blockers)
            + len(task_context.next_steps)
        )

        # Determine if zero-overlap fallback was triggered by checking similarity
        chunks = []
        chunks.append(
            f"User Profile:\nName: {profile.user_name}\n"
            f"Preferred Tools: {', '.join(profile.preferred_tools)}\n"
            f"Preferred Language: {profile.preferred_language}\n"
            f"Response Style: {profile.response_style}"
        )
        chunks.append(
            f"Task Context:\nCurrent Project: {task_context.current_project}\n"
            f"Active Experiment: {task_context.active_experiment}"
        )
        for goal in task_context.goals:
            chunks.append(f"Goal: {goal}")
        for blocker in task_context.blockers:
            chunks.append(f"Blocker: {blocker}")
        for next_step in task_context.next_steps:
            chunks.append(f"Next Step: {next_step}")

        vectorizer = TfidfVectorizer()
        chunk_matrix = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([trial_data.follow_up_query])
        scores = sk_cosine_similarity(query_vector, chunk_matrix).flatten()
        all_zero = scores.max() == 0.0

        if all_zero:
            # Zero-overlap fallback: all chunks are injected (Requirement 4.8)
            assert result.retrieved_context_count == total_chunks, (
                f"Zero-overlap fallback: expected all {total_chunks} chunks injected, "
                f"got {result.retrieved_context_count}"
            )
        else:
            # Normal case: at most top_k chunks are injected (Requirement 4.3)
            assert result.retrieved_context_count <= top_k, (
                f"Expected retrieved_context_count <= {top_k}, "
                f"got {result.retrieved_context_count}"
            )


# ---------------------------------------------------------------------------
# Property 6: vanilla_rag retrieved_context_count matches injected chunk count
# ---------------------------------------------------------------------------

class TestVanillaRagCountConsistency:
    """Feature: personalization-baselines, Property 6: vanilla_rag retrieved_context_count matches injected chunk count

    **Validates: Requirements 4.7**
    """

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100)
    def test_retrieved_context_count_matches_injected(self, trial_data):
        """retrieved_context_count must equal the number of chunks actually injected.

        Runs _run_vanilla_rag with a mocked AssistantAgent.run and verifies that
        retrieved_context_count == the number of chunks present in the injected
        context block.
        """
        # Feature: personalization-baselines, Property 6: vanilla_rag retrieved_context_count matches injected chunk count
        pipeline = MethodPipeline(method="vanilla_rag", top_k=3)

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "mocked response", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            method_user_id = f"{trial_data.user_id}__vanilla_rag"
            result = pipeline._run_vanilla_rag(trial_data, method_user_id)

        assert len(captured_prompts) == 1, "AssistantAgent.run should be called exactly once"
        prompt = captured_prompts[0]

        # Extract the context block between the two section markers
        assert "--- RETRIEVED CONTEXT ---" in prompt, (
            "Prompt missing '--- RETRIEVED CONTEXT ---' header"
        )
        assert "--- QUERY ---" in prompt, (
            "Prompt missing '--- QUERY ---' header"
        )

        # Parse the injected chunks from the prompt: they are separated by "\n\n"
        # between the two markers
        context_start = prompt.index("--- RETRIEVED CONTEXT ---") + len("--- RETRIEVED CONTEXT ---\n")
        context_end = prompt.index("\n\n--- QUERY ---")
        context_block = prompt[context_start:context_end]

        # Chunks are separated by "\n\n" within the context block
        injected_chunks = context_block.split("\n\n")
        injected_count = len(injected_chunks)

        assert result.retrieved_context_count == injected_count, (
            f"retrieved_context_count={result.retrieved_context_count} does not match "
            f"injected chunk count={injected_count}"
        )

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100)
    def test_method_is_vanilla_rag(self, trial_data):
        """For any SyntheticTrialData, _run_vanilla_rag must set method='vanilla_rag'."""
        # Feature: personalization-baselines, Property 6: vanilla_rag retrieved_context_count matches injected chunk count
        pipeline = MethodPipeline(method="vanilla_rag", top_k=3)

        mock_result = {
            "agent_name": "assistant_agent",
            "result": "mocked response",
            "rounds": 1,
        }

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            return_value=mock_result,
        ):
            method_user_id = f"{trial_data.user_id}__vanilla_rag"
            result = pipeline._run_vanilla_rag(trial_data, method_user_id)

        assert result.method == "vanilla_rag", (
            f"Expected method='vanilla_rag', got {result.method!r}"
        )


if __name__ == "__main__":
    t5 = TestVanillaRagTopKBound()
    print("Running Property 5: vanilla_rag retrieves at most top-k chunks...")
    t5.test_retrieved_at_most_top_k()
    print("PASSED: retrieved_context_count <= top_k for all inputs")

    t6 = TestVanillaRagCountConsistency()
    print("\nRunning Property 6: vanilla_rag retrieved_context_count matches injected chunk count...")
    t6.test_retrieved_context_count_matches_injected()
    print("PASSED: retrieved_context_count == injected chunk count for all inputs")
    t6.test_method_is_vanilla_rag()
    print("PASSED: method == 'vanilla_rag' for all inputs")
