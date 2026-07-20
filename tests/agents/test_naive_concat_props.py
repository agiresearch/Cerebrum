"""Property-based tests for the naive_concat pipeline path using Hypothesis.

Feature: personalization-baselines
Property 3: naive_concat retrieved_context_count is always zero
Property 4: naive_concat prompt contains profile and task fields
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

from hypothesis import given, settings
from hypothesis.strategies import (
    text,
    lists,
    composite,
    just,
    integers,
    floats,
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
# Property 3: naive_concat retrieved_context_count is always zero
# ---------------------------------------------------------------------------

class TestNaiveConcatRetrievedContextCountIsZero:
    """Feature: personalization-baselines, Property 3: naive_concat retrieved_context_count is always zero

    **Validates: Requirements 3.5**
    """

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100)
    def test_retrieved_context_count_is_zero(self, trial_data):
        """For any SyntheticTrialData, _run_naive_concat must set retrieved_context_count=0."""
        pipeline = MethodPipeline(method="naive_concat")

        mock_result = {
            "agent_name": "assistant_agent",
            "result": "mocked response",
            "rounds": 1,
        }

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            return_value=mock_result,
        ):
            method_user_id = f"{trial_data.user_id}__naive_concat"
            result = pipeline._run_naive_concat(trial_data, method_user_id)

        assert result.retrieved_context_count == 0, (
            f"Expected retrieved_context_count=0, got {result.retrieved_context_count}"
        )

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100)
    def test_method_is_naive_concat(self, trial_data):
        """For any SyntheticTrialData, _run_naive_concat must set method='naive_concat'."""
        pipeline = MethodPipeline(method="naive_concat")

        mock_result = {
            "agent_name": "assistant_agent",
            "result": "mocked response",
            "rounds": 1,
        }

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            return_value=mock_result,
        ):
            method_user_id = f"{trial_data.user_id}__naive_concat"
            result = pipeline._run_naive_concat(trial_data, method_user_id)

        assert result.method == "naive_concat", (
            f"Expected method='naive_concat', got {result.method!r}"
        )


# ---------------------------------------------------------------------------
# Property 4: naive_concat prompt contains profile and task fields
# ---------------------------------------------------------------------------

class TestNaiveConcatPromptCompleteness:
    """Feature: personalization-baselines, Property 4: naive_concat prompt contains profile and task fields

    **Validates: Requirements 3.1**
    """

    @given(
        profile=synthetic_profile_strategy(),
        task_context=synthetic_task_context_strategy(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_user_name(self, profile, task_context):
        """The naive_concat prompt must contain the user's name."""
        trial_data = SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query="test query",
            user_id="test_user",
        )
        pipeline = MethodPipeline(method="naive_concat")

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "ok", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            pipeline._run_naive_concat(trial_data, "test_user__naive_concat")

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert profile.user_name in prompt, (
            f"Prompt missing user_name={profile.user_name!r}"
        )

    @given(
        profile=synthetic_profile_strategy(),
        task_context=synthetic_task_context_strategy(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_at_least_one_preferred_tool(self, profile, task_context):
        """The naive_concat prompt must contain at least one preferred tool."""
        trial_data = SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query="test query",
            user_id="test_user",
        )
        pipeline = MethodPipeline(method="naive_concat")

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "ok", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            pipeline._run_naive_concat(trial_data, "test_user__naive_concat")

        prompt = captured_prompts[0]
        assert any(tool in prompt for tool in profile.preferred_tools), (
            f"Prompt missing all preferred_tools={profile.preferred_tools!r}"
        )

    @given(
        profile=synthetic_profile_strategy(),
        task_context=synthetic_task_context_strategy(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_preferred_language(self, profile, task_context):
        """The naive_concat prompt must contain the preferred language."""
        trial_data = SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query="test query",
            user_id="test_user",
        )
        pipeline = MethodPipeline(method="naive_concat")

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "ok", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            pipeline._run_naive_concat(trial_data, "test_user__naive_concat")

        prompt = captured_prompts[0]
        assert profile.preferred_language in prompt, (
            f"Prompt missing preferred_language={profile.preferred_language!r}"
        )

    @given(
        profile=synthetic_profile_strategy(),
        task_context=synthetic_task_context_strategy(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_current_project(self, profile, task_context):
        """The naive_concat prompt must contain the current project name."""
        trial_data = SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query="test query",
            user_id="test_user",
        )
        pipeline = MethodPipeline(method="naive_concat")

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "ok", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            pipeline._run_naive_concat(trial_data, "test_user__naive_concat")

        prompt = captured_prompts[0]
        assert task_context.current_project in prompt, (
            f"Prompt missing current_project={task_context.current_project!r}"
        )

    @given(
        profile=synthetic_profile_strategy(),
        task_context=synthetic_task_context_strategy(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_at_least_one_goal(self, profile, task_context):
        """The naive_concat prompt must contain at least one goal."""
        trial_data = SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query="test query",
            user_id="test_user",
        )
        pipeline = MethodPipeline(method="naive_concat")

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "ok", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            pipeline._run_naive_concat(trial_data, "test_user__naive_concat")

        prompt = captured_prompts[0]
        assert any(goal in prompt for goal in task_context.goals), (
            f"Prompt missing all goals={task_context.goals!r}"
        )

    @given(
        profile=synthetic_profile_strategy(),
        task_context=synthetic_task_context_strategy(),
    )
    @settings(max_examples=100)
    def test_prompt_contains_section_headers(self, profile, task_context):
        """The naive_concat prompt must contain the required section headers."""
        trial_data = SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query="test query",
            user_id="test_user",
        )
        pipeline = MethodPipeline(method="naive_concat")

        captured_prompts = []

        def capture_run(self_agent, task_input):
            captured_prompts.append(task_input)
            return {"agent_name": "assistant_agent", "result": "ok", "rounds": 1}

        with patch(
            "benchmarks.shared_memory.pipeline.AssistantAgent.run",
            capture_run,
        ):
            pipeline._run_naive_concat(trial_data, "test_user__naive_concat")

        prompt = captured_prompts[0]
        assert "--- USER PROFILE ---" in prompt, "Prompt missing '--- USER PROFILE ---' header"
        assert "--- TASK CONTEXT ---" in prompt, "Prompt missing '--- TASK CONTEXT ---' header"
        assert "--- QUERY ---" in prompt, "Prompt missing '--- QUERY ---' header"


if __name__ == "__main__":
    t3 = TestNaiveConcatRetrievedContextCountIsZero()
    print("Running Property 3: naive_concat retrieved_context_count is always zero...")
    t3.test_retrieved_context_count_is_zero()
    print("PASSED: retrieved_context_count == 0 for all inputs")
    t3.test_method_is_naive_concat()
    print("PASSED: method == 'naive_concat' for all inputs")

    t4 = TestNaiveConcatPromptCompleteness()
    print("\nRunning Property 4: naive_concat prompt contains profile and task fields...")
    t4.test_prompt_contains_user_name()
    print("PASSED: prompt contains user_name")
    t4.test_prompt_contains_at_least_one_preferred_tool()
    print("PASSED: prompt contains at least one preferred_tool")
    t4.test_prompt_contains_preferred_language()
    print("PASSED: prompt contains preferred_language")
    t4.test_prompt_contains_current_project()
    print("PASSED: prompt contains current_project")
    t4.test_prompt_contains_at_least_one_goal()
    print("PASSED: prompt contains at least one goal")
    t4.test_prompt_contains_section_headers()
    print("PASSED: prompt contains all section headers")
