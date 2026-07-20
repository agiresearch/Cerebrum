"""Property-based tests for the mem0_default pipeline path using Hypothesis.

Feature: personalization-baselines
Property 7: mem0_default delegates to AgentPipeline(share_memory=False)
Property 8: mem0_default method label is always set correctly
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import (
    text,
    lists,
    composite,
    from_regex,
)

from benchmarks.shared_memory.pipeline import MethodPipeline, AgentPipeline
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
# Property 7: mem0_default delegates to AgentPipeline(share_memory=False)
# ---------------------------------------------------------------------------

class TestMem0DefaultDelegation:
    """Feature: personalization-baselines, Property 7: mem0_default delegates correctly

    **Validates: Requirements 5.7**
    """

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_share_memory_is_false(self, trial_data):
        """For any SyntheticTrialData, mem0_default must create AgentPipeline
        with share_memory=False.
        """
        captured = {}

        original_init = AgentPipeline.__init__

        def capturing_init(self, share_memory=True, assistant_llms=None):
            captured["share_memory"] = share_memory
            original_init(self, share_memory=share_memory, assistant_llms=assistant_llms)

        mock_result = MagicMock()
        mock_result.method = "mem0_default"

        with patch.object(AgentPipeline, "__init__", capturing_init), \
             patch.object(AgentPipeline, "run_trial", return_value=mock_result):
            pipeline = MethodPipeline(method="mem0_default")
            method_user_id = f"{trial_data.user_id}__mem0_default"
            pipeline._run_mem0_default(trial_data, method_user_id)

        assert captured["share_memory"] is False, (
            f"Expected share_memory=False, got {captured['share_memory']}"
        )

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_user_id_is_scoped(self, trial_data):
        """For any SyntheticTrialData, the trial passed to AgentPipeline must
        have user_id set to the method_user_id (not the original).
        """
        method_user_id = f"{trial_data.user_id}__mem0_default"
        captured_trial = {}

        def capturing_run_trial(self, scoped_trial):
            captured_trial["user_id"] = scoped_trial.user_id
            result = MagicMock()
            result.method = "mem0_default"
            return result

        with patch.object(AgentPipeline, "run_trial", capturing_run_trial):
            pipeline = MethodPipeline(method="mem0_default")
            pipeline._run_mem0_default(trial_data, method_user_id)

        assert captured_trial["user_id"] == method_user_id, (
            f"Expected user_id='{method_user_id}', got '{captured_trial['user_id']}'"
        )

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_original_trial_data_unchanged(self, trial_data):
        """_run_mem0_default must not mutate the original trial_data."""
        original_user_id = trial_data.user_id
        method_user_id = f"{trial_data.user_id}__mem0_default"

        mock_result = MagicMock()
        mock_result.method = "mem0_default"

        with patch.object(AgentPipeline, "run_trial", return_value=mock_result):
            pipeline = MethodPipeline(method="mem0_default")
            pipeline._run_mem0_default(trial_data, method_user_id)

        assert trial_data.user_id == original_user_id, (
            f"Original trial_data.user_id was mutated: expected '{original_user_id}', "
            f"got '{trial_data.user_id}'"
        )


# ---------------------------------------------------------------------------
# Property 8: mem0_default method label
# ---------------------------------------------------------------------------

class TestMem0DefaultMethodLabel:
    """Feature: personalization-baselines, Property 8: mem0_default method label

    **Validates: Requirements 5.8**
    """

    @given(trial_data=synthetic_trial_data_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_method_is_mem0_default(self, trial_data):
        """For any SyntheticTrialData, _run_mem0_default must set method='mem0_default'."""
        mock_result = MagicMock()
        mock_result.method = "kernel_shared"  # Simulate AgentPipeline not setting it

        with patch.object(AgentPipeline, "run_trial", return_value=mock_result):
            pipeline = MethodPipeline(method="mem0_default")
            method_user_id = f"{trial_data.user_id}__mem0_default"
            result = pipeline._run_mem0_default(trial_data, method_user_id)

        assert result.method == "mem0_default", (
            f"Expected method='mem0_default', got {result.method!r}"
        )


if __name__ == "__main__":
    t7 = TestMem0DefaultDelegation()
    print("Running Property 7: mem0_default delegates to AgentPipeline(share_memory=False)...")
    t7.test_share_memory_is_false()
    print("PASSED: share_memory=False for all inputs")
    t7.test_user_id_is_scoped()
    print("PASSED: user_id is method-scoped for all inputs")
    t7.test_original_trial_data_unchanged()
    print("PASSED: original trial_data not mutated")

    t8 = TestMem0DefaultMethodLabel()
    print("\nRunning Property 8: mem0_default method label...")
    t8.test_method_is_mem0_default()
    print("PASSED: method == 'mem0_default' for all inputs")
