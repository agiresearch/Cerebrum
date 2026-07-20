"""Property-based tests for unified trial data across methods.

# Feature: personalization-baselines, Property 9: Unified trial data across methods

When running ``--method all``, the EvaluationOrchestrator SHALL generate one
``SyntheticTrialData`` instance per trial index and reuse it across all four
method pipelines. This ensures that personalization score differences are
attributable to the method and not to variation in the input data.

**Validates: Requirements 7.1, 7.2**
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch, call
from typing import List

from hypothesis import given, settings
from hypothesis.strategies import composite, integers, text, lists, from_regex

from benchmarks.shared_memory.models import (
    ConditionSummary,
    MemoryCounts,
    SummaryStatistics,
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
    TrialResult,
)
from benchmarks.shared_memory.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zero_stats() -> SummaryStatistics:
    return SummaryStatistics(mean=0.0, std=0.0, min=0.0, max=0.0)


def _make_condition_summary() -> ConditionSummary:
    s = _make_zero_stats()
    return ConditionSummary(
        profile_usage=s,
        task_usage=s,
        integration=s,
        latency=s,
        memory_total=s,
        memory_shared=s,
        memory_private=s,
        injected_memories=s,
        total_trials=1,
        failed_trials=0,
    )


def _make_trial_result(trial_index: int, condition: str) -> TrialResult:
    return TrialResult(
        trial_index=trial_index,
        condition=condition,
        method=condition,
        profile_usage_score=3,
        task_usage_score=3,
        integration_score=3,
        memory_counts=MemoryCounts(),
        latency_seconds=0.1,
    )


def _make_pipeline_result() -> PipelineResult:
    """Return a minimal PipelineResult for mocking."""
    return PipelineResult(
        profile_result={},
        task_result={},
        assistant_result={"result": "mock response"},
        assistant_response="mock response",
        latency_seconds=0.1,
        retrieval_log=None,
        injection_diagnostics=None,
        written_memories=[],
        method="",
        retrieved_context_count=0,
    )


def _make_synthetic_trial_data(trial_index: int) -> SyntheticTrialData:
    """Build a deterministic SyntheticTrialData for a given trial index."""
    return SyntheticTrialData(
        profile=SyntheticProfile(
            user_name=f"User {trial_index}",
            preferred_tools=["tool_a", "tool_b"],
            preferred_language="Python",
            response_style="concise",
        ),
        task_context=SyntheticTaskContext(
            current_project=f"Project {trial_index}",
            active_experiment=f"Experiment {trial_index}",
            goals=[f"Goal {trial_index}"],
            blockers=[],
            next_steps=[f"Step {trial_index}"],
        ),
        follow_up_query=f"What should I do next for trial {trial_index}?",
        plausible_actions=[f"Action {trial_index}"],
        user_id=f"user_{trial_index}",
    )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@composite
def trial_count_strategy(draw):
    """Generate a small number of trials (1–5) for orchestrator-level tests."""
    return draw(integers(min_value=1, max_value=5))


# ---------------------------------------------------------------------------
# Property 9: Unified trial data across methods
# ---------------------------------------------------------------------------

# Feature: personalization-baselines, Property 9: Unified trial data across methods
class TestUnifiedTrialDataAcrossMethods:
    """**Validates: Requirements 7.1, 7.2**"""

    @given(num_trials=trial_count_strategy())
    @settings(max_examples=20)
    def test_same_trial_data_passed_to_all_methods(self, num_trials: int):
        """When running --method all, the same SyntheticTrialData instance
        (same profile, task context, follow-up query) SHALL be passed to each
        method pipeline for each trial index.

        We verify this by:
        1. Building an EvaluationOrchestrator with method="all".
        2. Mocking SyntheticDataGenerator.generate_trial_data to return
           deterministic SyntheticTrialData objects keyed by trial index.
        3. Mocking all four MethodPipeline.run_trial calls to capture the
           trial_data argument passed to each.
        4. Asserting that for each trial index, all four methods received the
           same SyntheticTrialData (same profile, task context, follow-up query).
        """
        from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

        # Build deterministic trial data for each index
        expected_trial_data = {
            i: _make_synthetic_trial_data(i) for i in range(num_trials)
        }

        # Track which trial_data was passed to each (method, trial_index) call
        calls_by_method: dict = {m: [] for m in ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]}

        def make_mock_run_trial(method_name: str):
            """Return a mock run_trial that records the trial_data argument."""
            def mock_run_trial(trial_data: SyntheticTrialData) -> PipelineResult:
                calls_by_method[method_name].append(trial_data)
                result = _make_pipeline_result()
                result.method = method_name
                return result
            return mock_run_trial

        # We need to intercept MethodPipeline instantiation to inject our mocks.
        # We track instances by method name.
        pipeline_instances: dict = {}

        original_method_pipeline_init = None

        from benchmarks.shared_memory import pipeline as pipeline_module

        original_init = pipeline_module.MethodPipeline.__init__
        original_run_trial = pipeline_module.MethodPipeline.run_trial

        def tracking_init(self, method: str, top_k: int = 3):
            original_init(self, method=method, top_k=top_k)
            pipeline_instances[method] = self
            # Monkey-patch run_trial on this instance
            self.run_trial = make_mock_run_trial(method)

        with patch.object(
            pipeline_module.MethodPipeline, "__init__", tracking_init
        ), patch(
            "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
        ) as mock_gen_cls, patch(
            "benchmarks.shared_memory.run_evaluation.HybridJudge"
        ) as mock_judge_cls, patch(
            "benchmarks.shared_memory.run_evaluation.ResultsWriter"
        ) as mock_writer_cls, patch(
            "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
            return_value="http://localhost:8000",
        ), patch(
            "benchmarks.shared_memory.run_evaluation.tqdm",
            side_effect=lambda iterable, **kw: iterable,
        ):
            # Configure mock generator to return deterministic trial data
            mock_gen_instance = mock_gen_cls.return_value
            mock_gen_instance.generate_trial_data.side_effect = (
                lambda i: expected_trial_data[i]
            )

            # Configure mock judge
            mock_judge_instance = mock_judge_cls.return_value
            mock_judge_instance.evaluate.return_value = MagicMock(
                profile_usage_score=3,
                task_usage_score=3,
                integration_score=3,
            )

            # Configure mock writer
            mock_writer_instance = mock_writer_cls.return_value
            mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
            mock_writer_instance.write_json.return_value = "/tmp/results_all_methods.json"
            mock_writer_instance.write_csv.return_value = "/tmp/results.csv"
            mock_writer_instance.print_summary.return_value = None

            orch = EvaluationOrchestrator(
                trials=num_trials,
                output_dir="/tmp/test_unified",
                write_csv=False,
                condition=None,
                method="all",
            )
            orch.run()

        # Verify all four methods were called
        all_methods = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]
        for method_name in all_methods:
            assert len(calls_by_method[method_name]) == num_trials, (
                f"Method '{method_name}' was called {len(calls_by_method[method_name])} times, "
                f"expected {num_trials}"
            )

        # Verify that for each trial index, all four methods received the
        # same SyntheticTrialData (same profile, task context, follow-up query)
        for trial_index in range(num_trials):
            expected = expected_trial_data[trial_index]
            for method_name in all_methods:
                actual = calls_by_method[method_name][trial_index]
                assert actual.profile == expected.profile, (
                    f"Trial {trial_index}, method '{method_name}': "
                    f"profile mismatch.\n"
                    f"  Expected: {expected.profile}\n"
                    f"  Got:      {actual.profile}"
                )
                assert actual.task_context == expected.task_context, (
                    f"Trial {trial_index}, method '{method_name}': "
                    f"task_context mismatch.\n"
                    f"  Expected: {expected.task_context}\n"
                    f"  Got:      {actual.task_context}"
                )
                assert actual.follow_up_query == expected.follow_up_query, (
                    f"Trial {trial_index}, method '{method_name}': "
                    f"follow_up_query mismatch.\n"
                    f"  Expected: {expected.follow_up_query!r}\n"
                    f"  Got:      {actual.follow_up_query!r}"
                )

    @given(num_trials=trial_count_strategy())
    @settings(max_examples=20)
    def test_generate_trial_data_called_once_per_trial_index(self, num_trials: int):
        """When running --method all, generate_trial_data SHALL be called
        exactly once per trial index (not once per method × trial).

        This verifies that the orchestrator caches the generated data and
        reuses it across all four methods rather than regenerating it.
        """
        from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

        expected_trial_data = {
            i: _make_synthetic_trial_data(i) for i in range(num_trials)
        }

        from benchmarks.shared_memory import pipeline as pipeline_module

        original_init = pipeline_module.MethodPipeline.__init__

        def tracking_init(self, method: str, top_k: int = 3):
            original_init(self, method=method, top_k=top_k)
            self.run_trial = lambda td: _make_pipeline_result()

        with patch.object(
            pipeline_module.MethodPipeline, "__init__", tracking_init
        ), patch(
            "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
        ) as mock_gen_cls, patch(
            "benchmarks.shared_memory.run_evaluation.HybridJudge"
        ) as mock_judge_cls, patch(
            "benchmarks.shared_memory.run_evaluation.ResultsWriter"
        ) as mock_writer_cls, patch(
            "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
            return_value="http://localhost:8000",
        ), patch(
            "benchmarks.shared_memory.run_evaluation.tqdm",
            side_effect=lambda iterable, **kw: iterable,
        ):
            mock_gen_instance = mock_gen_cls.return_value
            mock_gen_instance.generate_trial_data.side_effect = (
                lambda i: expected_trial_data[i]
            )

            mock_judge_instance = mock_judge_cls.return_value
            mock_judge_instance.evaluate.return_value = MagicMock(
                profile_usage_score=3,
                task_usage_score=3,
                integration_score=3,
            )

            mock_writer_instance = mock_writer_cls.return_value
            mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
            mock_writer_instance.write_json.return_value = "/tmp/results_all_methods.json"
            mock_writer_instance.print_summary.return_value = None

            orch = EvaluationOrchestrator(
                trials=num_trials,
                output_dir="/tmp/test_unified",
                write_csv=False,
                condition=None,
                method="all",
            )
            orch.run()

        # generate_trial_data should be called exactly num_trials times total,
        # not num_trials * 4 (once per method)
        actual_call_count = mock_gen_instance.generate_trial_data.call_count
        assert actual_call_count == num_trials, (
            f"generate_trial_data was called {actual_call_count} times, "
            f"expected exactly {num_trials} (once per trial index, not once per method)"
        )

    @given(num_trials=trial_count_strategy())
    @settings(max_examples=20)
    def test_output_filename_is_results_all_methods(self, num_trials: int):
        """When running --method all, the output filename SHALL be
        'results_all_methods.json'.
        """
        from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

        expected_trial_data = {
            i: _make_synthetic_trial_data(i) for i in range(num_trials)
        }

        from benchmarks.shared_memory import pipeline as pipeline_module

        original_init = pipeline_module.MethodPipeline.__init__

        def tracking_init(self, method: str, top_k: int = 3):
            original_init(self, method=method, top_k=top_k)
            self.run_trial = lambda td: _make_pipeline_result()

        with patch.object(
            pipeline_module.MethodPipeline, "__init__", tracking_init
        ), patch(
            "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
        ) as mock_gen_cls, patch(
            "benchmarks.shared_memory.run_evaluation.HybridJudge"
        ) as mock_judge_cls, patch(
            "benchmarks.shared_memory.run_evaluation.ResultsWriter"
        ) as mock_writer_cls, patch(
            "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
            return_value="http://localhost:8000",
        ), patch(
            "benchmarks.shared_memory.run_evaluation.tqdm",
            side_effect=lambda iterable, **kw: iterable,
        ):
            mock_gen_instance = mock_gen_cls.return_value
            mock_gen_instance.generate_trial_data.side_effect = (
                lambda i: expected_trial_data[i]
            )

            mock_judge_instance = mock_judge_cls.return_value
            mock_judge_instance.evaluate.return_value = MagicMock(
                profile_usage_score=3,
                task_usage_score=3,
                integration_score=3,
            )

            mock_writer_instance = mock_writer_cls.return_value
            mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
            mock_writer_instance.write_json.return_value = "/tmp/results_all_methods.json"
            mock_writer_instance.print_summary.return_value = None

            orch = EvaluationOrchestrator(
                trials=num_trials,
                output_dir="/tmp/test_unified",
                write_csv=False,
                condition=None,
                method="all",
            )
            orch.run()

        # Verify write_json was called with filename="results_all_methods.json"
        write_json_calls = mock_writer_instance.write_json.call_args_list
        assert len(write_json_calls) == 1, (
            f"write_json should be called exactly once, got {len(write_json_calls)}"
        )
        _, kwargs = write_json_calls[0]
        filename_arg = kwargs.get("filename") or (
            write_json_calls[0][0][1] if len(write_json_calls[0][0]) > 1 else None
        )
        assert filename_arg == "results_all_methods.json", (
            f"Expected filename='results_all_methods.json', got {filename_arg!r}"
        )

    @given(num_trials=trial_count_strategy())
    @settings(max_examples=20)
    def test_methods_run_populated_in_metadata(self, num_trials: int):
        """When running --method all, ExperimentMetadata.methods_run SHALL
        contain all four method names.
        """
        from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

        expected_trial_data = {
            i: _make_synthetic_trial_data(i) for i in range(num_trials)
        }

        from benchmarks.shared_memory import pipeline as pipeline_module

        original_init = pipeline_module.MethodPipeline.__init__

        def tracking_init(self, method: str, top_k: int = 3):
            original_init(self, method=method, top_k=top_k)
            self.run_trial = lambda td: _make_pipeline_result()

        with patch.object(
            pipeline_module.MethodPipeline, "__init__", tracking_init
        ), patch(
            "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
        ) as mock_gen_cls, patch(
            "benchmarks.shared_memory.run_evaluation.HybridJudge"
        ) as mock_judge_cls, patch(
            "benchmarks.shared_memory.run_evaluation.ResultsWriter"
        ) as mock_writer_cls, patch(
            "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
            return_value="http://localhost:8000",
        ), patch(
            "benchmarks.shared_memory.run_evaluation.tqdm",
            side_effect=lambda iterable, **kw: iterable,
        ):
            mock_gen_instance = mock_gen_cls.return_value
            mock_gen_instance.generate_trial_data.side_effect = (
                lambda i: expected_trial_data[i]
            )

            mock_judge_instance = mock_judge_cls.return_value
            mock_judge_instance.evaluate.return_value = MagicMock(
                profile_usage_score=3,
                task_usage_score=3,
                integration_score=3,
            )

            mock_writer_instance = mock_writer_cls.return_value
            mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
            mock_writer_instance.write_json.return_value = "/tmp/results_all_methods.json"
            mock_writer_instance.print_summary.return_value = None

            orch = EvaluationOrchestrator(
                trials=num_trials,
                output_dir="/tmp/test_unified",
                write_csv=False,
                condition=None,
                method="all",
            )
            experiment = orch.run()

        expected_methods = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]
        actual_methods = experiment.experiment_metadata.methods_run
        assert sorted(actual_methods) == sorted(expected_methods), (
            f"methods_run mismatch.\n"
            f"  Expected: {sorted(expected_methods)}\n"
            f"  Got:      {sorted(actual_methods)}"
        )


if __name__ == "__main__":
    test = TestUnifiedTrialDataAcrossMethods()

    print("Running Property 9: Unified trial data across methods...")
    print("  test_same_trial_data_passed_to_all_methods...")
    test.test_same_trial_data_passed_to_all_methods()
    print("  PASSED")

    print("  test_generate_trial_data_called_once_per_trial_index...")
    test.test_generate_trial_data_called_once_per_trial_index()
    print("  PASSED")

    print("  test_output_filename_is_results_all_methods...")
    test.test_output_filename_is_results_all_methods()
    print("  PASSED")

    print("  test_methods_run_populated_in_metadata...")
    test.test_methods_run_populated_in_metadata()
    print("  PASSED")

    print("\nAll Property 9 tests passed.")
