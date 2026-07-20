"""Unit tests for kernel_shared path preservation.

Verifies that:
- AgentPipeline is called (not bypassed) when method is kernel_shared
- --condition phase1 maps to share_memory=False in the legacy path
- --condition phase2 maps to share_memory=True in the legacy path

Validates: Requirements 6.1, 6.2, 6.6
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

from benchmarks.shared_memory.pipeline import MethodPipeline, AgentPipeline
from benchmarks.shared_memory.models import (
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
    MemoryCounts,
    TrialResult,
    ConditionSummary,
    SummaryStatistics,
)


def _make_trial_data():
    """Return a minimal SyntheticTrialData for testing."""
    return SyntheticTrialData(
        profile=SyntheticProfile(
            user_name="Frank Green",
            preferred_tools=["docker", "kubernetes"],
            preferred_language="Go",
            response_style="technical",
        ),
        task_context=SyntheticTaskContext(
            current_project="microservices platform",
            active_experiment="service mesh",
            goals=["reduce latency", "improve reliability"],
            blockers=["network partitions"],
            next_steps=["add circuit breakers"],
        ),
        follow_up_query="How do I handle service failures gracefully?",
        user_id="frank_green",
    )


def _make_pipeline_result():
    """Return a minimal PipelineResult for mocking AgentPipeline.run_trial."""
    from benchmarks.shared_memory.pipeline import PipelineResult
    return PipelineResult(
        profile_result={},
        task_result={},
        assistant_result={"result": "mocked response"},
        assistant_response="mocked response",
        latency_seconds=1.0,
        method="kernel_shared",
        retrieved_context_count=None,
    )


def _make_trial_result(trial_index, condition):
    """Return a minimal non-failed TrialResult for mocking."""
    return TrialResult(
        trial_index=trial_index,
        condition=condition,
        profile_usage_score=3,
        task_usage_score=3,
        integration_score=3,
        memory_counts=MemoryCounts(total=2, shared=2, private=0),
        latency_seconds=1.0,
    )


def _make_condition_summary():
    """Return a minimal ConditionSummary for mocking."""
    s = SummaryStatistics(mean=0.0, std=0.0, min=0.0, max=0.0)
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


def test_agent_pipeline_is_called_for_kernel_shared():
    """MethodPipeline._run_kernel_shared must instantiate and call AgentPipeline."""
    pipeline = MethodPipeline(method="kernel_shared")
    trial_data = _make_trial_data()

    agent_pipeline_instances = []
    original_init = AgentPipeline.__init__

    def tracking_init(self, share_memory):
        agent_pipeline_instances.append(share_memory)
        original_init(self, share_memory)

    mock_pipeline_result = _make_pipeline_result()

    with patch.object(AgentPipeline, "__init__", tracking_init), \
         patch.object(AgentPipeline, "run_trial", return_value=mock_pipeline_result):
        result = pipeline._run_kernel_shared(trial_data, "frank_green__kernel_shared")

    assert len(agent_pipeline_instances) == 1, (
        f"Expected AgentPipeline to be instantiated once, got {len(agent_pipeline_instances)}"
    )
    assert result.method == "kernel_shared", (
        f"Expected method='kernel_shared', got {result.method!r}"
    )
    print("PASSED: AgentPipeline is instantiated and called for kernel_shared")


def test_kernel_shared_via_run_trial_calls_agent_pipeline():
    """MethodPipeline.run_trial with method=kernel_shared calls AgentPipeline."""
    pipeline = MethodPipeline(method="kernel_shared")
    trial_data = _make_trial_data()

    agent_pipeline_run_calls = []
    mock_pipeline_result = _make_pipeline_result()

    original_run = AgentPipeline.run_trial

    def tracking_run(self, td):
        agent_pipeline_run_calls.append(td.user_id)
        return mock_pipeline_result

    with patch.object(AgentPipeline, "run_trial", tracking_run):
        result = pipeline.run_trial(trial_data)

    assert len(agent_pipeline_run_calls) == 1, (
        f"Expected AgentPipeline.run_trial called once, got {len(agent_pipeline_run_calls)}"
    )
    assert result.method == "kernel_shared", (
        f"Expected method='kernel_shared', got {result.method!r}"
    )
    print("PASSED: AgentPipeline.run_trial called via MethodPipeline.run_trial")


def test_condition_phase1_maps_to_share_memory_false():
    """--condition phase1 must create AgentPipeline with share_memory=False."""
    from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

    with patch(
        "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
    ), patch(
        "benchmarks.shared_memory.run_evaluation.HybridJudge"
    ), patch(
        "benchmarks.shared_memory.run_evaluation.ResultsWriter"
    ) as mock_writer_cls:
        mock_writer_instance = mock_writer_cls.return_value
        mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
        mock_writer_instance.write_json.return_value = "/tmp/results.json"
        mock_writer_instance.write_csv.return_value = "/tmp/results.csv"

        orch = EvaluationOrchestrator(
            trials=1,
            output_dir="/tmp/test_output",
            write_csv=False,
            condition="phase1",
        )

    pipelines_created = []
    original_init = AgentPipeline.__init__

    def tracking_init(self, share_memory):
        pipelines_created.append(share_memory)
        original_init(self, share_memory)

    with patch(
        "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
        return_value="http://localhost:8000",
    ), patch.object(
        orch, "run_single_trial",
        return_value=_make_trial_result(0, "phase1"),
    ), patch(
        "benchmarks.shared_memory.run_evaluation.tqdm",
        side_effect=lambda iterable, **kw: iterable,
    ), patch.object(AgentPipeline, "__init__", tracking_init):
        orch.run()

    assert len(pipelines_created) == 1, (
        f"Expected 1 AgentPipeline, got {len(pipelines_created)}"
    )
    assert pipelines_created[0] is False, (
        f"phase1 must use share_memory=False, got {pipelines_created[0]}"
    )
    print("PASSED: --condition phase1 maps to share_memory=False")


def test_condition_phase2_maps_to_share_memory_true():
    """--condition phase2 must create AgentPipeline with share_memory=True."""
    from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

    with patch(
        "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
    ), patch(
        "benchmarks.shared_memory.run_evaluation.HybridJudge"
    ), patch(
        "benchmarks.shared_memory.run_evaluation.ResultsWriter"
    ) as mock_writer_cls:
        mock_writer_instance = mock_writer_cls.return_value
        mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
        mock_writer_instance.write_json.return_value = "/tmp/results.json"
        mock_writer_instance.write_csv.return_value = "/tmp/results.csv"

        orch = EvaluationOrchestrator(
            trials=1,
            output_dir="/tmp/test_output",
            write_csv=False,
            condition="phase2",
        )

    pipelines_created = []
    original_init = AgentPipeline.__init__

    def tracking_init(self, share_memory):
        pipelines_created.append(share_memory)
        original_init(self, share_memory)

    with patch(
        "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
        return_value="http://localhost:8000",
    ), patch.object(
        orch, "run_single_trial",
        return_value=_make_trial_result(0, "phase2"),
    ), patch(
        "benchmarks.shared_memory.run_evaluation.tqdm",
        side_effect=lambda iterable, **kw: iterable,
    ), patch.object(AgentPipeline, "__init__", tracking_init):
        orch.run()

    assert len(pipelines_created) == 1, (
        f"Expected 1 AgentPipeline, got {len(pipelines_created)}"
    )
    assert pipelines_created[0] is True, (
        f"phase2 must use share_memory=True, got {pipelines_created[0]}"
    )
    print("PASSED: --condition phase2 maps to share_memory=True")


def test_kernel_shared_method_user_id_scoping():
    """_run_kernel_shared passes method-scoped user_id to AgentPipeline."""
    pipeline = MethodPipeline(method="kernel_shared")
    trial_data = _make_trial_data()

    scoped_user_ids = []
    mock_pipeline_result = _make_pipeline_result()

    def tracking_run(self, td):
        scoped_user_ids.append(td.user_id)
        return mock_pipeline_result

    with patch.object(AgentPipeline, "run_trial", tracking_run):
        pipeline._run_kernel_shared(trial_data, "frank_green__kernel_shared")

    assert len(scoped_user_ids) == 1
    assert scoped_user_ids[0] == "frank_green__kernel_shared", (
        f"Expected method-scoped user_id='frank_green__kernel_shared', "
        f"got {scoped_user_ids[0]!r}"
    )
    print("PASSED: kernel_shared uses method-scoped user_id")


if __name__ == "__main__":
    test_agent_pipeline_is_called_for_kernel_shared()
    test_kernel_shared_via_run_trial_calls_agent_pipeline()
    test_condition_phase1_maps_to_share_memory_false()
    test_condition_phase2_maps_to_share_memory_true()
    test_kernel_shared_method_user_id_scoping()
    print("\nAll kernel_shared preservation tests passed.")
