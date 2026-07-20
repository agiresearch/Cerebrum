"""CLI entry point and orchestrator for the shared memory evaluation harness.

Runs a two-condition experiment comparing private-only memory (Phase 1)
against shared memory (Phase 2) across synthetic trials, collecting
profile-usage, task-usage, integration, memory-count, and latency metrics.

The kernel's ``memory.auto_inject`` is assumed to be enabled for both phases.
The only difference between phases is the ``share_memory`` flag on agents:
- Phase 1: agents write memories with sharing_policy="private" → kernel
  auto-inject finds nothing eligible to inject cross-agent.
- Phase 2: agents write memories with sharing_policy="shared" → kernel
  auto-inject retrieves and injects them into AssistantAgent's context.

Restart the kernel between phases to clear the memory store and prevent
rollover.

Usage::

    # Legacy --condition workflow (backward compatible)
    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/ --condition phase1 --csv
    # restart kernel to clear memory
    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/ --condition phase2 --csv

    # New --method workflow
    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/ --method naive_concat
    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/ --method all --csv
"""

import argparse
import logging
import os
import statistics
import sys
from datetime import datetime

# Ensure the project root is on sys.path when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tqdm import tqdm

from benchmarks.shared_memory.models import (
    ConditionResults,
    ExperimentMetadata,
    ExperimentResults,
    JudgeScores,
    MemoryCounts,
    TrialResult,
)
from benchmarks.shared_memory.judge import HybridJudge
from benchmarks.shared_memory.pipeline import AgentPipeline, MethodPipeline
from benchmarks.shared_memory.results import ResultsWriter
from benchmarks.shared_memory.synth import SyntheticDataGenerator
from cerebrum.config.config_manager import config

logger = logging.getLogger(__name__)

# All valid method names (excluding the meta-value "all")
_ALL_METHODS = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default"]


def _parse_model_arg(model_str: str | None) -> list | None:
    """Parse a 'name:backend' model string into an llms list.

    Accepts formats:
    - ``"qwen2.5:7b:ollama"``  → ``[{"name": "qwen2.5:7b", "backend": "ollama"}]``
    - ``"gpt-4.5:openai"``     → ``[{"name": "gpt-4.5", "backend": "openai"}]``
    - ``None``                 → ``None`` (use kernel default)

    The last colon-separated token is treated as the backend; everything
    before it is the model name.
    """
    if model_str is None:
        return None
    parts = model_str.rsplit(":", 1)
    if len(parts) == 2:
        name, backend = parts
        return [{"name": name, "backend": backend}]
    # No backend specified — pass name only
    return [{"name": model_str}]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Extracted as a standalone function so it can be imported and tested
    independently of the ``main()`` entry point.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Shared Memory Evaluation Harness — measures whether "
        "shared memory improves personalization quality.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per condition (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for result files (default: results/).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV file alongside the JSON results.",
    )
    parser.add_argument(
        "--condition",
        choices=["both", "phase1", "phase2"],
        default=None,
        help=(
            "Legacy condition flag. When specified, maps phase1 → kernel_shared "
            "with share_memory=False and phase2 → kernel_shared with "
            "share_memory=True. Use --method for the new workflow."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default", "all"],
        default="kernel_shared",
        help=(
            "Personalization method to evaluate (default: kernel_shared). "
            "Use 'all' to run all four methods on the same trial data."
        ),
    )
    parser.add_argument(
        "--kernel-url",
        type=str,
        default=None,
        help=(
            "AIOS kernel base URL (e.g. http://localhost:8001). "
            "Overrides the CEREBRUM_KERNEL_URL env var and config default."
        ),
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default=None,
        help=(
            "LLM for the assistant and synthetic data generation, as "
            "'name:backend' (e.g. 'qwen2.5:7b:ollama' or 'gpt-4o-mini:openai'). "
            "Defaults to the kernel's currently selected model."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "LLM for the judge, as 'name:backend' "
            "(e.g. 'gpt-4.5:openai' or 'qwen2.5:7b:ollama'). "
            "Defaults to the same model as --assistant-model."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run-specific discriminator appended to user_id for namespace "
            "isolation. Prevents residual memories from prior runs from "
            "polluting current run results. When not specified, a random "
            "UUID suffix is generated automatically."
        ),
    )
    parser.add_argument(
        "--kernel-logs",
        type=str,
        default=None,
        help=(
            "Path to captured AIOS kernel stdout file. When provided, the "
            "orchestrator parses injection log lines and cross-references "
            "them with harness-side audit results for ground truth verification."
        ),
    )
    return parser


class EvaluationOrchestrator:
    """Orchestrates the shared memory evaluation experiment.

    Args:
        trials: Number of trials to run per condition.
        output_dir: Directory path for writing result files.
        write_csv: If True, also write a CSV file alongside JSON.
        condition: Legacy condition flag — "both", "phase1", "phase2", or None.
            When set, the orchestrator uses the legacy AgentPipeline path for
            backward compatibility.
        method: Method to evaluate — one of "kernel_shared", "naive_concat",
            "vanilla_rag", "mem0_default", or "all". Ignored when ``condition``
            is set.
    """

    def __init__(
        self,
        trials: int,
        output_dir: str,
        write_csv: bool,
        condition: str,
        method: str = "kernel_shared",
        assistant_llms: list | None = None,
        judge_llms: list | None = None,
        kernel_url: str | None = None,
        run_id: str | None = None,
        kernel_logs_path: str | None = None,
    ):
        self.trials = trials
        self.output_dir = output_dir
        self.write_csv = write_csv
        self.condition = condition
        self.method = method
        self.assistant_llms = assistant_llms  # passed to AssistantAgent and SyntheticDataGenerator
        self.judge_llms = judge_llms or assistant_llms  # falls back to assistant model if not set
        self.run_id = run_id  # None → SyntheticDataGenerator auto-generates UUID

        # Resolve kernel URL: CLI arg > config default
        resolved_url = kernel_url or config.get_kernel_url()
        self.kernel_url = resolved_url

        self.generator = SyntheticDataGenerator(llms=self.assistant_llms, run_id=self.run_id)
        self.generator.kernel_url = resolved_url

        self.judge = HybridJudge(llms=self.judge_llms)
        self.judge.llm_judge.kernel_url = resolved_url

        self.writer = ResultsWriter(output_dir=output_dir, write_csv=write_csv)

        # Kernel log cross-reference: deferred to after trials complete.
        # The log file is written by the kernel DURING the run, so we parse
        # it after all trials finish and retroactively update injection_status.
        self.kernel_logs_path = kernel_logs_path
        self.kernel_log_counts: dict[str, int] | None = None

    def _cross_reference_kernel_logs(self, trials: list[TrialResult]) -> None:
        """Parse kernel logs after trials complete and update injection_status.

        The kernel appends injection lines to the log file DURING the run,
        so this must be called AFTER all trials finish. It retroactively
        updates each trial's retrieval_log.injection_status based on the
        kernel's ground truth.
        """
        if not self.kernel_logs_path:
            return

        from benchmarks.shared_memory.kernel_log_parser import parse_injection_lines
        try:
            self.kernel_log_counts = parse_injection_lines(self.kernel_logs_path)
            logger.info(
                "Post-run kernel log cross-reference: %d user_ids with injection data from %s",
                len(self.kernel_log_counts), self.kernel_logs_path,
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                "Failed to load kernel logs from %s: %s. "
                "Skipping post-run cross-reference.",
                self.kernel_logs_path, e,
            )
            return

        for trial in trials:
            if trial.failed or not trial.retrieval_log:
                continue
            # Look up user_id from the trial's synthetic data
            user_id = getattr(trial, "_user_id", None)
            if not user_id and trial.synthetic_profile:
                # Reconstruct from profile — but we don't have run_id here.
                # Use retrieval_log metadata or fall back to scanning all keys.
                pass
            # Try matching by checking all kernel log user_ids against trial index
            # The most reliable approach: check if any kernel log user_id
            # is a substring match for this trial's profile name.
            profile_name = ""
            if trial.synthetic_profile:
                profile_name = trial.synthetic_profile.user_name.lower().replace(" ", "_")

            kernel_count = 0
            for uid, count in self.kernel_log_counts.items():
                if profile_name and profile_name in uid:
                    kernel_count = count
                    break

            audit_count = trial.retrieval_log.shared_memory_count
            if kernel_count > 0 and audit_count == 0:
                logger.info(
                    "Trial %d: Kernel log confirms %d injections for '%s' "
                    "but audit returned 0. Updating injection_status.",
                    trial.trial_index, kernel_count, profile_name,
                )
                trial.retrieval_log.injection_status = "kernel_confirmed_audit_missed"
            elif kernel_count > 0 and audit_count > 0:
                logger.info(
                    "Trial %d: Kernel log (%d) and audit (%d) agree for '%s'.",
                    trial.trial_index, kernel_count, audit_count, profile_name,
                )

    def run_single_trial(
        self,
        trial_index: int,
        condition: str,
        pipeline: AgentPipeline,
        method: str = "",
        trial_data=None,
    ) -> TrialResult:
        """Execute one trial with log-and-continue error handling.

        Args:
            trial_index: Zero-based trial index.
            condition: Condition label for the trial result (e.g. "phase1",
                "kernel_shared", "naive_concat").
            pipeline: The AgentPipeline (or MethodPipeline) to run.
            method: Method name to record on the TrialResult. When empty,
                the condition string is used as the method label.
            trial_data: Pre-generated SyntheticTrialData to reuse. When None,
                the orchestrator generates fresh data for this trial.

        Returns:
            TrialResult with all fields populated.
        """
        effective_method = method if method else condition

        # Step 1: Generate (or reuse) synthetic data
        if trial_data is None:
            try:
                trial_data = self.generator.generate_trial_data(trial_index)
            except Exception as e:
                logger.error("Trial %d: synthetic data generation failed: %s", trial_index, e)
                return TrialResult(
                    trial_index=trial_index,
                    condition=condition,
                    method=effective_method,
                    failed=True,
                    error_message=str(e),
                )

        # Step 2: Run agent pipeline
        try:
            pipeline_result = pipeline.run_trial(trial_data)
        except Exception as e:
            logger.error("Trial %d: agent pipeline failed: %s", trial_index, e)
            return TrialResult(
                trial_index=trial_index,
                condition=condition,
                method=effective_method,
                failed=True,
                error_message=str(e),
                synthetic_profile=trial_data.profile,
                synthetic_task_context=trial_data.task_context,
                follow_up_query=trial_data.follow_up_query,
            )

        # Extract retrieval log from pipeline result
        retrieval_log = pipeline_result.retrieval_log

        # Phase 1 isolation verification
        if condition == "phase1" and retrieval_log:
            if not retrieval_log.cross_agent_found:
                logger.info("Trial %d: Phase 1 isolation verified — zero cross-agent memories.", trial_index)
            else:
                entries = [(e.owner_agent, e.memory_type) for e in retrieval_log.retrieved_memories]
                logger.warning(
                    "Trial %d: Cross-agent memory leakage detected! count=%d entries=%s",
                    trial_index, len([e for e in retrieval_log.retrieved_memories if e.owner_agent != "assistant_agent"]), entries,
                )

        # Phase 2 retrieval audit
        if condition == "phase2" and retrieval_log:
            entries = [(e.owner_agent, e.memory_type) for e in retrieval_log.retrieved_memories]
            logger.info(
                "Trial %d: Retrieved %d shared memories. Entries: %s",
                trial_index, retrieval_log.shared_memory_count, entries,
            )

        # Kernel log cross-reference is now handled post-run by
        # _cross_reference_kernel_logs() after all trials complete.

        # Step 3: Judge the assistant response
        try:
            scores = self.judge.evaluate(
                query=trial_data.follow_up_query,
                response=pipeline_result.assistant_response,
                profile=trial_data.profile,
                task_context=trial_data.task_context,
                plausible_actions=trial_data.plausible_actions,
            )
        except Exception as e:
            logger.warning("Trial %d: judge evaluation failed: %s", trial_index, e)
            scores = JudgeScores()

        # Memory counts heuristic (legacy path only)
        if condition == "phase2":
            memory_counts = MemoryCounts(total=2, shared=2, private=0)
        elif condition == "phase1":
            memory_counts = MemoryCounts(total=2, shared=0, private=2)
        else:
            memory_counts = MemoryCounts()

        return TrialResult(
            trial_index=trial_index,
            condition=condition,
            method=effective_method,
            profile_usage_score=scores.profile_usage_score,
            task_usage_score=scores.task_usage_score,
            integration_score=scores.integration_score,
            memory_counts=memory_counts,
            retrieved_context_count=pipeline_result.retrieved_context_count,
            latency_seconds=pipeline_result.latency_seconds,
            follow_up_query=trial_data.follow_up_query,
            assistant_response=pipeline_result.assistant_response or "",
            synthetic_profile=trial_data.profile,
            synthetic_task_context=trial_data.task_context,
            retrieval_log=retrieval_log,
            injection_diagnostics=pipeline_result.injection_diagnostics,
            written_memories=pipeline_result.written_memories,
        )

    def run(self) -> ExperimentResults:
        """Run the full experiment across all requested conditions/methods.

        When ``--condition`` is set (legacy path), the orchestrator uses
        ``AgentPipeline`` directly with the appropriate ``share_memory`` flag,
        preserving the original behavior exactly.

        When ``--method`` is used (new path), the orchestrator uses
        ``MethodPipeline`` for all methods. When ``--method all`` is specified,
        one ``SyntheticTrialData`` instance is generated per trial index and
        reused across all four method pipelines.
        """
        # ----------------------------------------------------------------
        # Legacy --condition path (backward compatible)
        # ----------------------------------------------------------------
        if self.condition is not None:
            return self._run_legacy_condition()

        # ----------------------------------------------------------------
        # New --method path
        # ----------------------------------------------------------------
        return self._run_method()

    def _run_legacy_condition(self) -> ExperimentResults:
        """Run the legacy --condition phase1/phase2 workflow unchanged."""
        if self.condition == "both":
            conditions = ["phase1", "phase2"]
        elif self.condition == "phase1":
            conditions = ["phase1"]
        else:
            conditions = ["phase2"]

        condition_results = []

        for cond in conditions:
            share_memory = cond == "phase2"
            pipeline = AgentPipeline(share_memory=share_memory, assistant_llms=self.assistant_llms)
            logger.info(
                "Running condition '%s' (share_memory=%s, kernel auto_inject assumed ON).",
                cond,
                share_memory,
            )

            trials: list[TrialResult] = []
            for i in tqdm(range(self.trials), desc=f"Condition: {cond}"):
                result = self.run_single_trial(i, cond, pipeline)
                trials.append(result)

            summary = self.writer.compute_summary_statistics(trials)
            condition_results.append(
                ConditionResults(condition=cond, trials=trials, summary=summary)
            )

        # Post-run kernel log cross-reference
        all_trials = [t for cr in condition_results for t in cr.trials]
        self._cross_reference_kernel_logs(all_trials)

        metadata = ExperimentMetadata(
            trials_per_condition=self.trials,
            timestamp=datetime.now().isoformat(),
            kernel_url=config.get_kernel_url(),
            conditions_run=conditions,
        )

        experiment = ExperimentResults(
            experiment_metadata=metadata,
            conditions=condition_results,
        )

        json_path = self.writer.write_json(experiment)
        logger.info("Results written to %s", json_path)

        if self.write_csv:
            csv_path = self.writer.write_csv(experiment)
            logger.info("CSV written to %s", csv_path)

        return experiment

    def _run_method(self) -> ExperimentResults:
        """Run the new --method workflow using MethodPipeline."""
        # Determine which methods to run
        if self.method == "all":
            methods_to_run = list(_ALL_METHODS)
        else:
            methods_to_run = [self.method]

        condition_results = []

        for method_name in methods_to_run:
            pipeline = MethodPipeline(method=method_name, assistant_llms=self.assistant_llms)
            logger.info("Running method '%s'.", method_name)

            trials: list[TrialResult] = []

            if self.method == "all":
                # Generate one SyntheticTrialData per trial index and reuse
                # it across all methods so scores are directly comparable.
                # The trial data cache is built on the first method pass and
                # reused for subsequent methods.
                if not hasattr(self, "_trial_data_cache"):
                    self._trial_data_cache: dict = {}

                for i in tqdm(range(self.trials), desc=f"Method: {method_name}"):
                    if i not in self._trial_data_cache:
                        try:
                            self._trial_data_cache[i] = self.generator.generate_trial_data(i)
                        except Exception as e:
                            logger.error("Trial %d: synthetic data generation failed: %s", i, e)
                            trials.append(TrialResult(
                                trial_index=i,
                                condition=method_name,
                                method=method_name,
                                failed=True,
                                error_message=str(e),
                            ))
                            continue

                    cached_data = self._trial_data_cache[i]
                    result = self.run_single_trial(
                        i, method_name, pipeline,
                        method=method_name,
                        trial_data=cached_data,
                    )
                    trials.append(result)
            else:
                # Single method — generate fresh data per trial
                for i in tqdm(range(self.trials), desc=f"Method: {method_name}"):
                    result = self.run_single_trial(
                        i, method_name, pipeline,
                        method=method_name,
                    )
                    trials.append(result)

            summary = self.writer.compute_summary_statistics(trials)
            condition_results.append(
                ConditionResults(condition=method_name, trials=trials, summary=summary)
            )

        # Post-run kernel log cross-reference
        all_trials = [t for cr in condition_results for t in cr.trials]
        self._cross_reference_kernel_logs(all_trials)

        # Determine output filename
        if self.method == "all":
            output_filename = "results_all_methods.json"
        else:
            output_filename = f"results_{self.method}.json"

        metadata = ExperimentMetadata(
            trials_per_condition=self.trials,
            timestamp=datetime.now().isoformat(),
            kernel_url=config.get_kernel_url(),
            conditions_run=methods_to_run,
            methods_run=methods_to_run,
        )

        experiment = ExperimentResults(
            experiment_metadata=metadata,
            conditions=condition_results,
        )

        json_path = self.writer.write_json(experiment, filename=output_filename)
        logger.info("Results written to %s", json_path)

        if self.write_csv:
            csv_path = self.writer.write_csv(experiment)
            logger.info("CSV written to %s", csv_path)

        # Print comparative summary after all methods complete
        self.writer.print_summary(experiment)

        return experiment


def main():
    """Parse CLI arguments and run the evaluation orchestrator."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orchestrator = EvaluationOrchestrator(
        trials=args.trials,
        output_dir=args.output,
        write_csv=args.csv,
        condition=args.condition,
        method=args.method,
        assistant_llms=_parse_model_arg(args.assistant_model),
        judge_llms=_parse_model_arg(args.judge_model),
        run_id=args.run_id,
        kernel_logs_path=args.kernel_logs,
    )

    experiment = orchestrator.run()

    # Legacy --condition path: print the original summary format
    if args.condition is not None:
        meta = experiment.experiment_metadata
        print(f"\n{'=' * 60}")
        print(f"Experiment complete — {meta.timestamp}")
        print(f"Conditions: {', '.join(meta.conditions_run)}")
        print(f"Trials per condition: {meta.trials_per_condition}")
        print(f"{'=' * 60}")

        for cond_result in experiment.conditions:
            s = cond_result.summary
            print(f"\n--- {cond_result.condition} ---")
            print(f"  Profile Usage:      mean={s.profile_usage.mean:.2f}  std={s.profile_usage.std:.2f}")
            print(f"  Task Usage:         mean={s.task_usage.mean:.2f}  std={s.task_usage.std:.2f}")
            print(f"  Integration:        mean={s.integration.mean:.2f}  std={s.integration.std:.2f}")
            print(f"  Latency (s):        mean={s.latency.mean:.2f}  std={s.latency.std:.2f}")
            print(f"  Memory total:       mean={s.memory_total.mean:.2f}")
            print(f"  Injected memories:  mean={s.injected_memories.mean:.2f}  std={s.injected_memories.std:.2f}")
            print(f"  Trials: {s.total_trials} total, {s.failed_trials} failed")

            non_failed = [t for t in cond_result.trials if not t.failed]
            mean_shared = statistics.mean([
                t.retrieval_log.shared_memory_count
                for t in non_failed if t.retrieval_log
            ]) if any(t.retrieval_log for t in non_failed) else 0.0
            cross_agent_count = sum(
                1 for t in non_failed if t.retrieval_log and t.retrieval_log.cross_agent_found
            )
            print(f"  Shared mem retrieved: mean={mean_shared:.2f}")
            print(f"  Cross-agent trials:  {cross_agent_count}/{len(non_failed)}")

        # Comparative analysis: Phase 1 vs Phase 2
        conditions_by_name = {c.condition: c.summary for c in experiment.conditions}
        if "phase1" in conditions_by_name and "phase2" in conditions_by_name:
            p1 = conditions_by_name["phase1"]
            p2 = conditions_by_name["phase2"]

            print(f"\n{'=' * 60}")
            print("Comparative Analysis: Phase 1 (private) vs Phase 2 (shared)")
            print(f"{'=' * 60}")
            header = f"  {'Metric':<22} {'Phase1':>12} {'Phase2':>12} {'Delta':>12}"
            print(header)
            print(f"  {'-' * 58}")

            rows = [
                ("Profile Usage mean", p1.profile_usage.mean, p2.profile_usage.mean),
                ("Profile Usage std", p1.profile_usage.std, p2.profile_usage.std),
                ("Task Usage mean", p1.task_usage.mean, p2.task_usage.mean),
                ("Task Usage std", p1.task_usage.std, p2.task_usage.std),
                ("Integration mean", p1.integration.mean, p2.integration.mean),
                ("Integration std", p1.integration.std, p2.integration.std),
                ("Injected mem mean", p1.injected_memories.mean, p2.injected_memories.mean),
            ]
            for label, v1, v2 in rows:
                delta = v2 - v1
                sign = "+" if delta >= 0 else ""
                print(f"  {label:<22} {v1:>12.2f} {v2:>12.2f} {sign + f'{delta:.2f}':>12}")

        print()


if __name__ == "__main__":
    main()
