"""Agent execution pipeline for the shared memory evaluation harness.

Handles loading agents locally and running them in sequence:
ProfileAgent → TaskAgent → AssistantAgent. The share_memory flag
controls whether agents store memories as shared or private,
corresponding to the Phase 2 and Phase 1 experimental conditions.

The pipeline captures injection diagnostics from the kernel response
(when auto_inject is enabled) and tracks written memory metadata
by intercepting create_memory calls from the harness side.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

logger = logging.getLogger(__name__)

from cerebrum.example.agents.profile_agent.agent import ProfileAgent
from cerebrum.example.agents.task_agent.agent import TaskAgent
from cerebrum.example.agents.assistant_agent.agent import AssistantAgent
from cerebrum.memory.apis import search_memories, create_memory

from benchmarks.shared_memory.models import (
    InjectedMemoryEntry,
    InjectionDiagnostics,
    RetrievalLog,
    RetrievalLogEntry,
    SyntheticTrialData,
    WrittenMemoryRecord,
)


@dataclass
class PipelineResult:
    """Result from a single trial's agent pipeline execution."""

    profile_result: dict
    task_result: dict
    assistant_result: dict
    assistant_response: str
    latency_seconds: float
    retrieval_log: Optional[RetrievalLog] = None
    injection_diagnostics: Optional[InjectionDiagnostics] = None
    written_memories: List[WrittenMemoryRecord] = field(default_factory=list)
    method: str = ""
    retrieved_context_count: Optional[int] = None


class AgentPipeline:
    """Runs the three-agent pipeline for a single trial.

    Instantiates ProfileAgent, TaskAgent, and AssistantAgent in sequence,
    configuring the share_memory attribute based on the experimental
    condition. Measures AssistantAgent latency for metric collection.

    Instead of patching search_memories on the agent side, the pipeline:
    - Patches create_memory to capture WrittenMemoryRecord entries
    - Extracts injection diagnostics from the kernel's llm_chat response
    - Falls back to a harness-side search_memories audit query when the
      kernel does not return diagnostics

    Args:
        share_memory: If True, agents use sharing_policy="shared" (Phase 2).
            If False, agents use sharing_policy="private" (Phase 1).
        assistant_llms: Optional model override for AssistantAgent LLM calls.
    """

    def __init__(self, share_memory: bool, assistant_llms: list | None = None):
        self.share_memory = share_memory
        self.assistant_llms = assistant_llms

    def run_trial(self, trial_data: SyntheticTrialData) -> PipelineResult:
        """Execute the full agent pipeline for one trial.

        Args:
            trial_data: The synthetic data for this trial, containing
                profile, task_context, and follow_up_query.

        Returns:
            PipelineResult with all agent outputs, assistant latency,
            injection diagnostics, and written memory records.
        """
        written_records: List[WrittenMemoryRecord] = []

        # Capture reference to real create_memory before patching
        from cerebrum.memory.apis import create_memory as _real_create_memory

        def capturing_create_memory(agent_name, content, metadata=None, base_url=None):
            """Intercept create_memory to capture written metadata."""
            if metadata:
                written_records.append(WrittenMemoryRecord(
                    agent_name=metadata.get("owner_agent", agent_name),
                    memory_type=metadata.get("memory_type", ""),
                    sharing_policy=metadata.get("sharing_policy", "private"),
                    user_id=metadata.get("user_id", ""),
                ))
            # Call through to the real create_memory (captured before patch)
            return _real_create_memory(agent_name, content, metadata=metadata, base_url=base_url)

        with patch("cerebrum.memory.apis.create_memory", side_effect=capturing_create_memory):
            # Step 1: Run ProfileAgent with synthetic profile data
            profile_agent = ProfileAgent("profile_agent")
            profile_agent.share_memory = self.share_memory
            profile_agent.user_id = trial_data.user_id
            profile_agent.llms = self.assistant_llms
            profile_result = profile_agent.run(
                json.dumps(trial_data.profile.model_dump())
            )

            # Diagnostic: verify Mem0 actually stored profile memories
            if self.share_memory:
                diag_response = search_memories(
                    agent_name="profile_agent",
                    query=f"profile {trial_data.user_id}",
                )
                diag_results = []
                if diag_response and isinstance(diag_response, dict):
                    resp = diag_response.get("response", {})
                    if resp and isinstance(resp, dict):
                        diag_results = resp.get("search_results", []) or []
                if diag_results:
                    logger.info(
                        "Mem0 diagnostic: found %d results for user_id=%s after ProfileAgent write",
                        len(diag_results), trial_data.user_id
                    )
                else:
                    logger.warning(
                        "Mem0 diagnostic: 0 results for user_id=%s after ProfileAgent write. "
                        "Mem0 fact extraction may have failed silently.",
                        trial_data.user_id
                    )

            # Step 2: Run TaskAgent with synthetic task context data
            task_agent = TaskAgent("task_agent")
            task_agent.share_memory = self.share_memory
            task_agent.user_id = trial_data.user_id
            task_agent.llms = self.assistant_llms
            task_result = task_agent.run(
                json.dumps(trial_data.task_context.model_dump())
            )

            # Step 3: Run AssistantAgent with follow-up query, measuring latency
            assistant_agent = AssistantAgent("assistant_agent")
            assistant_agent.share_memory = self.share_memory
            assistant_agent.user_id = trial_data.user_id
            assistant_agent.llms = self.assistant_llms

            start_time = time.time()
            assistant_result = assistant_agent.run(trial_data.follow_up_query)
            latency_seconds = time.time() - start_time

        # Extract the response text from the assistant result
        assistant_response = assistant_result.get("result", "")

        # Try to extract injection diagnostics from the kernel response
        injection_diagnostics = self._extract_injection_diagnostics(assistant_result)

        # Build retrieval log: use kernel diagnostics or fall back to audit query
        if injection_diagnostics and injection_diagnostics.injected_count > 0:
            retrieval_log = self._retrieval_log_from_diagnostics(injection_diagnostics)
            # injection_status defaults to "confirmed"
        else:
            retrieval_log = self._audit_shared_memories(
                trial_data.user_id, trial_data.follow_up_query
            )
            if retrieval_log.shared_memory_count > 0:
                retrieval_log.injection_status = "audit_inferred"
            elif self.share_memory:
                retrieval_log.injection_status = "unknown"
                logger.warning(
                    "Observability gap: kernel diagnostics absent and audit "
                    "query returned 0 results for Phase 2 trial. "
                    "Injection status unknown."
                )

        return PipelineResult(
            profile_result=profile_result,
            task_result=task_result,
            assistant_result=assistant_result,
            assistant_response=assistant_response,
            latency_seconds=latency_seconds,
            retrieval_log=retrieval_log,
            injection_diagnostics=injection_diagnostics,
            written_memories=written_records,
        )

    def _extract_injection_diagnostics(
        self, assistant_result: dict
    ) -> Optional[InjectionDiagnostics]:
        """Extract injection diagnostics from the kernel's llm_chat response.

        The kernel may include an ``injection_diagnostics`` field in the
        response when ``auto_inject`` is enabled.

        Args:
            assistant_result: The raw result dict from AssistantAgent.run().

        Returns:
            InjectionDiagnostics if the kernel provided them, else None.
        """
        diag_data = assistant_result.get("injection_diagnostics")
        if not isinstance(diag_data, dict):
            return None

        entries = []
        for mem in diag_data.get("injected_memories", []):
            entries.append(InjectedMemoryEntry(
                owner_agent=mem.get("owner_agent", ""),
                memory_type=mem.get("memory_type", ""),
                match_score=mem.get("match_score"),
            ))

        return InjectionDiagnostics(
            injected_count=diag_data.get("injected_count", len(entries)),
            injected_memories=entries,
        )

    def _retrieval_log_from_diagnostics(
        self, diagnostics: InjectionDiagnostics
    ) -> RetrievalLog:
        """Build a RetrievalLog from kernel injection diagnostics.

        Args:
            diagnostics: The InjectionDiagnostics extracted from the response.

        Returns:
            RetrievalLog populated from the diagnostics data.
        """
        entries = []
        cross_agent = False
        for mem in diagnostics.injected_memories:
            entries.append(RetrievalLogEntry(
                owner_agent=mem.owner_agent,
                memory_type=mem.memory_type,
            ))
            if mem.owner_agent != "assistant_agent":
                cross_agent = True

        return RetrievalLog(
            shared_memory_count=diagnostics.injected_count,
            retrieved_memories=entries,
            cross_agent_found=cross_agent,
        )

    # Kernel context injector defaults (must match AIOS config.yaml)
    RELEVANCE_THRESHOLD = 0.3
    MAX_INJECTED_MEMORIES = 5  # memory.max_injected_memories default
    # Kernel over-fetches 4× to compensate for post-retrieval filtering
    AUDIT_TOP_K = MAX_INJECTED_MEMORIES * 4  # = 20
    WRITER_AGENTS = ["profile_agent", "task_agent"]

    def _audit_shared_memories(self, user_id: str, query_text: str) -> RetrievalLog:
        """Query the kernel from the harness side to audit shared memories.

        Issues one query per writer agent in WRITER_AGENTS, routing each
        through the writer's own agent_name so the kernel returns that
        agent's owned memories. Merges and deduplicates results before
        applying the existing relevance threshold filter.

        This works around the kernel's agent-scoping without modifying
        the kernel itself — each query routes through the correct owner
        so the syscall handler returns results for that agent's memories.

        Args:
            user_id: The user identifier to scope the audit query.
            query_text: The actual user message (follow-up query) — matches
                the kernel injector's ``content`` parameter.

        Returns:
            RetrievalLog built from the merged audit query results.
        """
        if not user_id:
            return RetrievalLog()

        from cerebrum.memory.apis import MemoryQuery
        from cerebrum.utils.communication import send_request
        from cerebrum.config.config_manager import config

        # Fan-out: one query per writer agent, routed through that
        # agent's name so the kernel returns its owned memories.
        # Each writer gets its own try/except for per-writer isolation.
        all_results = []
        for writer_agent in self.WRITER_AGENTS:
            try:
                query_obj = MemoryQuery(
                    operation_type="retrieve_memory",
                    params={
                        "content": query_text,
                        "k": self.AUDIT_TOP_K,
                        "user_id": user_id,
                        "sharing_policy": "shared",
                        "agent_name": writer_agent,
                    },
                )
                result = send_request(writer_agent, query_obj, config.get_kernel_url())
                # Per-writer instrumentation: raw result count before filtering
                raw_count = 0
                if isinstance(result, dict):
                    resp = result.get("response", {})
                    if isinstance(resp, dict):
                        raw_count = len(resp.get("search_results", []) or [])
                logger.debug(
                    "Audit fan-out [%s]: raw_count=%d", writer_agent, raw_count
                )
                all_results.append(result)
            except Exception as e:
                logger.warning("Audit query for %s failed: %s", writer_agent, e)
                continue

        # If ALL writers failed, return empty RetrievalLog gracefully
        if not all_results:
            return RetrievalLog()

        # Merge search_results from all fan-out responses
        merged_search_results = []
        for result in all_results:
            if isinstance(result, dict):
                resp = result.get("response", {})
                if isinstance(resp, dict):
                    merged_search_results.extend(resp.get("search_results", []) or [])

        # Post-merge instrumentation: total merged count before dedup
        logger.info(
            "Audit post-merge: merged_count=%d (before dedup)", len(merged_search_results)
        )

        # Deduplicate by memory_id only (stable provider key).
        # Entries with null/empty memory_id skip dedup entirely — always included.
        seen_ids = set()
        unique_results = []
        dropped_ids = []
        for mem in merged_search_results:
            mem_id = mem.get("memory_id")
            if not mem_id:
                # Null/empty memory_id: skip dedup, always include
                unique_results.append(mem)
                continue
            if mem_id in seen_ids:
                dropped_ids.append(mem_id)
                continue
            seen_ids.add(mem_id)
            unique_results.append(mem)

        # Post-dedup instrumentation: deduplicated count and dropped entries
        logger.info(
            "Audit post-dedup: unique_count=%d, dropped=%d",
            len(unique_results), len(dropped_ids),
        )
        if dropped_ids:
            logger.debug(
                "Audit dedup dropped memory_ids: %s", dropped_ids
            )

        # Pass merged, deduplicated results through existing filter
        merged_response = {"response": {"search_results": unique_results}}

        # Post-threshold instrumentation: final count passed to filter
        logger.info(
            "Audit post-threshold: passing %d entries to _build_retrieval_log_from_search",
            len(unique_results),
        )

        return self._build_retrieval_log_from_search(merged_response)

    def _build_retrieval_log_from_search(
        self, search_result: dict
    ) -> RetrievalLog:
        """Build a RetrievalLog from a raw search_memories response.

        Applies the same post-retrieval relevance filter as the kernel
        context injector: memories with score < RELEVANCE_THRESHOLD (0.5)
        are dropped, while memories with score=None are kept (matching
        the kernel's behavior for unscored results).

        Args:
            search_result: Raw result dict from search_memories.

        Returns:
            RetrievalLog with shared_memory_count, retrieved_memories,
            and cross_agent_found populated from the filtered results.
        """
        entries = []
        shared_count = 0
        cross_agent = False

        if not isinstance(search_result, dict):
            return RetrievalLog()

        resp = search_result.get("response", {})
        if not isinstance(resp, dict):
            return RetrievalLog()

        search_results = resp.get("search_results", []) or []
        for mem in search_results:
            # Apply kernel relevance threshold: drop below threshold,
            # keep None-scored memories (matches kernel behavior)
            score = mem.get("score")
            if score is not None and score < self.RELEVANCE_THRESHOLD:
                continue

            meta = mem.get("metadata", {})
            if not meta:
                continue
            owner = meta.get("owner_agent", "")
            mem_type = meta.get("memory_type", "")
            if not owner:
                continue
            entries.append(RetrievalLogEntry(
                owner_agent=owner,
                memory_type=mem_type,
            ))
            if meta.get("sharing_policy") == "shared":
                shared_count += 1
            if owner != "assistant_agent":
                cross_agent = True

        return RetrievalLog(
            shared_memory_count=shared_count,
            retrieved_memories=entries,
            cross_agent_found=cross_agent,
        )


class MethodPipeline:
    """Dispatches trial execution to the appropriate method implementation.

    Acts as a router that derives a method-scoped user ID for each trial
    and delegates to the correct execution path. The existing AgentPipeline
    is called unchanged for the ``kernel_shared`` method.

    Args:
        method: One of ``"kernel_shared"``, ``"naive_concat"``,
            ``"vanilla_rag"``, or ``"mem0_default"``.
        top_k: Number of chunks/entries to retrieve for ``vanilla_rag``
            and ``mem0_default`` (default 3).
    """

    def __init__(self, method: str, top_k: int = 3, assistant_llms: list | None = None):
        self.method = method
        self.top_k = top_k
        self.assistant_llms = assistant_llms

    def run_trial(self, trial_data: SyntheticTrialData) -> PipelineResult:
        """Route to the correct execution path based on self.method.

        Derives a Method_User_ID by appending the method name to the base
        user ID with a double-underscore separator, then dispatches to the
        appropriate private method.

        Args:
            trial_data: The synthetic data for this trial.

        Returns:
            PipelineResult with method field set to self.method.

        Raises:
            ValueError: If self.method is not a recognised method name.
        """
        method_user_id = f"{trial_data.user_id}__{self.method}"
        if self.method == "kernel_shared":
            return self._run_kernel_shared(trial_data, method_user_id)
        elif self.method == "naive_concat":
            return self._run_naive_concat(trial_data, method_user_id)
        elif self.method == "vanilla_rag":
            return self._run_vanilla_rag(trial_data, method_user_id)
        elif self.method == "mem0_default":
            return self._run_mem0_default(trial_data, method_user_id)
        else:
            raise ValueError(
                f"Unknown method: {self.method!r}. "
                "Valid methods are: kernel_shared, naive_concat, vanilla_rag, mem0_default."
            )

    def _run_kernel_shared(
        self, trial_data: SyntheticTrialData, method_user_id: str
    ) -> PipelineResult:
        """Delegate to the existing AgentPipeline unchanged.

        Creates a copy of trial_data with user_id replaced by method_user_id
        so that memory namespacing is scoped to this method, then runs the
        existing AgentPipeline with share_memory=True.

        Args:
            trial_data: Original trial data.
            method_user_id: Method-scoped user ID (e.g. ``alice__kernel_shared``).

        Returns:
            PipelineResult from AgentPipeline with method set to
            ``"kernel_shared"``.
        """
        scoped_trial = trial_data.model_copy(update={"user_id": method_user_id})
        pipeline = AgentPipeline(share_memory=True, assistant_llms=self.assistant_llms)
        result = pipeline.run_trial(scoped_trial)
        result.method = "kernel_shared"
        return result

    def _run_naive_concat(
        self, trial_data: SyntheticTrialData, method_user_id: str
    ) -> PipelineResult:
        """Naive concatenation baseline — prepend full context to the prompt.

        Builds a structured plain-text context block from the synthetic profile
        and task context, prepends it to the follow-up query, and passes the
        combined string to AssistantAgent.run(). No memory system is accessed.

        Args:
            trial_data: The synthetic data for this trial.
            method_user_id: Method-scoped user ID (e.g. ``alice__naive_concat``).

        Returns:
            PipelineResult with method="naive_concat", retrieved_context_count=0,
            and injection_diagnostics recording the character length of the
            injected context block.
        """
        profile = trial_data.profile
        task_context = trial_data.task_context
        follow_up_query = trial_data.follow_up_query

        # Build the structured plain-text context block
        context_block = (
            f"--- USER PROFILE ---\n"
            f"Name: {profile.user_name}\n"
            f"Preferred Tools: {', '.join(profile.preferred_tools)}\n"
            f"Preferred Language: {profile.preferred_language}\n"
            f"Response Style: {profile.response_style}\n"
            f"\n"
            f"--- TASK CONTEXT ---\n"
            f"Current Project: {task_context.current_project}\n"
            f"Active Experiment: {task_context.active_experiment}\n"
            f"Goals: {', '.join(task_context.goals)}\n"
            f"Blockers: {', '.join(task_context.blockers)}\n"
            f"Next Steps: {', '.join(task_context.next_steps)}\n"
            f"\n"
            f"--- QUERY ---\n"
            f"{follow_up_query}"
        )

        # Record the character length of the injected context block
        injection_diagnostics = InjectionDiagnostics(injected_count=len(context_block))

        # Instantiate AssistantAgent with method-scoped user_id
        assistant_agent = AssistantAgent("assistant_agent")
        assistant_agent.user_id = method_user_id
        assistant_agent.llms = self.assistant_llms

        start_time = time.time()
        assistant_result = assistant_agent.run(context_block)
        latency_seconds = time.time() - start_time

        assistant_response = assistant_result.get("result", "")

        return PipelineResult(
            profile_result={},
            task_result={},
            assistant_result=assistant_result,
            assistant_response=assistant_response,
            latency_seconds=latency_seconds,
            retrieval_log=None,
            injection_diagnostics=injection_diagnostics,
            written_memories=[],
            method="naive_concat",
            retrieved_context_count=0,
        )

    def _run_vanilla_rag(
        self, trial_data: SyntheticTrialData, method_user_id: str
    ) -> PipelineResult:
        """Vanilla RAG baseline — TF-IDF retrieval over context chunks.

        Builds document chunks from the profile and task context (one chunk per
        logical section), fits a TfidfVectorizer on those chunks, transforms the
        follow-up query with the same vectorizer, computes cosine similarity, and
        selects the top-k most relevant chunks to inject into the AssistantAgent
        prompt. No kernel memory or Mem0 API is called.

        Args:
            trial_data: The synthetic data for this trial.
            method_user_id: Method-scoped user ID (e.g. ``alice__vanilla_rag``).

        Returns:
            PipelineResult with method="vanilla_rag" and retrieved_context_count
            set to the number of chunks actually injected.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        profile = trial_data.profile
        task_context = trial_data.task_context
        follow_up_query = trial_data.follow_up_query

        # Build one chunk per logical section
        chunks: List[str] = []

        # Profile block
        chunks.append(
            f"User Profile:\n"
            f"Name: {profile.user_name}\n"
            f"Preferred Tools: {', '.join(profile.preferred_tools)}\n"
            f"Preferred Language: {profile.preferred_language}\n"
            f"Response Style: {profile.response_style}"
        )

        # Task block
        chunks.append(
            f"Task Context:\n"
            f"Current Project: {task_context.current_project}\n"
            f"Active Experiment: {task_context.active_experiment}"
        )

        # One chunk per goal
        for goal in task_context.goals:
            chunks.append(f"Goal: {goal}")

        # One chunk per blocker
        for blocker in task_context.blockers:
            chunks.append(f"Blocker: {blocker}")

        # One chunk per next step
        for next_step in task_context.next_steps:
            chunks.append(f"Next Step: {next_step}")

        # Fit TF-IDF on chunks; transform query with the same vectorizer
        vectorizer = TfidfVectorizer()
        chunk_matrix = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([follow_up_query])

        # Compute cosine similarity between query and each chunk
        scores = cosine_similarity(query_vector, chunk_matrix).flatten()

        # Select top-k chunks; fall back to all chunks if all scores are 0.0
        if scores.max() == 0.0:
            selected_chunks = chunks
        else:
            k = min(self.top_k, len(chunks))
            top_indices = scores.argsort()[::-1][:k]
            selected_chunks = [chunks[i] for i in top_indices]

        # Format selected chunks as a context block and prepend to the query
        context_parts = "\n\n".join(selected_chunks)
        augmented_query = (
            f"--- RETRIEVED CONTEXT ---\n"
            f"{context_parts}\n\n"
            f"--- QUERY ---\n"
            f"{follow_up_query}"
        )

        # Instantiate AssistantAgent with method-scoped user_id
        assistant_agent = AssistantAgent("assistant_agent")
        assistant_agent.user_id = method_user_id
        assistant_agent.llms = self.assistant_llms

        start_time = time.time()
        assistant_result = assistant_agent.run(augmented_query)
        latency_seconds = time.time() - start_time

        assistant_response = assistant_result.get("result", "")

        return PipelineResult(
            profile_result={},
            task_result={},
            assistant_result=assistant_result,
            assistant_response=assistant_response,
            latency_seconds=latency_seconds,
            retrieval_log=None,
            injection_diagnostics=None,
            written_memories=[],
            method="vanilla_rag",
            retrieved_context_count=len(selected_chunks),
        )

    def _run_mem0_default(
        self, trial_data: SyntheticTrialData, method_user_id: str
    ) -> PipelineResult:
        """Private-memory baseline using the same 3-agent pipeline.

        Uses the identical AgentPipeline as kernel_shared, but with
        share_memory=False. ProfileAgent and TaskAgent still write memories,
        but with sharing_policy="private", so the kernel's auto_inject will
        NOT surface them cross-agent to AssistantAgent.

        This ensures the only variable between kernel_shared and mem0_default
        is whether memories are shared or private.

        Args:
            trial_data: Original trial data.
            method_user_id: Method-scoped user ID (e.g. ``alice__mem0_default``).

        Returns:
            PipelineResult from AgentPipeline with method set to
            ``"mem0_default"``.
        """
        scoped_trial = trial_data.model_copy(update={"user_id": method_user_id})
        pipeline = AgentPipeline(share_memory=False, assistant_llms=self.assistant_llms)
        result = pipeline.run_trial(scoped_trial)
        result.method = "mem0_default"
        return result
