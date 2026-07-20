"""Context Injector — AIOS kernel cross-agent memory injection pipeline.

Intercepts ``llm_chat`` calls when ``auto_inject=true``, retrieves shared
memories from other agents for the resolved ``user_id``, formats them as
natural language, and prepends them to the LLM messages.

This module implements the **write barrier** that gates cross-agent retrieval
until pending writes for the target ``user_id`` are visible in the Mem0 store.
The barrier ensures deterministic injection regardless of Mem0 indexing latency.

Key components:
- ``should_invoke_barrier(...)`` — gating predicate (short-circuits for
  preservation clauses 3.1, 3.3, 3.4, 3.5).
- ``write_barrier(...)`` — bounded-wait logic: flush (if supported) → poll
  loop with ``get_all`` + ``reconcile`` every ``poll_interval_ms`` until
  pending count == 0 or timeout.
- ``ContextInjector`` — orchestrates the injection pipeline including
  user_id resolution, barrier gating, retrieval, ranking, filtering,
  token-cap, and formatting.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aios.config.config_loader import get_config, KernelConfig, MemoryConfig, WriteBarrierConfig
from aios.memory.mem0_provider import Mem0Provider
from aios.memory.pending_writes_tracker import tracker, PendingWritesTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BarrierResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarrierResult:
    """Result of a write barrier invocation.

    Attributes:
        status: One of "ok" (all writes visible), "timeout" (barrier timed
            out with residual pending writes), or "noop" (no pending writes
            to wait on; barrier was a no-op).
        waited_ms: Milliseconds the barrier actually waited before returning.
        residual_pending: Number of pending writes still not visible when the
            barrier returned. Zero on "ok" and "noop".
    """

    status: str  # "ok" | "timeout" | "noop"
    waited_ms: int
    residual_pending: int


# ---------------------------------------------------------------------------
# InjectionResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class InjectionResult:
    """Result of a ContextInjector.inject() call.

    Attributes:
        memories: List of memory dicts that passed all filters and are ready
            for injection into the LLM prompt.
        total_retrieved: Total number of cross-agent shared memories before
            any filtering (relevance threshold, max count, token cap).
        barrier_result: The BarrierResult if the write barrier was invoked,
            or None if the barrier was not needed.
        diagnostics: Observability dict containing barrier metadata and
            filter statistics for audit consumers.
    """

    memories: list[dict[str, Any]] = field(default_factory=list)
    total_retrieved: int = 0
    barrier_result: BarrierResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gating Predicate
# ---------------------------------------------------------------------------


def should_invoke_barrier(
    user_id: str,
    calling_agent: str,
    cross_agent: bool,
    config: MemoryConfig,
    pending_tracker: PendingWritesTracker,
) -> bool:
    """Determine whether the write barrier should be invoked before retrieval.

    Each False return maps 1:1 to a preservation clause:
    - auto_inject=false → clause 3.4
    - not cross_agent   → clause 3.3
    - no pending writes → clause 3.1
    - only private pending writes from other agents → clause 3.5

    Args:
        user_id: The resolved user_id for the retrieval request.
        calling_agent: The agent that triggered llm_chat.
        cross_agent: Whether the retrieval scope includes other agents.
        config: The memory subsystem configuration.
        pending_tracker: The PendingWritesTracker instance.

    Returns:
        True if the barrier should be invoked, False otherwise.
    """
    # Kill-switch: barrier disabled in config
    if not config.write_barrier.enabled:
        return False

    # Clause 3.4: auto_inject=false bypasses the injector entirely
    if not config.auto_inject:
        return False

    # Clause 3.3: single-agent retrieval does not need the barrier
    if not cross_agent:
        return False

    # Clause 3.1: no pending writes for this user_id
    pending = pending_tracker.pending_for(user_id)
    if not pending:
        return False

    # Clause 3.5: filter to only shared writes from other agents
    # If none remain, no barrier needed (only private writes pending)
    cross_agent_pending = [
        w for w in pending
        if w.sharing_policy == "shared" and w.owner_agent != calling_agent
    ]
    if not cross_agent_pending:
        return False

    return True


# ---------------------------------------------------------------------------
# Write Barrier (bounded-wait logic)
# ---------------------------------------------------------------------------


def _write_barrier(
    user_id: str,
    pending_tracker: PendingWritesTracker,
    mem0_provider: Mem0Provider,
    barrier_config: WriteBarrierConfig,
) -> BarrierResult:
    """Execute the bounded-wait write barrier for a user_id.

    Implements the design's hybrid strategy:
    1. Call ``flush()`` on the Mem0 provider if supported.
    2. Poll ``get_all(user_id)`` + ``reconcile()`` every
       ``poll_interval_ms`` until all pending writes are acknowledged
       or ``timeout_ms`` is exceeded.

    On timeout, emits a structured warning and returns with
    ``status="timeout"``. The caller proceeds with retrieval against
    whatever Mem0 state is currently committed.

    Args:
        user_id: The user context whose pending writes to wait on.
        pending_tracker: The PendingWritesTracker instance.
        mem0_provider: The Mem0Provider for flush and get_all calls.
        barrier_config: The write barrier configuration (timeout, poll interval).

    Returns:
        A BarrierResult describing the outcome.
    """
    start_ms = int(time.time() * 1000)

    # Check if there's anything to wait on
    pending_count = len(pending_tracker.pending_for(user_id))
    if pending_count == 0:
        return BarrierResult(status="noop", waited_ms=0, residual_pending=0)

    # Branch A: explicit flush when supported
    if mem0_provider.supports_flush():
        mem0_provider.flush(user_id=user_id, timeout_ms=barrier_config.timeout_ms)

    # Branch B: bounded visibility poll (always runs, confirms visibility)
    while True:
        current_pending = pending_tracker.pending_for(user_id)
        if not current_pending:
            # All writes are now visible
            waited_ms = int(time.time() * 1000) - start_ms
            return BarrierResult(status="ok", waited_ms=waited_ms, residual_pending=0)

        elapsed_ms = int(time.time() * 1000) - start_ms
        if elapsed_ms >= barrier_config.timeout_ms:
            residual = len(current_pending)
            waited_ms = int(time.time() * 1000) - start_ms
            logger.warning(
                "write_barrier_timeout: user_id=%s, residual_pending=%d, waited_ms=%d",
                user_id,
                residual,
                waited_ms,
            )
            return BarrierResult(
                status="timeout",
                waited_ms=waited_ms,
                residual_pending=residual,
            )

        # Probe Mem0 for visible memories and reconcile
        visible = mem0_provider.get_all(user_id=user_id)
        pending_tracker.reconcile(user_id, visible)

        # Sleep before next poll
        time.sleep(barrier_config.poll_interval_ms / 1000.0)


# ---------------------------------------------------------------------------
# ContextInjector
# ---------------------------------------------------------------------------


class ContextInjector:
    """Kernel component that injects cross-agent shared memories into LLM calls.

    The injector pipeline:
    1. Resolve user_id (pass-through, post-Bug-1 fix).
    2. Check the gating predicate (should_invoke_barrier).
    3. If barrier needed: execute write_barrier (bounded-wait).
    4. Retrieve cross-agent shared memories via Mem0Provider.
    5. Apply relevance threshold filtering.
    6. Apply max_injected_memories cap.
    7. Apply max_memory_tokens token cap.
    8. Return the InjectionResult with diagnostics.

    Args:
        mem0_provider: The Mem0Provider instance for memory operations.
        config: Optional KernelConfig override. If not provided, loads from
            the module-level singleton via ``get_config()``.
        pending_tracker: Optional PendingWritesTracker override. Defaults to
            the module-level singleton ``tracker``.
    """

    def __init__(
        self,
        mem0_provider: Mem0Provider,
        config: KernelConfig | None = None,
        pending_tracker: PendingWritesTracker | None = None,
    ) -> None:
        self._mem0_provider = mem0_provider
        self._config = config or get_config()
        self._tracker = pending_tracker or tracker

    @property
    def memory_config(self) -> MemoryConfig:
        """Return the memory subsystem config."""
        return self._config.memory

    @property
    def barrier_config(self) -> WriteBarrierConfig:
        """Return the write barrier config."""
        return self._config.memory.write_barrier

    def inject(
        self,
        user_id: str,
        calling_agent: str,
        cross_agent: bool = True,
    ) -> InjectionResult:
        """Execute the injection pipeline for an LLM call.

        Steps:
        1. Resolve user_id (pass-through since this is post-Bug-1).
        2. Evaluate the gating predicate.
        3. If barrier needed, call write_barrier.
        4. Retrieve cross-agent shared memories.
        5. Apply relevance threshold, max memories, token cap.
        6. Return result with diagnostics.

        On barrier ``status="timeout"``, retrieval still runs against whatever
        Mem0 has committed so far — no exception is raised. The warning
        emitted inside write_barrier is sufficient for observability.

        Args:
            user_id: The resolved user_id for the retrieval.
            calling_agent: The agent that triggered the LLM call.
            cross_agent: Whether to retrieve cross-agent shared memories.
                Defaults to True.

        Returns:
            An InjectionResult containing the filtered memories and
            diagnostics (including BarrierResult when the barrier ran).
        """
        # Step 1: Resolve user_id (request-scoped only, no fallbacks)
        resolved_user_id = self._resolve_user_id(user_id)

        # Early guard: skip user-scoped retrieval when no user_id provided
        if not resolved_user_id:
            return InjectionResult(
                memories=[],
                total_retrieved=0,
                barrier_result=None,
                diagnostics={
                    "user_id": None,
                    "calling_agent": calling_agent,
                    "cross_agent": cross_agent,
                    "barrier_invoked": False,
                    "barrier_result": None,
                    "total_retrieved": 0,
                    "post_filter_count": 0,
                    "skipped_reason": "no_request_scoped_user_id",
                },
            )

        # Step 2: Gating predicate
        barrier_result: BarrierResult | None = None
        barrier_invoked = should_invoke_barrier(
            user_id=resolved_user_id,
            calling_agent=calling_agent,
            cross_agent=cross_agent,
            config=self.memory_config,
            pending_tracker=self._tracker,
        )

        # Step 3: Write barrier (if needed)
        if barrier_invoked:
            barrier_result = _write_barrier(
                user_id=resolved_user_id,
                pending_tracker=self._tracker,
                mem0_provider=self._mem0_provider,
                barrier_config=self.barrier_config,
            )
            # On timeout, we proceed with retrieval against current Mem0 state
            # (no raise — the warning inside _write_barrier is sufficient)

        # Step 4: Retrieve cross-agent shared memories
        if cross_agent:
            raw_memories = self._mem0_provider.get_all_shared_cross_agent(
                user_id=resolved_user_id,
                calling_agent=calling_agent,
            )
        else:
            raw_memories = []

        total_retrieved = len(raw_memories)

        # Step 5: Apply relevance threshold
        filtered = self._apply_relevance_threshold(raw_memories)

        # Step 6: Apply max_injected_memories cap
        filtered = self._apply_max_memories(filtered)

        # Step 7: Apply max_memory_tokens token cap
        filtered = self._apply_token_cap(filtered)

        # Step 8: Build diagnostics and return
        diagnostics: dict[str, Any] = {
            "user_id": resolved_user_id,
            "calling_agent": calling_agent,
            "cross_agent": cross_agent,
            "barrier_invoked": barrier_invoked,
            "barrier_result": barrier_result,
            "total_retrieved": total_retrieved,
            "post_filter_count": len(filtered),
        }

        return InjectionResult(
            memories=filtered,
            total_retrieved=total_retrieved,
            barrier_result=barrier_result,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_user_id(self, user_id: str) -> str | None:
        """Resolve the request-scoped user_id for retrieval.

        Returns the stripped user_id if it is a non-empty string, or None
        if the caller did not provide a valid request-scoped identity.
        No fallback logic is applied — there is no lookup of
        latest_user_id, known_user_ids, agent_name, or any other
        process-global state.

        Args:
            user_id: The user_id as provided by the caller.

        Returns:
            The stripped user_id string, or None if missing/empty/whitespace.
        """
        if not user_id or not str(user_id).strip():
            return None
        return str(user_id).strip()

    def _apply_relevance_threshold(
        self, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter memories below the configured relevance threshold.

        Memories without a ``relevance_score`` field are included (they are
        assumed to pass — this matches existing kernel behavior where
        ``get_all`` results don't carry relevance scores but are included).

        Args:
            memories: List of memory dicts to filter.

        Returns:
            Memories with relevance_score >= threshold, or no score field.
        """
        threshold = self.memory_config.relevance_threshold
        result = []
        for mem in memories:
            score = mem.get("relevance_score")
            if score is None or score >= threshold:
                result.append(mem)
        return result

    def _apply_max_memories(
        self, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Truncate to max_injected_memories count.

        Args:
            memories: List of memory dicts (already relevance-filtered).

        Returns:
            At most ``max_injected_memories`` memories.
        """
        max_count = self.memory_config.max_injected_memories
        return memories[:max_count]

    def _apply_token_cap(
        self, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Truncate memories to fit within max_memory_tokens.

        Uses a simple character-based approximation (4 chars ≈ 1 token)
        consistent with the existing kernel behavior.

        Args:
            memories: List of memory dicts (already count-capped).

        Returns:
            Memories fitting within the token budget.
        """
        max_tokens = self.memory_config.max_memory_tokens
        # Approximate: 4 characters ≈ 1 token
        max_chars = max_tokens * 4
        result = []
        total_chars = 0
        for mem in memories:
            content = mem.get("content", "")
            content_len = len(content) if isinstance(content, str) else len(str(content))
            if total_chars + content_len > max_chars and result:
                # Adding this memory would exceed the budget
                break
            result.append(mem)
            total_chars += content_len
        return result
