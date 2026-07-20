"""Mem0Provider — AIOS kernel adapter around the Mem0 library.

Wraps the Mem0 client to provide ``add``, ``get_all``, ``search``, and the
optional ``flush`` primitive.  Hooks into the ``PendingWritesTracker`` so that
the write barrier in the ContextInjector can determine when in-flight writes
become visible.

Thread-safety: each public method acquires no global locks beyond what the
tracker itself uses (the tracker is internally locked).  The underlying Mem0
client is assumed to be thread-safe per its own documentation.

Requirements: 2.1, 2.2
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from aios.memory.pending_writes_tracker import tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type for add() — backward-compatible NamedTuple
# ---------------------------------------------------------------------------


class AddResult(NamedTuple):
    """Result of a Mem0Provider.add() call.

    Callers that only destructure the first element (``result``) continue to
    work unchanged.  Callers that need the write_id for barrier tracking can
    access the second element.

    Attributes:
        result: The original return value from the underlying Mem0 client's
            ``add()`` method (dict, list, or whatever the client returns).
        write_id: The tracker-assigned identifier for this pending write, or
            None if tracking was not applicable (e.g., missing metadata).
    """

    result: Any
    write_id: str | None


# ---------------------------------------------------------------------------
# Mem0Provider
# ---------------------------------------------------------------------------


class Mem0Provider:
    """Kernel-side adapter around the Mem0 library client.

    Provides memory CRUD operations and exposes the optional ``flush()``
    primitive required by the write barrier.

    Args:
        client: The underlying Mem0 client instance (e.g., ``mem0.MemoryClient``
            or a test fake).  Must expose at minimum ``add(**kwargs)`` and
            ``get_all(user_id=...)`` methods.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Flush support
    # ------------------------------------------------------------------

    def supports_flush(self) -> bool:
        """Return True when the underlying Mem0 client exposes a synchronous flush.

        The write barrier prefers an explicit flush to drain in-flight writes
        before falling back to the bounded visibility poll.

        Returns:
            True if ``self._client`` has a callable ``flush`` attribute,
            False otherwise.
        """
        return callable(getattr(self._client, "flush", None))

    def flush(self, user_id: str, timeout_ms: int) -> None:
        """Flush pending writes for *user_id* via the underlying client.

        Delegates to ``self._client.flush(user_id=user_id, timeout_ms=timeout_ms)``
        when supported.  Acts as a no-op when the client does not expose a
        synchronous flush (callers should fall back to the bounded poll).

        Args:
            user_id: The user context whose pending writes should be flushed.
            timeout_ms: Maximum time in milliseconds to wait for the flush
                to complete.
        """
        if self.supports_flush():
            try:
                self._client.flush(user_id=user_id, timeout_ms=timeout_ms)
            except Exception:
                logger.warning(
                    "Mem0 client flush raised an exception for user_id=%s; "
                    "falling back to bounded poll.",
                    user_id,
                    exc_info=True,
                )
        # No-op path: client does not support flush — barrier will use poll.

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AddResult:
        """Add a memory to the Mem0 store and record the write in the tracker.

        After enqueueing the write to the underlying Mem0 client, this method
        records the write in the ``PendingWritesTracker`` so the write barrier
        can gate subsequent cross-agent retrievals until the write is visible.

        If the underlying Mem0 client exposes an internal completion callback,
        the tracker is acknowledged immediately.  Otherwise, acknowledgement
        is deferred to the barrier-loop's ``reconcile(...)`` probe path.

        Backward compatibility: the return type is a ``NamedTuple(result,
        write_id)``.  Callers that only use the first positional element (the
        original Mem0 return value) are unaffected.

        Args:
            content: The memory content string to store.
            metadata: Dict with at least ``user_id``, ``owner_agent``, and
                ``sharing_policy`` keys.  If metadata is missing or incomplete,
                the write still proceeds but tracking is skipped.
            **kwargs: Additional keyword arguments forwarded to the underlying
                Mem0 client ``add()`` call (e.g., ``user_id``).

        Returns:
            An ``AddResult`` NamedTuple with:
                - ``result``: The raw return value from the Mem0 client.
                - ``write_id``: The tracker write_id (str), or None if
                  tracking was not applicable.
        """
        # 1. Delegate to the underlying Mem0 client
        client_result = self._client.add(content, metadata=metadata, **kwargs)

        # 2. Record in tracker (if metadata is sufficient)
        write_id: str | None = None
        if metadata and all(
            k in metadata for k in ("user_id", "owner_agent", "sharing_policy")
        ):
            write_id = tracker.record(
                user_id=metadata["user_id"],
                owner_agent=metadata["owner_agent"],
                sharing_policy=metadata["sharing_policy"],
            )
            logger.debug(
                "Recorded pending write %s for user_id=%s (agent=%s, policy=%s)",
                write_id,
                metadata["user_id"],
                metadata["owner_agent"],
                metadata["sharing_policy"],
            )

            # 3. Attempt immediate acknowledgement if the client exposes a
            #    completion signal (e.g., a synchronous add that blocks until
            #    indexed).  Heuristic: if the client's add() returned a dict
            #    with a truthy "id" or "status" == "done", treat it as
            #    already-visible.
            if self._has_completion_signal(client_result):
                tracker.acknowledge(write_id)
                logger.debug(
                    "Immediately acknowledged write %s (client signalled completion)",
                    write_id,
                )

        return AddResult(result=client_result, write_id=write_id)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_all(self, user_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Retrieve all memories for *user_id* from the Mem0 store.

        This is the primary retrieval method used by the ContextInjector.
        The write barrier calls this to confirm pending writes are visible.

        Args:
            user_id: The user context to retrieve memories for.
            **kwargs: Additional keyword arguments forwarded to the client.

        Returns:
            A list of memory dicts, each containing at minimum ``content``
            and ``metadata`` keys.
        """
        result = self._client.get_all(user_id=user_id, **kwargs)
        # Normalize: some Mem0 versions return a dict with a "results" key
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        if isinstance(result, list):
            return result
        return []

    def get_all_shared_cross_agent(
        self,
        user_id: str,
        calling_agent: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve shared memories from other agents for cross-agent injection.

        Filters ``get_all(user_id)`` to return only memories that satisfy:
        - ``sharing_policy == "shared"``
        - ``owner_agent != calling_agent``

        This is the retrieval method the ContextInjector uses for cross-agent
        injection after the write barrier has completed.

        Args:
            user_id: The user context to retrieve memories for.
            calling_agent: The agent requesting injection (excluded from
                results since agents don't inject their own memories).
            **kwargs: Additional keyword arguments forwarded to the client.

        Returns:
            A list of memory dicts satisfying the cross-agent sharing filter.
        """
        all_memories = self.get_all(user_id=user_id, **kwargs)
        shared: list[dict[str, Any]] = []
        for memory in all_memories:
            meta = memory.get("metadata", {})
            # Support both nested metadata and flat top-level keys
            owner = meta.get("owner_agent") or memory.get("owner_agent", "")
            policy = meta.get("sharing_policy") or memory.get("sharing_policy", "")
            if policy == "shared" and owner != calling_agent:
                shared.append(memory)
        return shared

    def search(
        self,
        query: str,
        *,
        user_id: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search memories for *user_id* matching *query*.

        Args:
            query: The search query string.
            user_id: The user context to scope the search to.
            **kwargs: Additional keyword arguments forwarded to the client.

        Returns:
            A list of memory dicts ranked by relevance.
        """
        result = self._client.search(query=query, user_id=user_id, **kwargs)
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        if isinstance(result, list):
            return result
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_completion_signal(client_result: Any) -> bool:
        """Heuristic to detect if the Mem0 client signalled write completion.

        Some Mem0 client versions return a dict with status information
        indicating the write is already committed and indexed.  When this
        signal is present, the tracker can be acknowledged immediately
        rather than waiting for the reconcile probe.

        Args:
            client_result: The raw return value from ``client.add()``.

        Returns:
            True if the result indicates the write is already visible.
        """
        if not isinstance(client_result, dict):
            return False
        # Check for explicit status == "done" or "completed"
        status = client_result.get("status", "")
        if status in ("done", "completed", "indexed"):
            return True
        # Check for an explicit "indexed" flag
        if client_result.get("indexed") is True:
            return True
        return False
