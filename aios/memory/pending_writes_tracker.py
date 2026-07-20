"""Pending Writes Tracker for the AIOS kernel memory subsystem.

Tracks in-flight writes to the Mem0 provider that have been submitted but not
yet confirmed as visible (committed and indexed). The write barrier in the
ContextInjector uses this tracker to determine when it is safe to proceed with
cross-agent retrieval.

Thread-safe: all mutations are guarded by a threading.Lock since the kernel
runs agent syscalls concurrently.

Requirements: 2.1, 2.2
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PendingWrite:
    """A record of a write submitted to Mem0Provider.add() but not yet visible.

    Attributes:
        write_id: Unique identifier for this pending write (uuid4).
        user_id: The user context this write belongs to.
        owner_agent: The agent that submitted the write (e.g. "profile_agent").
        sharing_policy: "shared" or "private".
        submitted_at_ms: Epoch milliseconds when the write was recorded.
    """

    write_id: str
    user_id: str
    owner_agent: str
    sharing_policy: str
    submitted_at_ms: int


class PendingWritesTracker:
    """In-memory tracker of pending (not-yet-visible) writes keyed by user_id.

    The tracker is append-only from the perspective of external callers: writes
    are recorded on Mem0Provider.add() and acknowledged either by an explicit
    completion signal or by the barrier-loop's reconcile() probe.

    This component is observation-only — it does not alter the write path itself.
    Standalone writes still flow through Mem0Provider.add unchanged.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id -> dict of write_id -> PendingWrite
        self._pending: dict[str, dict[str, PendingWrite]] = {}

    def record(self, user_id: str, owner_agent: str, sharing_policy: str) -> str:
        """Record a new pending write for the given user_id.

        Called by Mem0Provider.add() after enqueueing the write to Mem0.

        Args:
            user_id: The user context the write targets.
            owner_agent: The agent that issued the write.
            sharing_policy: "shared" or "private".

        Returns:
            The generated write_id (uuid4 string) for this pending write.
        """
        write_id = str(uuid.uuid4())
        submitted_at_ms = int(time.time() * 1000)
        pending_write = PendingWrite(
            write_id=write_id,
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
            submitted_at_ms=submitted_at_ms,
        )
        with self._lock:
            if user_id not in self._pending:
                self._pending[user_id] = {}
            self._pending[user_id][write_id] = pending_write
        return write_id

    def acknowledge(self, write_id: str) -> None:
        """Acknowledge that a pending write is now visible in Mem0.

        Called when the Mem0 client signals completion, or when reconcile()
        determines the write is visible via a get_all probe.

        Args:
            write_id: The write_id returned by record().
        """
        with self._lock:
            for user_id, writes in self._pending.items():
                if write_id in writes:
                    del writes[write_id]
                    # Clean up empty user entries
                    if not writes:
                        del self._pending[user_id]
                    return

    def pending_for(self, user_id: str) -> list[PendingWrite]:
        """Return the list of pending writes for the given user_id.

        Used by the gating predicate (shouldInvokeBarrier) and the barrier
        loop to determine if there are in-flight writes to wait on.

        Args:
            user_id: The user context to query.

        Returns:
            A list of PendingWrite records for the user_id, or an empty list
            if no writes are pending.
        """
        with self._lock:
            user_writes = self._pending.get(user_id, {})
            return list(user_writes.values())

    def reconcile(self, user_id: str, visible_memories: list[dict[str, Any]]) -> None:
        """Best-effort match pending writes against visible memories.

        When no explicit completion signal is available from the Mem0 client,
        the barrier loop calls this method with the result of
        get_all(user_id) to determine which pending writes have been indexed.

        Matching heuristic: a pending write is considered visible if any memory
        in visible_memories matches on (user_id, owner_agent, sharing_policy).
        This is a best-effort approach — content matching could be added for
        higher precision, but metadata matching is sufficient for the barrier's
        purpose of confirming writes are indexed.

        Args:
            user_id: The user context being reconciled.
            visible_memories: List of memory dicts returned by
                Mem0Provider.get_all(user_id). Each dict is expected to have
                a "metadata" key with at least "owner_agent" and
                "sharing_policy" fields, or top-level keys for those fields.
        """
        with self._lock:
            user_writes = self._pending.get(user_id)
            if not user_writes:
                return

            # Build a set of (owner_agent, sharing_policy) tuples from visible
            # memories for efficient lookup.
            visible_signatures: set[tuple[str, str]] = set()
            for memory in visible_memories:
                metadata = memory.get("metadata", {})
                # Support both nested metadata and flat top-level keys
                owner_agent = metadata.get("owner_agent") or memory.get("owner_agent", "")
                sharing_policy = (
                    metadata.get("sharing_policy") or memory.get("sharing_policy", "")
                )
                if owner_agent and sharing_policy:
                    visible_signatures.add((owner_agent, sharing_policy))

            # Acknowledge pending writes whose (owner_agent, sharing_policy)
            # pair appears in the visible set.
            to_remove: list[str] = []
            for write_id, pw in user_writes.items():
                if (pw.owner_agent, pw.sharing_policy) in visible_signatures:
                    to_remove.append(write_id)

            for write_id in to_remove:
                del user_writes[write_id]

            # Clean up empty user entries
            if not user_writes:
                del self._pending[user_id]


# Module-level singleton instance, constructed at import time.
# All kernel components share this single tracker instance.
tracker = PendingWritesTracker()
