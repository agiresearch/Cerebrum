"""Idempotency Property — writeBarrier called twice is observationally equivalent to once.

**Validates: Requirements 2.2**

This test verifies that calling writeBarrier (via injector.inject()) twice in
succession for the same user_id is observationally equivalent to calling it
once:
1. No duplicate side effects (same number of memories in the store)
2. Identical retrieval input state (second call's injected memories match first)
3. Second call returns status="noop" with waited_ms < poll_interval_ms

Approach:
- Submit shared writes for a user_id and trigger injector.inject() (first call)
- After the first call returns with status="ok", immediately call
  injector.inject() again for the same user_id
- Assert: the second call's barrier_result has status="noop" and waited_ms
  is very small (< poll_interval_ms)
- Assert: the second call's injected memories are identical to the first call's
- Assert: no duplicate side effects (same number of memories in the store)

Uses hypothesis to generate various write sequences and latencies.
Uses low barrier_timeout_ms (500ms) and poll_interval_ms (10ms) for speed.

Run on the FIXED kernel — expected outcome: PASSES.
"""

import sys
sys.path.insert(0, ".")

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from aios.config.config_loader import KernelConfig, MemoryConfig, WriteBarrierConfig
from aios.memory.context_injector import ContextInjector
from aios.memory.mem0_provider import Mem0Provider
from aios.memory.pending_writes_tracker import PendingWritesTracker


# ---------------------------------------------------------------------------
# Mem0 Fake Client with configurable indexing latency
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory entry in the fake Mem0 store."""
    content: str
    user_id: str
    owner_agent: str
    sharing_policy: str
    memory_type: str = "profile"
    indexed: bool = False
    indexed_at: Optional[float] = None


class Mem0FakeClient:
    """Fake Mem0 client with configurable indexing latency.

    Simulates Mem0 store behavior: add() enqueues entries which become
    visible to get_all() only after the indexing latency elapses.
    """

    def __init__(self, indexing_latency_ms: float = 200.0):
        self.indexing_latency_ms = indexing_latency_ms
        self._store: List[MemoryEntry] = []
        self._lock = threading.Lock()

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> dict:
        """Enqueue a memory write. Visible after indexing_latency_ms."""
        metadata = metadata or {}
        entry = MemoryEntry(
            content=content,
            user_id=metadata.get("user_id", ""),
            owner_agent=metadata.get("owner_agent", ""),
            sharing_policy=metadata.get("sharing_policy", "private"),
            memory_type=metadata.get("memory_type", "profile"),
            indexed=False,
        )
        with self._lock:
            self._store.append(entry)

        # Schedule indexing completion after the configured latency
        def _mark_indexed():
            time.sleep(self.indexing_latency_ms / 1000.0)
            with self._lock:
                entry.indexed = True
                entry.indexed_at = time.time()

        t = threading.Thread(target=_mark_indexed, daemon=True)
        t.start()

        return {"status": "enqueued"}

    def get_all(self, user_id: str = "", **kwargs) -> List[Dict[str, Any]]:
        """Return only indexed (visible) entries for the given user_id."""
        with self._lock:
            results = []
            for e in self._store:
                if e.user_id == user_id and e.indexed:
                    results.append({
                        "content": e.content,
                        "metadata": {
                            "user_id": e.user_id,
                            "owner_agent": e.owner_agent,
                            "sharing_policy": e.sharing_policy,
                            "memory_type": e.memory_type,
                        },
                    })
            return results

    def count_entries(self, user_id: str = "") -> int:
        """Count all entries (indexed or not) for a user_id."""
        with self._lock:
            return sum(1 for e in self._store if e.user_id == user_id)

    def reset(self):
        """Clear all entries."""
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Helper: create a fresh fixed injector setup
# ---------------------------------------------------------------------------

def make_fixed_injector(
    indexing_latency_ms: float = 200.0,
    barrier_timeout_ms: int = 500,
    poll_interval_ms: int = 10,
):
    """Create a ContextInjector with write barrier using the Mem0FakeClient.

    Returns (injector, mem0_provider, tracker, client).
    """
    client = Mem0FakeClient(indexing_latency_ms=indexing_latency_ms)
    ptracker = PendingWritesTracker()
    provider = Mem0Provider(client)

    config = KernelConfig(
        memory=MemoryConfig(
            auto_inject=True,
            auto_extract=True,
            relevance_threshold=0.0,
            max_injected_memories=100,
            max_memory_tokens=100000,
            write_barrier=WriteBarrierConfig(
                enabled=True,
                timeout_ms=barrier_timeout_ms,
                poll_interval_ms=poll_interval_ms,
            ),
        )
    )

    injector = ContextInjector(
        mem0_provider=provider,
        config=config,
        pending_tracker=ptracker,
    )

    return injector, provider, ptracker, client


def add_memory(provider, ptracker, content, user_id, owner_agent, sharing_policy, memory_type="profile"):
    """Helper to add a memory through the provider and record in tracker."""
    metadata = {
        "user_id": user_id,
        "owner_agent": owner_agent,
        "sharing_policy": sharing_policy,
        "memory_type": memory_type,
    }
    provider._client.add(content, metadata=metadata)
    write_id = ptracker.record(
        user_id=user_id,
        owner_agent=owner_agent,
        sharing_policy=sharing_policy,
    )
    return write_id


# ---------------------------------------------------------------------------
# Strategies for hypothesis
# ---------------------------------------------------------------------------

_owner_agents = st.sampled_from(["profile_agent", "task_agent", "notes_agent"])
_user_ids = st.sampled_from(["alice", "bob", "charlie"])


@dataclass
class WriteSpec:
    """Specification for a single memory write."""
    owner_agent: str
    user_id: str
    content: str


_write_spec_strategy = st.builds(
    WriteSpec,
    owner_agent=_owner_agents,
    user_id=_user_ids,
    content=st.text(min_size=1, max_size=50),
)


# ---------------------------------------------------------------------------
# Idempotency Property — Hypothesis PBT
# ---------------------------------------------------------------------------

@given(
    writes=st.lists(_write_spec_strategy, min_size=1, max_size=4),
    indexing_latency_ms=st.floats(min_value=10.0, max_value=300.0),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_idempotency_property(writes, indexing_latency_ms):
    """Idempotency Property: writeBarrier called twice == called once.

    **Validates: Requirements 2.2**

    Given a sequence of shared writes for a user_id:
    1. First inject() call triggers the barrier, waits for writes, returns status="ok"
    2. Second inject() call for the same user_id sees no pending writes,
       returns status="noop" with waited_ms < poll_interval_ms
    3. Injected memories from both calls are identical
    4. No duplicate side effects in the store (entry count unchanged)
    """
    poll_interval_ms = 10
    barrier_timeout_ms = 500

    # Use the first write's user_id as the target
    target_user_id = writes[0].user_id

    # Filter writes to the target user_id
    target_writes = [w for w in writes if w.user_id == target_user_id]

    # Must have at least one write for the target user
    assume(len(target_writes) > 0)

    # All writes are shared from non-assistant agents (to trigger the barrier)
    # Filter out writes from assistant_agent since those won't trigger barrier
    cross_agent_writes = [w for w in target_writes if w.owner_agent != "assistant_agent"]
    assume(len(cross_agent_writes) > 0)

    # Set up the injector
    injector, provider, ptracker, client = make_fixed_injector(
        indexing_latency_ms=indexing_latency_ms,
        barrier_timeout_ms=barrier_timeout_ms,
        poll_interval_ms=poll_interval_ms,
    )

    # Submit shared writes for the target user_id
    for w in target_writes:
        add_memory(
            provider, ptracker,
            content=w.content,
            user_id=w.user_id,
            owner_agent=w.owner_agent,
            sharing_policy="shared",
        )

    # Count entries in store before first call
    store_count_before = client.count_entries(target_user_id)

    # --- First inject() call: barrier runs, waits for writes ---
    result_1 = injector.inject(
        user_id=target_user_id,
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # First call should have barrier_result with status="ok"
    assert result_1.barrier_result is not None, (
        "First call: barrier should have been invoked"
    )
    assert result_1.barrier_result.status == "ok", (
        f"First call: expected barrier status='ok', "
        f"got '{result_1.barrier_result.status}'"
    )

    # Count entries after first call (should be unchanged — barrier is read-only)
    store_count_after_first = client.count_entries(target_user_id)

    # --- Second inject() call: barrier should be a noop ---
    result_2 = injector.inject(
        user_id=target_user_id,
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # Count entries after second call
    store_count_after_second = client.count_entries(target_user_id)

    # --- Assertion 1: Second call returns status="noop" ---
    # After first barrier completes, all writes are acknowledged in the tracker.
    # The gating predicate should short-circuit (no pending writes) OR the
    # barrier itself returns noop immediately.
    if result_2.barrier_result is not None:
        # If barrier was invoked, it should be a noop (no pending writes)
        assert result_2.barrier_result.status == "noop", (
            f"Second call: expected barrier status='noop' (all writes already "
            f"acknowledged), got '{result_2.barrier_result.status}'. "
            f"waited_ms={result_2.barrier_result.waited_ms}, "
            f"residual_pending={result_2.barrier_result.residual_pending}"
        )
        # waited_ms should be very small (< poll_interval_ms)
        assert result_2.barrier_result.waited_ms < poll_interval_ms, (
            f"Second call: expected waited_ms < poll_interval_ms ({poll_interval_ms}), "
            f"got waited_ms={result_2.barrier_result.waited_ms}"
        )
    # If barrier_result is None, the gating predicate returned False
    # (no pending writes), which is equivalent to noop — also acceptable.

    # --- Assertion 2: Injected memories are identical ---
    content_set_1 = frozenset(m.get("content", "") for m in result_1.memories)
    content_set_2 = frozenset(m.get("content", "") for m in result_2.memories)
    assert content_set_1 == content_set_2, (
        f"Idempotency violated! Injected memories differ between calls. "
        f"First call: {content_set_1}, Second call: {content_set_2}"
    )

    # --- Assertion 3: No duplicate side effects ---
    # The store entry count should not change between calls
    assert store_count_before == store_count_after_first == store_count_after_second, (
        f"Duplicate side effects detected! Store entry count changed: "
        f"before={store_count_before}, after_first={store_count_after_first}, "
        f"after_second={store_count_after_second}"
    )


# ---------------------------------------------------------------------------
# Main runner (plain Python script per Cerebrum test conventions)
# ---------------------------------------------------------------------------

results = []


def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


def run_all():
    """Run the Idempotency Property tests.

    Expected: ALL tests PASS on the FIXED kernel.
    """
    print("=" * 70)
    print("Idempotency Property — writeBarrier Called Twice == Called Once")
    print("")
    print("EXPECTED: Test PASSES — second barrier call is a noop")
    print("Validates: Requirements 2.2")
    print("=" * 70)

    # Run the hypothesis PBT
    print("\n--- Hypothesis PBT: Idempotency Property (50 examples) ---")
    try:
        test_idempotency_property()
        record(
            "idempotency_property",
            True,
            "50 examples passed — second barrier call is always noop, "
            "identical results, no duplicate side effects",
        )
    except AssertionError as e:
        record("idempotency_property", False, str(e)[:300])
    except Exception as e:
        record("idempotency_property", False, f"Unexpected error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results)}")

    if failed == 0:
        print(f"\n  ALL {passed} tests PASSED — Idempotency property verified:")
        print("  Calling writeBarrier twice is observationally equivalent to")
        print("  calling it once (no duplicate side effects, identical state,")
        print("  second call returns noop).")
    else:
        print(f"\n  {failed} test(s) FAILED.")

    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    sys.exit(0 if all_passed else 1)
