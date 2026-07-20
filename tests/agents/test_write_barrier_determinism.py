"""Property 4: Determinism Under Latency Permutation — Result Invariant Under Latency Permutation.

**Validates: Requirements 2.5**

This test verifies that for a FIXED write sequence and query, the injected
memory set is identical regardless of how long Mem0 takes to index each write
(as long as each latency is within barrier_timeout_ms). The write barrier
ensures that the injection result is a function of the writes and the query,
NOT of indexer timing.

Approach:
- Fix a specific write sequence (2 shared writes from profile_agent and
  task_agent for user_id "alice")
- Generate random Mem0 indexing latencies via hypothesis (each ≤ barrier_timeout_ms)
- For each latency permutation, run the full retrieval and capture the
  injected memory set
- Assert: the injected memory set is identical across all permutations
  (set equality, ignoring rank ties handled by the existing tie-breaker)

Uses lower timeout values for test speed:
- barrier_timeout_ms = 500 ms
- poll_interval_ms = 10 ms
- Latencies drawn from [10, 400] ms (all within timeout)

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
# Mem0 Fake Client with per-write configurable latency
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
    """Fake Mem0 client with per-write configurable indexing latency.

    Unlike the standard Mem0FakeClient that uses a single latency for all
    writes, this version accepts a list of latencies — one per write — to
    simulate different indexing durations for each memory entry.
    """

    def __init__(self, latencies_ms: Optional[List[float]] = None):
        """Initialize with a list of per-write latencies.

        Args:
            latencies_ms: List of latencies in ms, one per expected write.
                          If None or exhausted, defaults to 100ms.
        """
        self._latencies_ms = latencies_ms or []
        self._write_index = 0
        self._store: List[MemoryEntry] = []
        self._lock = threading.Lock()

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> dict:
        """Enqueue a memory write. Visible after the per-write latency."""
        metadata = metadata or {}
        entry = MemoryEntry(
            content=content,
            user_id=metadata.get("user_id", ""),
            owner_agent=metadata.get("owner_agent", ""),
            sharing_policy=metadata.get("sharing_policy", "private"),
            memory_type=metadata.get("memory_type", "profile"),
            indexed=False,
        )

        # Determine this write's latency
        with self._lock:
            if self._write_index < len(self._latencies_ms):
                latency_ms = self._latencies_ms[self._write_index]
            else:
                latency_ms = 100.0  # default fallback
            self._write_index += 1
            self._store.append(entry)

        # Schedule indexing completion after the per-write latency
        def _mark_indexed():
            time.sleep(latency_ms / 1000.0)
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

    def reset(self):
        """Clear all entries."""
        with self._lock:
            self._store.clear()
            self._write_index = 0


# ---------------------------------------------------------------------------
# Helper: create a fresh injector with per-write latencies
# ---------------------------------------------------------------------------

def make_determinism_injector(
    latencies_ms: List[float],
    barrier_timeout_ms: int = 500,
    poll_interval_ms: int = 10,
):
    """Create a ContextInjector with per-write latencies for determinism testing.

    Args:
        latencies_ms: List of indexing latencies (one per write).
        barrier_timeout_ms: Barrier timeout (all latencies should be below this).
        poll_interval_ms: Barrier poll interval.

    Returns:
        (injector, provider, tracker, client)
    """
    client = Mem0FakeClient(latencies_ms=latencies_ms)
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
# Fixed write sequence used across all permutations
# ---------------------------------------------------------------------------

FIXED_WRITES = [
    {
        "content": '{"user_name": "Alice", "preferred_language": "Python"}',
        "user_id": "alice",
        "owner_agent": "profile_agent",
        "sharing_policy": "shared",
        "memory_type": "profile",
    },
    {
        "content": '{"current_project": "web server", "goals": ["scale to 10k"]}',
        "user_id": "alice",
        "owner_agent": "task_agent",
        "sharing_policy": "shared",
        "memory_type": "task_context",
    },
]

# The expected injected content set (both writes are shared, cross-agent)
EXPECTED_CONTENT_SET = frozenset(w["content"] for w in FIXED_WRITES)


# ---------------------------------------------------------------------------
# Property 4: Determinism Under Latency Permutation — Hypothesis PBT
# ---------------------------------------------------------------------------

# Strategy: generate a list of 2 latencies (one per write), each in [10, 400]ms
# All within the 500ms barrier timeout, so the barrier should succeed.
_latency_strategy = st.lists(
    st.integers(min_value=10, max_value=400),
    min_size=2,
    max_size=2,
)


@given(
    latencies_ms=_latency_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_determinism_under_latency_permutation(latencies_ms):
    """Property 4: Determinism Under Latency Permutation.

    **Validates: Requirements 2.5**

    Fix a write sequence (2 shared writes from profile_agent and task_agent
    for user_id "alice") and a query (cross-agent retrieval by assistant_agent).

    Given permutations of Mem0 latencies (each ≤ barrier_timeout_ms, drawn
    from [10, 400] ms with barrier_timeout_ms=500):
    - Run the full retrieval and capture the injected memory set
    - Assert: the injected memory set is identical to the expected set
      (set equality) regardless of the latency permutation

    The write barrier ensures that injection is a function of writes and
    query, not of indexer timing.
    """
    barrier_timeout_ms = 500
    poll_interval_ms = 10

    # Create injector with this latency permutation
    injector, provider, ptracker, client = make_determinism_injector(
        latencies_ms=[float(l) for l in latencies_ms],
        barrier_timeout_ms=barrier_timeout_ms,
        poll_interval_ms=poll_interval_ms,
    )

    # Submit the FIXED write sequence
    for write_spec in FIXED_WRITES:
        add_memory(
            provider, ptracker,
            content=write_spec["content"],
            user_id=write_spec["user_id"],
            owner_agent=write_spec["owner_agent"],
            sharing_policy=write_spec["sharing_policy"],
            memory_type=write_spec["memory_type"],
        )

    # Trigger cross-agent retrieval immediately (before indexing completes)
    # The barrier should wait for all writes to become visible
    result = injector.inject(
        user_id="alice",
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # Extract the injected content set
    injected_content_set = frozenset(
        m.get("content", "") for m in result.memories
    )

    # Assert: injected set is identical to expected set regardless of latencies
    assert injected_content_set == EXPECTED_CONTENT_SET, (
        f"Determinism violated! Injected content differs from expected. "
        f"latencies_ms={latencies_ms}, "
        f"expected={EXPECTED_CONTENT_SET}, "
        f"got={injected_content_set}, "
        f"injected_count={len(result.memories)}, "
        f"barrier_result={result.barrier_result}"
    )

    # Assert: barrier was invoked and succeeded (not timed out)
    assert result.barrier_result is not None, (
        "Barrier should have been invoked for pending shared cross-agent writes"
    )
    assert result.barrier_result.status == "ok", (
        f"Expected barrier status='ok' (all latencies within timeout), "
        f"got status='{result.barrier_result.status}' with "
        f"waited_ms={result.barrier_result.waited_ms}, "
        f"residual_pending={result.barrier_result.residual_pending}, "
        f"latencies_ms={latencies_ms}"
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
    """Run the Property 4: Determinism Under Latency Permutation tests.

    Expected: ALL tests PASS on the FIXED kernel.
    """
    print("=" * 70)
    print("Property 4: Determinism Under Latency Permutation")
    print("Result Invariant Under Latency Permutation")
    print("")
    print("EXPECTED: Test PASSES — injection is a function of writes and")
    print("query, not of indexer timing")
    print("Validates: Requirements 2.5")
    print("=" * 70)

    # Run the hypothesis PBT
    print("\n--- Hypothesis PBT: Determinism Property (50 examples) ---")
    try:
        test_determinism_under_latency_permutation()
        record(
            "determinism_under_latency_permutation",
            True,
            "50 examples passed — injected set is invariant under latency permutation",
        )
    except AssertionError as e:
        record("determinism_under_latency_permutation", False, str(e)[:300])
    except Exception as e:
        record("determinism_under_latency_permutation", False, f"Unexpected error: {e}")

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
        print(f"\n  ALL {passed} tests PASSED — Property 4 verified:")
        print("  The write barrier ensures deterministic injection regardless")
        print("  of Mem0 indexing latency permutations.")
    else:
        print(f"\n  {failed} test(s) FAILED.")

    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    sys.exit(0 if all_passed else 1)
