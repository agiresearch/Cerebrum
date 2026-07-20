"""Property 3: Bounded Wait — Slow Indexer Cannot Stall the Pipeline.

**Validates: Requirements 2.1, 2.2 (bounded variant)**

This test verifies that when Mem0 indexing latency exceeds the configured
barrier_timeout_ms, the write barrier:
1. Returns within barrier_timeout_ms + poll_interval_ms + small_slack (bounded wait)
2. Emits a structured write_barrier_timeout warning with user_id and residual_pending
3. Allows retrieval to proceed (no exception, no hang)

Uses lower timeout/poll values for test speed:
- barrier_timeout_ms = 100 ms
- poll_interval_ms = 10 ms
- Mem0 fake latencies: 200–500 ms (always exceed timeout)

Run on the FIXED kernel — expected outcome: PASSES.
"""

import sys
sys.path.insert(0, ".")

import logging
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
# Mem0 Fake Client with configurable per-write latency
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

    Simulates a slow indexer that takes longer than the barrier timeout
    to make writes visible.
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

    def reset(self):
        """Clear all entries."""
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Helper: create a fresh fixed injector setup with LOW timeouts for speed
# ---------------------------------------------------------------------------

def make_fixed_injector(
    indexing_latency_ms: float = 300.0,
    barrier_timeout_ms: int = 100,
    poll_interval_ms: int = 10,
):
    """Create a ContextInjector with write barrier using the Mem0FakeClient.

    Uses LOW timeout/poll values for test speed. The indexing_latency_ms
    should EXCEED barrier_timeout_ms to trigger the timeout path.

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
# Log capture handler for verifying structured warnings
# ---------------------------------------------------------------------------

class WarningCapture(logging.Handler):
    """Logging handler that captures write_barrier_timeout warnings."""

    def __init__(self):
        super().__init__()
        self.warnings: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        if "write_barrier_timeout" in record.getMessage():
            self.warnings.append(record)

    def reset(self):
        self.warnings.clear()


# ---------------------------------------------------------------------------
# Property 3: Bounded Wait — Hypothesis PBT
# ---------------------------------------------------------------------------

# Strategy: generate latencies that ALWAYS exceed the barrier_timeout_ms (100ms)
# Range: [200, 500] ms — always exceeds the 100ms timeout
_slow_latency_strategy = st.integers(min_value=200, max_value=500)

# Strategy: 1-3 write sequences (keep small for speed since each test
# involves real sleeps up to ~110ms for the barrier timeout)
_num_writes_strategy = st.integers(min_value=1, max_value=3)


@given(
    indexing_latency_ms=_slow_latency_strategy,
    num_writes=_num_writes_strategy,
    user_id=st.sampled_from(["alice", "bob", "charlie"]),
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_bounded_wait_property(indexing_latency_ms, num_writes, user_id):
    """Property 3: Bounded Wait — barrier returns within timeout + ε.

    **Validates: Requirements 2.1, 2.2 (bounded variant)**

    Given write sequences with Mem0 latency exceeding barrier_timeout_ms:
    - Assert: barrier returns within barrier_timeout_ms + poll_interval_ms + slack
    - Assert: a structured write_barrier_timeout warning is emitted
    - Assert: retrieval proceeds (no exception, no hang)
    - Assert: barrier result has status="timeout" and residual_pending > 0
    """
    barrier_timeout_ms = 100
    poll_interval_ms = 10
    # Allow small slack for thread scheduling / time measurement overhead
    slack_ms = 50

    injector, provider, ptracker, client = make_fixed_injector(
        indexing_latency_ms=float(indexing_latency_ms),
        barrier_timeout_ms=barrier_timeout_ms,
        poll_interval_ms=poll_interval_ms,
    )

    # Set up log capture
    ci_logger = logging.getLogger("aios.memory.context_injector")
    capture = WarningCapture()
    ci_logger.addHandler(capture)
    ci_logger.setLevel(logging.WARNING)

    try:
        # Submit writes that will take longer than the barrier timeout to index
        for i in range(num_writes):
            add_memory(
                provider, ptracker,
                content=f'{{"data": "write_{i}"}}',
                user_id=user_id,
                owner_agent="profile_agent",
                sharing_policy="shared",
                memory_type="profile",
            )

        # Measure the time the injection takes (includes barrier wait)
        start_ms = time.time() * 1000

        # Trigger cross-agent retrieval — this invokes the write barrier
        # which should time out since latency > timeout
        result = injector.inject(
            user_id=user_id,
            calling_agent="assistant_agent",
            cross_agent=True,
        )

        elapsed_ms = time.time() * 1000 - start_ms

        # --- Assertion 1: Barrier returns within bounded time ---
        max_allowed_ms = barrier_timeout_ms + poll_interval_ms + slack_ms
        assert elapsed_ms <= max_allowed_ms, (
            f"Barrier exceeded bounded wait! "
            f"Elapsed {elapsed_ms:.1f} ms > allowed {max_allowed_ms} ms "
            f"(timeout={barrier_timeout_ms}, poll={poll_interval_ms}, slack={slack_ms}). "
            f"indexing_latency_ms={indexing_latency_ms}"
        )

        # --- Assertion 2: Barrier result indicates timeout ---
        assert result.barrier_result is not None, (
            "Barrier should have been invoked for pending shared cross-agent writes"
        )
        assert result.barrier_result.status == "timeout", (
            f"Expected barrier status='timeout' (latency {indexing_latency_ms}ms > "
            f"timeout {barrier_timeout_ms}ms), got status='{result.barrier_result.status}'"
        )
        assert result.barrier_result.residual_pending > 0, (
            f"Expected residual_pending > 0 on timeout, got "
            f"{result.barrier_result.residual_pending}"
        )

        # --- Assertion 3: Structured warning emitted ---
        assert len(capture.warnings) > 0, (
            f"Expected a write_barrier_timeout warning to be emitted, "
            f"but captured {len(capture.warnings)} warnings"
        )
        warning_msg = capture.warnings[0].getMessage()
        assert user_id in warning_msg, (
            f"Warning should contain user_id='{user_id}', got: {warning_msg}"
        )
        assert "residual_pending" in warning_msg, (
            f"Warning should contain 'residual_pending', got: {warning_msg}"
        )

        # --- Assertion 4: Retrieval proceeds (no exception, no hang) ---
        # The fact that we reached this point without exception proves no hang.
        # The result should be an InjectionResult (possibly with 0 memories
        # since the writes haven't indexed yet, but that's expected on timeout)
        assert result is not None, "Injection result should not be None"
        assert result.memories is not None, "Memories list should not be None"
        # Memories may be empty (writes not yet indexed) — that's acceptable
        # on timeout. The key property is: no exception, no infinite block.

    finally:
        # Clean up log handler
        ci_logger.removeHandler(capture)


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
    """Run the Property 3: Bounded Wait tests.

    Expected: ALL tests PASS on the FIXED kernel.
    """
    print("=" * 70)
    print("Property 3: Bounded Wait — Slow Indexer Cannot Stall the Pipeline")
    print("")
    print("EXPECTED: Test PASSES — pipeline never blocks indefinitely")
    print("Validates: Requirements 2.1, 2.2 (bounded variant)")
    print("=" * 70)

    # Run the hypothesis PBT
    print("\n--- Hypothesis PBT: Bounded Wait Property (30 examples) ---")
    try:
        test_bounded_wait_property()
        record(
            "bounded_wait_property",
            True,
            "30 examples passed — barrier always returns within timeout + ε",
        )
    except AssertionError as e:
        record("bounded_wait_property", False, str(e)[:300])
    except Exception as e:
        record("bounded_wait_property", False, f"Unexpected error: {e}")

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
        print(f"\n  ALL {passed} tests PASSED — Property 3 verified:")
        print("  The write barrier is bounded and never stalls the pipeline.")
    else:
        print(f"\n  {failed} test(s) FAILED.")

    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    sys.exit(0 if all_passed else 1)
