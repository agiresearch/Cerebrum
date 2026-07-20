"""Verification tests: Property 1 bug-condition exploration tests on the FIXED kernel.

Property 1: Expected Behavior — Cross-Agent Retrieval Observes Prior Writes

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

These are the SAME sub-cases (a)–(e) from test_write_barrier_bug_condition.py,
but exercised against the FIXED ContextInjector (with write barrier +
PendingWritesTracker). All sub-cases MUST PASS, confirming the bug is fixed.

The Mem0Fake is adapted to work with the Mem0Provider interface:
- client.add(content, metadata=metadata) → stores with indexing latency
- client.get_all(user_id=user_id) → returns only indexed entries as dicts
"""

import sys
sys.path.insert(0, ".")

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from aios.config.config_loader import KernelConfig, MemoryConfig, WriteBarrierConfig
from aios.memory.context_injector import ContextInjector
from aios.memory.mem0_provider import Mem0Provider
from aios.memory.pending_writes_tracker import PendingWritesTracker


# ---------------------------------------------------------------------------
# Mem0 Fake Client adapted for Mem0Provider interface
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
    """Fake Mem0 client adapted for the Mem0Provider interface.

    The Mem0Provider calls:
    - self._client.add(content, metadata=metadata)
    - self._client.get_all(user_id=user_id)

    This fake simulates configurable indexing latency: add() enqueues
    but the entry is not visible to get_all() until after the latency.
    """

    def __init__(self, indexing_latency_ms: float = 200.0):
        self.indexing_latency_ms = indexing_latency_ms
        self._store: List[MemoryEntry] = []
        self._lock = threading.Lock()

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> dict:
        """Enqueue a memory write. Visible after indexing_latency_ms.

        Returns a dict (no completion signal) so the tracker relies on
        reconcile() to acknowledge writes.
        """
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

        # Return a plain dict — no completion signal, so the tracker
        # will use reconcile() in the barrier loop
        return {"status": "enqueued"}

    def get_all(self, user_id: str = "", **kwargs) -> List[Dict[str, Any]]:
        """Return only indexed (visible) entries for the given user_id.

        Returns dicts in the format expected by Mem0Provider.get_all():
        each entry has 'content' and 'metadata' keys.
        """
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
# Helper: create a fresh fixed injector setup
# ---------------------------------------------------------------------------

def make_fixed_injector(
    indexing_latency_ms: float = 200.0,
    barrier_timeout_ms: int = 2000,
    poll_interval_ms: int = 25,
):
    """Create a ContextInjector with write barrier using the Mem0FakeClient.

    Returns (injector, mem0_provider, tracker, client) for test use.
    """
    client = Mem0FakeClient(indexing_latency_ms=indexing_latency_ms)
    tracker = PendingWritesTracker()
    provider = Mem0Provider(client)

    config = KernelConfig(
        memory=MemoryConfig(
            auto_inject=True,
            auto_extract=True,
            relevance_threshold=0.0,  # No threshold filtering for these tests
            max_injected_memories=100,  # High cap for these tests
            max_memory_tokens=100000,  # High cap for these tests
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
        pending_tracker=tracker,
    )

    return injector, provider, tracker, client


def add_memory(provider, tracker, content, user_id, owner_agent, sharing_policy, memory_type="profile"):
    """Helper to add a memory through the provider (which records in tracker)."""
    # We need to monkey-patch the module-level tracker used by Mem0Provider.add()
    # Instead, we call the provider's add with metadata, but Mem0Provider uses
    # the module-level tracker. We need to use our own tracker.
    # The cleanest approach: record directly in our tracker and call the client.
    metadata = {
        "user_id": user_id,
        "owner_agent": owner_agent,
        "sharing_policy": sharing_policy,
        "memory_type": memory_type,
    }
    # Call the client directly to store
    provider._client.add(content, metadata=metadata)
    # Record in our tracker
    write_id = tracker.record(
        user_id=user_id,
        owner_agent=owner_agent,
        sharing_policy=sharing_policy,
    )
    return write_id


# ---------------------------------------------------------------------------
# Sub-case (a): Fast-inference race — NOW PASSES with write barrier
# ---------------------------------------------------------------------------

def test_a_fast_inference_race():
    """(a) Fast-inference race — write barrier ensures visibility.

    **Validates: Requirements 2.1, 2.2, 2.3**

    Mem0 fake configured with 200 ms indexing latency. Submit two shared
    writes (ProfileAgent, TaskAgent) for user_id="alice", then trigger
    AssistantAgent cross-agent retrieval 50 ms later.

    Assert: injected count == 2 (write barrier waits for indexing)
    EXPECTED on FIXED kernel: PASS
    """
    injector, provider, tracker, client = make_fixed_injector(
        indexing_latency_ms=200.0,
        barrier_timeout_ms=2000,
        poll_interval_ms=25,
    )

    # ProfileAgent writes a shared memory
    add_memory(
        provider, tracker,
        content='{"user_name": "Alice", "preferred_language": "Python"}',
        user_id="alice",
        owner_agent="profile_agent",
        sharing_policy="shared",
        memory_type="profile",
    )

    # TaskAgent writes a shared memory
    add_memory(
        provider, tracker,
        content='{"current_project": "web server", "goals": ["scale to 10k"]}',
        user_id="alice",
        owner_agent="task_agent",
        sharing_policy="shared",
        memory_type="task_context",
    )

    # Wait 50 ms (simulating fast inference window — retrieval dispatched
    # well before the 200 ms indexing latency completes)
    time.sleep(0.050)

    # AssistantAgent triggers cross-agent retrieval via the FIXED injector
    result = injector.inject(
        user_id="alice",
        calling_agent="assistant_agent",
        cross_agent=True,
    )
    injected_count = len(result.memories)

    # Assert: the write barrier ensures both writes are visible
    assert injected_count == 2, (
        f"Expected injected count == 2 (write barrier should wait for indexing), "
        f"got {injected_count}. Barrier result: {result.barrier_result}"
    )

    # Verify barrier was invoked
    assert result.barrier_result is not None, "Barrier should have been invoked"
    assert result.barrier_result.status == "ok", (
        f"Barrier should have completed successfully, got status={result.barrier_result.status}"
    )


# ---------------------------------------------------------------------------
# Sub-case (b): Slow-inference nondeterminism — NOW deterministic
# ---------------------------------------------------------------------------

def test_b_slow_inference_nondeterminism():
    """(b) Slow-inference — all iterations deterministically return count == 2.

    **Validates: Requirements 2.4, 2.5**

    Same writes, retrieval initiated 500 ms later. Run 50 iterations.
    Assert all 50 produce injected count == 2.

    EXPECTED on FIXED kernel: PASS — write barrier makes this deterministic.
    """
    import random

    successes = 0
    failures = 0
    iteration_results = []

    for i in range(50):
        # Use a latency that straddles the retrieval delay (same as unfixed test)
        latency_ms = random.uniform(100, 800)
        injector, provider, tracker, client = make_fixed_injector(
            indexing_latency_ms=latency_ms,
            barrier_timeout_ms=2000,
            poll_interval_ms=25,
        )

        add_memory(
            provider, tracker,
            content='{"user_name": "Alice"}',
            user_id="alice",
            owner_agent="profile_agent",
            sharing_policy="shared",
        )
        add_memory(
            provider, tracker,
            content='{"current_project": "web server"}',
            user_id="alice",
            owner_agent="task_agent",
            sharing_policy="shared",
        )

        # Retrieval after 500 ms (simulating slow inference window)
        time.sleep(0.500)

        result = injector.inject(
            user_id="alice",
            calling_agent="assistant_agent",
            cross_agent=True,
        )
        count = len(result.memories)
        iteration_results.append(count)

        if count == 2:
            successes += 1
        else:
            failures += 1

    # Assert ALL 50 produce count == 2 (deterministic with write barrier)
    all_two = all(r == 2 for r in iteration_results)
    assert all_two, (
        f"Expected all 50 iterations to produce injected count == 2, "
        f"but got {successes} successes and {failures} failures. "
        f"Results: {iteration_results[:10]}..."
    )


# ---------------------------------------------------------------------------
# Sub-case (c): Re-run determinism — NOW deterministic
# ---------------------------------------------------------------------------

def test_c_rerun_determinism():
    """(c) Re-run determinism — all 50 runs produce identical results.

    **Validates: Requirements 2.5**

    Fix latency=200 ms and inter-syscall delay=50 ms. Run the same sequence
    50 times. Assert all 50 outcomes are identical (count == 2).

    EXPECTED on FIXED kernel: PASS — write barrier ensures determinism.
    """
    outcomes = []

    for _ in range(50):
        injector, provider, tracker, client = make_fixed_injector(
            indexing_latency_ms=200.0,
            barrier_timeout_ms=2000,
            poll_interval_ms=25,
        )

        add_memory(
            provider, tracker,
            content='{"user_name": "Alice"}',
            user_id="alice",
            owner_agent="profile_agent",
            sharing_policy="shared",
        )

        # Small inter-syscall delay
        time.sleep(0.050)

        add_memory(
            provider, tracker,
            content='{"current_project": "web server"}',
            user_id="alice",
            owner_agent="task_agent",
            sharing_policy="shared",
        )

        # Retrieval 50 ms after last write (well before 200 ms indexing)
        time.sleep(0.050)

        result = injector.inject(
            user_id="alice",
            calling_agent="assistant_agent",
            cross_agent=True,
        )
        outcomes.append(len(result.memories))

    # Assert all outcomes are identical (deterministic behavior with barrier)
    unique_outcomes = set(outcomes)
    all_identical = len(unique_outcomes) == 1 and unique_outcomes == {2}

    assert all_identical, (
        f"Expected all 50 runs to produce identical result (injected count == 2), "
        f"but got {len(unique_outcomes)} distinct outcomes: {unique_outcomes}. "
        f"Distribution: count_0={outcomes.count(0)}, count_1={outcomes.count(1)}, "
        f"count_2={outcomes.count(2)}."
    )


# ---------------------------------------------------------------------------
# Sub-case (d): Mixed sharing policy — NOW correctly returns 2 shared
# ---------------------------------------------------------------------------

def test_d_mixed_sharing_policy():
    """(d) Mixed sharing policy — barrier waits, then filter applies correctly.

    **Validates: Requirements 2.1, 2.2**

    Three pending writes: 2 sharing_policy="shared", 1 sharing_policy="private".
    Assert: injected count == 2 (shared only) and private excluded.

    EXPECTED on FIXED kernel: PASS — barrier waits for indexing, sharing filter works.
    """
    injector, provider, tracker, client = make_fixed_injector(
        indexing_latency_ms=200.0,
        barrier_timeout_ms=2000,
        poll_interval_ms=25,
    )

    # Two shared writes
    add_memory(
        provider, tracker,
        content='{"user_name": "Alice"}',
        user_id="alice",
        owner_agent="profile_agent",
        sharing_policy="shared",
        memory_type="profile",
    )
    add_memory(
        provider, tracker,
        content='{"current_project": "web server"}',
        user_id="alice",
        owner_agent="task_agent",
        sharing_policy="shared",
        memory_type="task_context",
    )

    # One private write (should never be injected cross-agent)
    add_memory(
        provider, tracker,
        content='{"private_notes": "internal only"}',
        user_id="alice",
        owner_agent="notes_agent",
        sharing_policy="private",
        memory_type="profile",
    )

    # Retrieval 50 ms later (before 200 ms indexing completes)
    time.sleep(0.050)

    result = injector.inject(
        user_id="alice",
        calling_agent="assistant_agent",
        cross_agent=True,
    )
    injected_count = len(result.memories)

    # Verify private is excluded
    private_leaked = any(
        m.get("metadata", {}).get("sharing_policy") == "private"
        for m in result.memories
    )

    assert injected_count == 2 and not private_leaked, (
        f"Expected injected count == 2 (2 shared, private excluded), "
        f"got injected_count={injected_count}, private_leaked={private_leaked}. "
        f"Barrier result: {result.barrier_result}"
    )


# ---------------------------------------------------------------------------
# Sub-case (e): Hypothesis PBT — NOW passes on FIXED kernel
# ---------------------------------------------------------------------------

# Strategies for generating write sequences
_owner_agents = st.sampled_from(["profile_agent", "task_agent", "notes_agent"])
_sharing_policies = st.sampled_from(["shared", "private"])
_user_ids = st.sampled_from(["alice", "bob", "charlie"])


@dataclass
class WriteSpec:
    """Specification for a single memory write."""
    owner_agent: str
    sharing_policy: str
    user_id: str
    content: str = "test content"


_write_spec_strategy = st.builds(
    WriteSpec,
    owner_agent=_owner_agents,
    sharing_policy=_sharing_policies,
    user_id=_user_ids,
    content=st.text(min_size=1, max_size=50),
)


@given(
    writes=st.lists(_write_spec_strategy, min_size=1, max_size=5),
    indexing_latency_ms=st.floats(min_value=0.0, max_value=300.0),
    inter_syscall_delay_ms=st.floats(min_value=0.0, max_value=100.0),
    calling_agent=st.just("assistant_agent"),
)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_e_hypothesis_pbt_fixed(
    writes, indexing_latency_ms, inter_syscall_delay_ms, calling_agent,
):
    """(e) Hypothesis PBT — write barrier ensures correct injection.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    Generates 1–5 writes with random owner_agent, sharing_policy, user_id,
    random Mem0 latencies in [0, 300] ms, and random inter-syscall delays
    in [0, 100] ms.

    Asserts: post-call injected memory set equals the set of writes
    satisfying the sharing-policy filter (sharing_policy="shared" AND
    owner_agent != calling_agent AND user_id matches).

    EXPECTED on FIXED kernel: PASS — write barrier waits for indexing.
    """
    # Pick a target user_id from the writes (use the first write's user_id)
    target_user_id = writes[0].user_id

    # Filter writes to only those for the target user_id
    target_writes = [w for w in writes if w.user_id == target_user_id]

    # Skip if no writes match the target user
    assume(len(target_writes) > 0)

    # Calculate expected injected set: shared writes from other agents
    expected_injected = [
        w for w in target_writes
        if w.sharing_policy == "shared" and w.owner_agent != calling_agent
    ]

    # Skip trivial case where no shared cross-agent writes exist
    assume(len(expected_injected) > 0)

    # Only test with latencies that would create a race (latency > delay)
    # This ensures the bug condition is triggered (barrier must actually wait)
    assume(indexing_latency_ms > inter_syscall_delay_ms)

    # Set up the FIXED injector with write barrier
    injector, provider, tracker, client = make_fixed_injector(
        indexing_latency_ms=indexing_latency_ms,
        barrier_timeout_ms=2000,
        poll_interval_ms=25,
    )

    # Submit all writes for the target user
    for w in target_writes:
        add_memory(
            provider, tracker,
            content=w.content,
            user_id=w.user_id,
            owner_agent=w.owner_agent,
            sharing_policy=w.sharing_policy,
        )

    # Wait the inter-syscall delay before triggering retrieval
    if inter_syscall_delay_ms > 0:
        time.sleep(inter_syscall_delay_ms / 1000.0)

    # Trigger cross-agent retrieval (WITH write barrier on fixed kernel)
    result = injector.inject(
        user_id=target_user_id,
        calling_agent=calling_agent,
        cross_agent=True,
    )
    injected_count = len(result.memories)
    expected_count = len(expected_injected)

    # Assert: injected set equals expected set
    assert injected_count == expected_count, (
        f"Write barrier failed! "
        f"Expected injected count == {expected_count} "
        f"(shared cross-agent writes for user_id={target_user_id!r}), "
        f"got {injected_count}. "
        f"Writes: {[(w.owner_agent, w.sharing_policy) for w in target_writes]}, "
        f"indexing_latency_ms={indexing_latency_ms:.1f}, "
        f"inter_syscall_delay_ms={inter_syscall_delay_ms:.1f}. "
        f"Barrier result: {result.barrier_result}"
    )


# ---------------------------------------------------------------------------
# Main runner
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
    """Run all bug-condition verification tests on the FIXED kernel.

    All sub-cases (a)–(e) MUST PASS, confirming the write barrier fixes
    the cross-agent retrieval race condition.
    """
    print("=" * 70)
    print("Bug-Condition Verification Tests (Property 1) — FIXED Kernel")
    print("Cross-Agent Retrieval Observes Prior Writes")
    print("")
    print("EXPECTED: All tests PASS on the FIXED kernel.")
    print("Passing confirms the write-barrier fix resolves the race.")
    print("=" * 70)

    # (a) Fast-inference race
    print("\n--- (a) Fast-inference race (deterministic) ---")
    try:
        test_a_fast_inference_race()
        record("a_fast_inference_race", True, "Write barrier waited for indexing — PASS")
    except AssertionError as e:
        record("a_fast_inference_race", False, str(e)[:200])
    except Exception as e:
        record("a_fast_inference_race", False, f"Unexpected error: {e}")

    # (b) Slow-inference determinism
    print("\n--- (b) Slow-inference determinism (50 iterations) ---")
    try:
        test_b_slow_inference_nondeterminism()
        record("b_slow_inference_determinism", True, "All 50 iterations deterministic — PASS")
    except AssertionError as e:
        record("b_slow_inference_determinism", False, str(e)[:200])
    except Exception as e:
        record("b_slow_inference_determinism", False, f"Unexpected error: {e}")

    # (c) Re-run determinism
    print("\n--- (c) Re-run determinism (50 runs) ---")
    try:
        test_c_rerun_determinism()
        record("c_rerun_determinism", True, "All 50 runs identical — PASS")
    except AssertionError as e:
        record("c_rerun_determinism", False, str(e)[:200])
    except Exception as e:
        record("c_rerun_determinism", False, f"Unexpected error: {e}")

    # (d) Mixed sharing policy
    print("\n--- (d) Mixed sharing policy edge case ---")
    try:
        test_d_mixed_sharing_policy()
        record("d_mixed_sharing_policy", True, "2 shared injected, private excluded — PASS")
    except AssertionError as e:
        record("d_mixed_sharing_policy", False, str(e)[:200])
    except Exception as e:
        record("d_mixed_sharing_policy", False, f"Unexpected error: {e}")

    # (e) Hypothesis PBT
    print("\n--- (e) Hypothesis PBT (200 examples) ---")
    try:
        test_e_hypothesis_pbt_fixed()
        record("e_hypothesis_pbt_fixed", True, "200 examples passed without falsification — PASS")
    except AssertionError as e:
        record("e_hypothesis_pbt_fixed", False, str(e)[:200])
    except Exception as e:
        record("e_hypothesis_pbt_fixed", False, f"Hypothesis error: {e}")

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
        print(f"\n  ALL {passed} tests PASSED — the write-barrier fix resolves")
        print("  the cross-agent retrieval race condition (Property 1 verified).")
    else:
        print(f"\n  {failed} test(s) FAILED — write barrier may not be working correctly.")

    print("=" * 70)

    # Return True if all tests passed (expected for fixed kernel)
    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    # Exit 0 if all tests passed (bug fixed)
    # Exit 1 if any test failed (bug still present or barrier broken)
    sys.exit(0 if all_passed else 1)
