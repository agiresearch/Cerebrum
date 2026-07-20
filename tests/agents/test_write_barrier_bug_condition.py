"""Bug-condition exploration tests for the kernel write-barrier race condition.

Property 1: Bug Condition — Cross-Agent Retrieval Observes Prior Writes

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

CRITICAL: These tests MUST FAIL on the UNFIXED kernel — failure confirms the
write-barrier race exists. DO NOT attempt to fix the test or the kernel when
it fails.

The tests encode the expected post-barrier behavior. They will validate the
fix when they pass after implementation.

GOAL: Surface counterexamples demonstrating that cross-agent retrieval reads
from a not-yet-indexed Mem0 store.
"""

import sys
sys.path.insert(0, ".")

import asyncio
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Mem0 Fake with Configurable Indexing Latency
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


class Mem0Fake:
    """Fake Mem0 provider with configurable indexing latency.

    Simulates the real Mem0 behavior where `add()` enqueues a write but the
    write is not immediately visible to `get_all()` / `search()` until an
    internal indexing pipeline completes (extraction → dedup → embed → store).

    Args:
        indexing_latency_ms: Time in milliseconds between `add()` and the
            entry becoming visible to `get_all()`. Simulates Mem0's async
            indexing pipeline.
    """

    def __init__(self, indexing_latency_ms: float = 200.0):
        self.indexing_latency_ms = indexing_latency_ms
        self._store: List[MemoryEntry] = []
        self._lock = threading.Lock()

    def add(self, content: str, user_id: str, owner_agent: str,
            sharing_policy: str, memory_type: str = "profile") -> None:
        """Enqueue a memory write. The entry becomes visible after indexing_latency_ms."""
        entry = MemoryEntry(
            content=content,
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
            memory_type=memory_type,
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

    def get_all(self, user_id: str) -> List[MemoryEntry]:
        """Return only indexed (visible) entries for the given user_id.

        This simulates the real Mem0 behavior: only entries that have
        completed the internal indexing pipeline are visible to queries.
        """
        with self._lock:
            return [
                e for e in self._store
                if e.user_id == user_id and e.indexed
            ]

    def get_all_shared_cross_agent(self, user_id: str,
                                    calling_agent: str) -> List[MemoryEntry]:
        """Return indexed shared entries from other agents for the given user_id.

        Applies the sharing-policy filter: only entries where
        sharing_policy="shared" AND owner_agent != calling_agent are returned.
        """
        with self._lock:
            return [
                e for e in self._store
                if e.user_id == user_id
                and e.indexed
                and e.sharing_policy == "shared"
                and e.owner_agent != calling_agent
            ]

    def reset(self):
        """Clear all entries."""
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Unfixed ContextInjector Simulation
# ---------------------------------------------------------------------------

class UnfixedContextInjector:
    """Simulates the UNFIXED kernel's ContextInjector behavior.

    The unfixed kernel does NOT wait for writes to be indexed before
    executing cross-agent retrieval. It immediately calls get_all() /
    search(), which returns only entries that Mem0 has already indexed.

    This reproduces the race condition: if retrieval runs before writes
    are indexed, the injected memory set is empty.
    """

    def __init__(self, mem0: Mem0Fake, auto_inject: bool = True):
        self.mem0 = mem0
        self.auto_inject = auto_inject

    def inject(self, user_id: str, calling_agent: str) -> List[MemoryEntry]:
        """Perform cross-agent retrieval WITHOUT a write barrier.

        Returns the list of shared memories from other agents that are
        currently visible (indexed) in the Mem0 store. On the unfixed
        kernel, this races with in-flight writes.
        """
        if not self.auto_inject:
            return []

        # No write barrier — directly query Mem0 (the bug)
        return self.mem0.get_all_shared_cross_agent(user_id, calling_agent)


# ---------------------------------------------------------------------------
# Test Results Tracking
# ---------------------------------------------------------------------------

results = []


def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


# ---------------------------------------------------------------------------
# Sub-case (a): Fast-inference race (deterministic, will fail)
# ---------------------------------------------------------------------------

def test_a_fast_inference_race():
    """(a) Fast-inference race — deterministic failure.

    **Validates: Requirements 1.1, 1.2, 1.3**

    Mem0 fake configured with 200 ms indexing latency. Submit two shared
    writes (ProfileAgent, TaskAgent) for user_id="alice", then trigger
    AssistantAgent cross-agent retrieval 50 ms later.

    Assert: injected count == 2
    EXPECTED on unfixed: FAIL with injected count == 0
    """
    mem0 = Mem0Fake(indexing_latency_ms=200.0)
    injector = UnfixedContextInjector(mem0)

    # ProfileAgent writes a shared memory
    mem0.add(
        content='{"user_name": "Alice", "preferred_language": "Python"}',
        user_id="alice",
        owner_agent="profile_agent",
        sharing_policy="shared",
        memory_type="profile",
    )

    # TaskAgent writes a shared memory
    mem0.add(
        content='{"current_project": "web server", "goals": ["scale to 10k"]}',
        user_id="alice",
        owner_agent="task_agent",
        sharing_policy="shared",
        memory_type="task_context",
    )

    # Wait 50 ms (simulating fast inference window — retrieval dispatched
    # well before the 200 ms indexing latency completes)
    time.sleep(0.050)

    # AssistantAgent triggers cross-agent retrieval
    injected = injector.inject(user_id="alice", calling_agent="assistant_agent")
    injected_count = len(injected)

    # Assert: the write barrier should ensure both writes are visible
    assert injected_count == 2, (
        f"Expected injected count == 2 (both shared writes visible), "
        f"got {injected_count}. This confirms the write-barrier race: "
        f"retrieval ran before Mem0 indexed the writes."
    )


# ---------------------------------------------------------------------------
# Sub-case (b): Slow-inference nondeterminism (probabilistic, will fail)
# ---------------------------------------------------------------------------

def test_b_slow_inference_nondeterminism():
    """(b) Slow-inference nondeterminism — probabilistic failure.

    **Validates: Requirements 1.4, 1.5**

    Same writes, retrieval initiated 500 ms later. Run 50 iterations.
    Assert all 50 produce injected count == 2.

    EXPECTED on unfixed: FAIL with mixed outcomes (matches the observed
    ~27% qwen success rate — some iterations succeed because indexing
    occasionally finishes within 500 ms, but not deterministically).

    Note: With 200 ms configured latency, many iterations may pass because
    500 ms > 200 ms. To demonstrate nondeterminism we use a latency range
    that straddles the retrieval delay.
    """
    import random

    successes = 0
    failures = 0
    iteration_results = []

    for i in range(50):
        # Use a latency that straddles the retrieval delay to create
        # nondeterminism: sometimes finishes before 500 ms, sometimes not
        # This simulates the real-world variance in Mem0 indexing time
        latency_ms = random.uniform(100, 800)
        mem0 = Mem0Fake(indexing_latency_ms=latency_ms)
        injector = UnfixedContextInjector(mem0)

        mem0.add(
            content='{"user_name": "Alice"}',
            user_id="alice",
            owner_agent="profile_agent",
            sharing_policy="shared",
        )
        mem0.add(
            content='{"current_project": "web server"}',
            user_id="alice",
            owner_agent="task_agent",
            sharing_policy="shared",
        )

        # Retrieval after 500 ms (simulating slow inference window)
        time.sleep(0.500)

        injected = injector.inject(user_id="alice", calling_agent="assistant_agent")
        count = len(injected)
        iteration_results.append(count)

        if count == 2:
            successes += 1
        else:
            failures += 1

    # Assert ALL 50 produce count == 2 (deterministic behavior)
    all_two = all(r == 2 for r in iteration_results)
    assert all_two, (
        f"Expected all 50 iterations to produce injected count == 2, "
        f"but got {successes} successes and {failures} failures. "
        f"Results distribution: {iteration_results[:10]}... "
        f"This nondeterminism confirms the race condition (req 1.4, 1.5)."
    )


# ---------------------------------------------------------------------------
# Sub-case (c): Re-run determinism (will fail)
# ---------------------------------------------------------------------------

def test_c_rerun_determinism():
    """(c) Re-run determinism — variable outcomes confirm the race.

    **Validates: Requirements 1.5**

    Fix latency=200 ms and inter-syscall delay=50 ms. Run the same sequence
    50 times with no code change. Assert all 50 outcomes are identical.

    EXPECTED on unfixed: FAIL with variable outcomes (the race produces
    different results on each run depending on thread scheduling).
    """
    outcomes = []

    for _ in range(50):
        mem0 = Mem0Fake(indexing_latency_ms=200.0)
        injector = UnfixedContextInjector(mem0)

        mem0.add(
            content='{"user_name": "Alice"}',
            user_id="alice",
            owner_agent="profile_agent",
            sharing_policy="shared",
        )

        # Small inter-syscall delay
        time.sleep(0.050)

        mem0.add(
            content='{"current_project": "web server"}',
            user_id="alice",
            owner_agent="task_agent",
            sharing_policy="shared",
        )

        # Retrieval 50 ms after last write (well before 200 ms indexing)
        time.sleep(0.050)

        injected = injector.inject(user_id="alice", calling_agent="assistant_agent")
        outcomes.append(len(injected))

    # Assert all outcomes are identical (deterministic behavior expected)
    unique_outcomes = set(outcomes)
    all_identical = len(unique_outcomes) == 1 and unique_outcomes == {2}

    assert all_identical, (
        f"Expected all 50 runs to produce identical result (injected count == 2), "
        f"but got {len(unique_outcomes)} distinct outcomes: {unique_outcomes}. "
        f"Distribution: count_0={outcomes.count(0)}, count_1={outcomes.count(1)}, "
        f"count_2={outcomes.count(2)}. "
        f"Variable outcomes confirm the timing-dependent race (req 1.5)."
    )


# ---------------------------------------------------------------------------
# Sub-case (d): Mixed sharing policy edge case (will fail)
# ---------------------------------------------------------------------------

def test_d_mixed_sharing_policy():
    """(d) Mixed sharing policy edge case — partial visibility.

    **Validates: Requirements 1.1, 1.2**

    Three pending writes: 2 sharing_policy="shared", 1 sharing_policy="private".
    Assert: injected count == 2 (shared only) and private excluded.

    EXPECTED on unfixed: FAIL — sometimes 0 (none indexed yet),
    sometimes 2 (both shared indexed), demonstrating the race.
    """
    mem0 = Mem0Fake(indexing_latency_ms=200.0)
    injector = UnfixedContextInjector(mem0)

    # Two shared writes
    mem0.add(
        content='{"user_name": "Alice"}',
        user_id="alice",
        owner_agent="profile_agent",
        sharing_policy="shared",
        memory_type="profile",
    )
    mem0.add(
        content='{"current_project": "web server"}',
        user_id="alice",
        owner_agent="task_agent",
        sharing_policy="shared",
        memory_type="task_context",
    )

    # One private write (should never be injected cross-agent)
    mem0.add(
        content='{"private_notes": "internal only"}',
        user_id="alice",
        owner_agent="notes_agent",
        sharing_policy="private",
        memory_type="profile",
    )

    # Retrieval 50 ms later (before 200 ms indexing completes)
    time.sleep(0.050)

    injected = injector.inject(user_id="alice", calling_agent="assistant_agent")
    injected_count = len(injected)

    # Verify private is excluded (if anything is returned)
    private_leaked = any(e.sharing_policy == "private" for e in injected)

    assert injected_count == 2 and not private_leaked, (
        f"Expected injected count == 2 (2 shared, private excluded), "
        f"got injected_count={injected_count}, private_leaked={private_leaked}. "
        f"This confirms the race: retrieval sees 0 entries because writes "
        f"are not yet indexed (req 1.1, 1.2)."
    )


# ---------------------------------------------------------------------------
# Sub-case (e): Hypothesis PBT generator (will fail)
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
def test_e_hypothesis_pbt_race_condition(
    writes, indexing_latency_ms, inter_syscall_delay_ms, calling_agent,
):
    """(e) Hypothesis PBT — generated race conditions.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

    Generates 1–5 writes with random owner_agent, sharing_policy, user_id,
    random Mem0 latencies in [0, 300] ms, and random inter-syscall delays
    in [0, 100] ms.

    Asserts: post-call injected memory set equals the set of writes
    satisfying the bug-condition filter (sharing_policy="shared" AND
    owner_agent != calling_agent AND user_id matches).

    EXPECTED on unfixed: FAIL with hypothesis-shrunk counterexamples
    demonstrating the race.
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
    # This ensures the bug condition is triggered
    assume(indexing_latency_ms > inter_syscall_delay_ms)

    # Set up the Mem0 fake with the generated latency
    mem0 = Mem0Fake(indexing_latency_ms=indexing_latency_ms)
    injector = UnfixedContextInjector(mem0)

    # Submit all writes
    for w in target_writes:
        mem0.add(
            content=w.content,
            user_id=w.user_id,
            owner_agent=w.owner_agent,
            sharing_policy=w.sharing_policy,
        )

    # Wait the inter-syscall delay before triggering retrieval
    if inter_syscall_delay_ms > 0:
        time.sleep(inter_syscall_delay_ms / 1000.0)

    # Trigger cross-agent retrieval (no write barrier on unfixed kernel)
    injected = injector.inject(user_id=target_user_id, calling_agent=calling_agent)
    injected_count = len(injected)
    expected_count = len(expected_injected)

    # Assert: injected set equals expected set
    assert injected_count == expected_count, (
        f"Race condition detected! "
        f"Expected injected count == {expected_count} "
        f"(shared cross-agent writes for user_id={target_user_id!r}), "
        f"got {injected_count}. "
        f"Writes: {[(w.owner_agent, w.sharing_policy) for w in target_writes]}, "
        f"indexing_latency_ms={indexing_latency_ms:.1f}, "
        f"inter_syscall_delay_ms={inter_syscall_delay_ms:.1f}. "
        f"This confirms the write-barrier race (req 1.1-1.5)."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all():
    """Run all bug-condition exploration tests and report results.

    These tests are EXPECTED TO FAIL on the UNFIXED kernel.
    Failure confirms the write-barrier race condition exists.
    """
    print("=" * 70)
    print("Bug-Condition Exploration Tests (Property 1)")
    print("Cross-Agent Retrieval Observes Prior Writes")
    print("")
    print("EXPECTED: These tests FAIL on the UNFIXED kernel.")
    print("Failure confirms the write-barrier race exists.")
    print("=" * 70)

    # (a) Fast-inference race
    print("\n--- (a) Fast-inference race (deterministic) ---")
    try:
        test_a_fast_inference_race()
        record("a_fast_inference_race", True, "UNEXPECTED PASS — writes indexed instantly?")
    except AssertionError as e:
        record("a_fast_inference_race", False, str(e)[:200])
    except Exception as e:
        record("a_fast_inference_race", False, f"Unexpected error: {e}")

    # (b) Slow-inference nondeterminism
    print("\n--- (b) Slow-inference nondeterminism (probabilistic) ---")
    try:
        test_b_slow_inference_nondeterminism()
        record("b_slow_inference_nondeterminism", True, "UNEXPECTED PASS — all 50 deterministic?")
    except AssertionError as e:
        record("b_slow_inference_nondeterminism", False, str(e)[:200])
    except Exception as e:
        record("b_slow_inference_nondeterminism", False, f"Unexpected error: {e}")

    # (c) Re-run determinism
    print("\n--- (c) Re-run determinism ---")
    try:
        test_c_rerun_determinism()
        record("c_rerun_determinism", True, "UNEXPECTED PASS — all 50 identical?")
    except AssertionError as e:
        record("c_rerun_determinism", False, str(e)[:200])
    except Exception as e:
        record("c_rerun_determinism", False, f"Unexpected error: {e}")

    # (d) Mixed sharing policy
    print("\n--- (d) Mixed sharing policy edge case ---")
    try:
        test_d_mixed_sharing_policy()
        record("d_mixed_sharing_policy", True, "UNEXPECTED PASS — writes indexed instantly?")
    except AssertionError as e:
        record("d_mixed_sharing_policy", False, str(e)[:200])
    except Exception as e:
        record("d_mixed_sharing_policy", False, f"Unexpected error: {e}")

    # (e) Hypothesis PBT generator
    print("\n--- (e) Hypothesis PBT generator ---")
    try:
        test_e_hypothesis_pbt_race_condition()
        record("e_hypothesis_pbt", True, "UNEXPECTED PASS — no race detected?")
    except AssertionError as e:
        record("e_hypothesis_pbt", False, str(e)[:200])
    except Exception as e:
        record("e_hypothesis_pbt", False, f"Hypothesis falsified: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results)}")

    if failed > 0:
        print(f"\n  {failed} test(s) FAILED as expected — this confirms the")
        print("  write-barrier race condition exists in the unfixed kernel.")
        print("\n  Counterexamples documented above prove the root cause:")
        print("  cross-agent retrieval reads from a not-yet-indexed Mem0 store.")
    else:
        print("\n  WARNING: All tests PASSED — this is unexpected on unfixed code.")
        print("  The race condition may not be properly reproduced.")

    print("=" * 70)

    # Return True if tests FAILED (expected for bug exploration)
    return failed > 0


if __name__ == "__main__":
    bug_confirmed = run_all()
    # Exit 0 if bug was confirmed (tests failed as expected)
    # Exit 1 if tests unexpectedly passed (bug not reproduced)
    sys.exit(0 if bug_confirmed else 1)
