"""Bug-condition exploration test for kernel-memory-write-barrier (Property 1).

This test simulates the UNFIXED AIOS kernel's ContextInjector + Mem0Provider
pipeline using two embedded fakes:

  * `Mem0Fake` — an in-process stand-in for `Mem0Provider` with configurable
    per-write indexing latency. After `add(...)` returns, the write is
    enqueued but only becomes visible to `get_all(...)` once the virtual
    clock has advanced past the write's `visible_at_ms`. This mirrors Mem0's
    real-world async pipeline (LLM extraction + dedup + embed + ChromaDB).

  * `UnfixedKernelSimulator` — mirrors the kernel's `ContextInjector` cross-
    agent retrieval path WITHOUT a write barrier. `llm_chat(...)` advances
    the virtual clock by `inter_syscall_delay_ms`, then runs `get_all(...)`
    + the existing sharing-policy filter (sharing_policy=="shared" AND
    owner_agent != calling_agent). No wait between writes and reads — the
    retrieval observes whatever Mem0 has indexed so far.

The simulator faithfully reproduces the structural race condition described
in `.kiro/specs/kernel-memory-write-barrier/bugfix.md`: when prior writes
have not yet been indexed by the time retrieval runs, cross-agent injection
sees zero (or partial) results.

This test is EXPECTED TO FAIL on the unfixed simulator. Each subcase failure
surfaces a concrete counterexample that confirms the write-barrier race.

Repository scope
----------------
The kernel itself lives in https://github.com/agiresearch/AIOS, NOT in this
repo. This file lives in Cerebrum so the bug condition can be explored
locally via the existing `python tests/agents/...` convention. The AIOS
team should mirror this test inside the kernel test suite at
`aios/tests/memory/test_write_barrier_bug_condition.py` (the Mem0 fake +
simulator can be replaced with the real `Mem0Provider` + `ContextInjector`
plus a controllable Mem0 latency injection).

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import random
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, ".")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


LatencyFn = Callable[[], float]
CALLING_AGENT = "assistant_agent"
OWNER_AGENTS = ("profile_agent", "task_agent", "notes_agent")
SHARING_POLICIES = ("shared", "private")


# ---------------------------------------------------------------------------
# Mem0 fake + kernel simulator
# ---------------------------------------------------------------------------

@dataclass
class _PendingMemory:
    """A write enqueued in the Mem0 fake.

    Visible to `get_all(user_id)` only once `clock_ms >= visible_at_ms`.
    """

    write_id: str
    user_id: str
    owner_agent: str
    sharing_policy: str
    content: str
    submitted_at_ms: float
    visible_at_ms: float


@dataclass
class Mem0Fake:
    """In-process fake for the kernel's Mem0Provider.

    Models Mem0's async indexing pipeline. `add(...)` enqueues the write
    and returns immediately; the write only becomes visible to `get_all`
    after `submitted_at_ms + indexing_latency_ms` on the virtual clock.

    All time is virtual: `clock_ms` only advances when the simulator
    explicitly calls `advance_clock(...)`. This makes timing deterministic
    and reproducible regardless of wall-clock wobble.
    """

    indexing_latency_ms: float = 200.0  # default per bugfix.md sub-case (a)
    clock_ms: float = 0.0
    _store: list[_PendingMemory] = field(default_factory=list)

    def add(
        self,
        *,
        user_id: str,
        owner_agent: str,
        sharing_policy: str,
        content: str,
    ) -> str:
        """Enqueue a write. Mirrors Mem0Provider.add(...) (fire-and-forget)."""
        write_id = str(uuid.uuid4())
        self._store.append(
            _PendingMemory(
                write_id=write_id,
                user_id=user_id,
                owner_agent=owner_agent,
                sharing_policy=sharing_policy,
                content=content,
                submitted_at_ms=self.clock_ms,
                visible_at_ms=self.clock_ms + self.indexing_latency_ms,
            )
        )
        return write_id

    def get_all(self, *, user_id: str) -> list[dict]:
        """Return all writes for `user_id` that are visible right now.

        Mirrors Mem0Provider.get_all(user_id=...). Writes with
        `visible_at_ms > clock_ms` are NOT returned — the indexer hasn't
        finished committing them yet.
        """
        return [
            {
                "id": m.write_id,
                "user_id": m.user_id,
                "owner_agent": m.owner_agent,
                "sharing_policy": m.sharing_policy,
                "content": m.content,
            }
            for m in self._store
            if m.user_id == user_id and m.visible_at_ms <= self.clock_ms
        ]

    def advance_clock(self, ms: float) -> None:
        """Advance the virtual clock by `ms` milliseconds."""
        if ms < 0:
            raise ValueError(f"clock advance must be non-negative; got {ms}")
        self.clock_ms += ms


@dataclass
class InjectionResult:
    """Result of a simulated cross-agent retrieval + injection."""

    injected: list[dict]
    cross_agent_count: int


class UnfixedKernelSimulator:
    """Simulates the UNFIXED AIOS kernel's ContextInjector cross-agent path.

    Mirrors `aios/memory/context_injector.py` retrieval logic on the
    unfixed kernel: resolve user_id, run get_all, apply the existing
    sharing-policy filter, return injected memories. There is NO write
    barrier — retrieval reads whatever Mem0 has indexed at the moment
    of the call.

    This is intentionally minimal and structural: it captures the same
    race that the production kernel exhibits, namely "issue retrieval
    before pending writes are visible".
    """

    def __init__(self, mem0: Mem0Fake) -> None:
        self.mem0 = mem0

    def llm_chat(
        self,
        *,
        user_id: str,
        calling_agent: str,
        inter_syscall_delay_ms: float,
    ) -> InjectionResult:
        """Trigger a cross-agent retrieval after `inter_syscall_delay_ms`.

        Models the time gap between the last `create_memory` syscall and
        the start of `AssistantAgent.llm_chat` (e.g., 50 ms for a fast
        scheduler, 500 ms for a slow one).
        """
        self.mem0.advance_clock(inter_syscall_delay_ms)

        visible = self.mem0.get_all(user_id=user_id)

        # Mirrors _apply_sharing_filter in Mem0Provider:
        # cross-agent injection only includes shared, non-self memories.
        cross_agent = [
            m
            for m in visible
            if m["sharing_policy"] == "shared"
            and m["owner_agent"] != calling_agent
        ]
        return InjectionResult(injected=cross_agent, cross_agent_count=len(cross_agent))


# ---------------------------------------------------------------------------
# Helpers — the "expected post-barrier" oracle
# ---------------------------------------------------------------------------

def _expected_cross_agent_set(
    writes: list[dict], *, user_id: str, calling_agent: str
) -> set[str]:
    """Return the set of write_ids that SHOULD be injected post-barrier.

    Mirrors the bug-condition predicate in `bugfix.md`:
      sharing_policy == "shared" AND owner_agent != calling_agent
    AND user_id matches.

    This is what the FIXED kernel SHALL inject regardless of latency.
    """
    return {
        w["write_id"]
        for w in writes
        if w["user_id"] == user_id
        and w["sharing_policy"] == "shared"
        and w["owner_agent"] != calling_agent
    }


def _run_pipeline(
    *,
    writes: list[dict],
    user_id: str,
    indexing_latency_ms: float,
    inter_syscall_delay_ms: float,
    calling_agent: str = CALLING_AGENT,
) -> tuple[InjectionResult, list[dict]]:
    """Drive the unfixed simulator with `writes` then trigger retrieval.

    Returns `(injection_result, writes_with_ids)` where
    `writes_with_ids[i]` is the input write augmented with the
    `write_id` returned by Mem0Fake.add.
    """
    mem0 = Mem0Fake(indexing_latency_ms=indexing_latency_ms)
    sim = UnfixedKernelSimulator(mem0)

    writes_with_ids: list[dict] = []
    for w in writes:
        write_id = mem0.add(
            user_id=w["user_id"],
            owner_agent=w["owner_agent"],
            sharing_policy=w["sharing_policy"],
            content=w["content"],
        )
        writes_with_ids.append({**w, "write_id": write_id})

    result = sim.llm_chat(
        user_id=user_id,
        calling_agent=calling_agent,
        inter_syscall_delay_ms=inter_syscall_delay_ms,
    )
    return result, writes_with_ids


# ---------------------------------------------------------------------------
# Result tracker
# ---------------------------------------------------------------------------

results: list[tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


# ---------------------------------------------------------------------------
# Sub-case (a) — Fast-inference race (deterministic, expected to FAIL)
# ---------------------------------------------------------------------------

def test_a_fast_inference_race() -> tuple[bool, str]:
    """(a) Fast-inference race: 200 ms latency, 50 ms inter-syscall delay.

    EXPECTED on UNFIXED simulator: FAIL with injected count == 0.
    Mirrors bugfix.md sub-case (a).

    Validates: Requirements 1.1, 1.2, 1.3
    """
    writes = [
        {
            "user_id": "alice",
            "owner_agent": "profile_agent",
            "sharing_policy": "shared",
            "content": "alice prefers dark mode",
        },
        {
            "user_id": "alice",
            "owner_agent": "task_agent",
            "sharing_policy": "shared",
            "content": "alice is debugging the auth flow",
        },
    ]
    result, _ = _run_pipeline(
        writes=writes,
        user_id="alice",
        indexing_latency_ms=200.0,
        inter_syscall_delay_ms=50.0,
    )
    expected_count = 2
    actual_count = result.cross_agent_count
    passed = actual_count == expected_count
    detail = (
        f"latency=200ms, delay=50ms, expected_injected={expected_count}, "
        f"actual_injected={actual_count}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (b) — Slow-inference nondeterminism (expected to FAIL across iters)
# ---------------------------------------------------------------------------

def test_b_slow_inference_nondeterminism(iterations: int = 50) -> tuple[bool, str]:
    """(b) Slow-inference nondeterminism: 200 ms latency, 500 ms delay.

    Run `iterations` times; assert all produce injected count == 2.

    EXPECTED on UNFIXED simulator: passes deterministically once delay
    exceeds latency. The bug surfaces when the delay distribution
    overlaps the latency tail. To capture the documented behavior, we
    also probe with random delays drawn from [0, 600] ms so the per-
    iteration outcomes vary, mirroring the observed ~27% qwen rate.

    Validates: Requirements 1.4, 1.5
    """
    rng = random.Random(0xBEEF)
    successes = 0
    failures = 0
    sample_failure: str = ""

    for i in range(iterations):
        # Span both sides of the indexing-latency boundary so the
        # outcome is genuinely mixed (this is what `bugfix.md` describes
        # as the "intermittent ~27% with slow inference" symptom).
        delay = rng.uniform(0.0, 600.0)
        writes = [
            {
                "user_id": "alice",
                "owner_agent": "profile_agent",
                "sharing_policy": "shared",
                "content": "alice prefers dark mode",
            },
            {
                "user_id": "alice",
                "owner_agent": "task_agent",
                "sharing_policy": "shared",
                "content": "alice is debugging auth",
            },
        ]
        result, _ = _run_pipeline(
            writes=writes,
            user_id="alice",
            indexing_latency_ms=200.0,
            inter_syscall_delay_ms=delay,
        )
        if result.cross_agent_count == 2:
            successes += 1
        else:
            failures += 1
            if not sample_failure:
                sample_failure = (
                    f"iter={i}, delay={delay:.1f}ms, "
                    f"injected={result.cross_agent_count}, expected=2"
                )

    # Property: ALL 50 iterations should inject 2 memories.
    passed = failures == 0
    detail = (
        f"iterations={iterations}, successes={successes}, failures={failures}; "
        f"first failure: {sample_failure or 'none'}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (c) — Re-run determinism (expected to FAIL)
# ---------------------------------------------------------------------------

def test_c_rerun_determinism(iterations: int = 50) -> tuple[bool, str]:
    """(c) Re-run determinism: fix latency=200ms and delay=50ms; run 50x.

    All outcomes must be identical. On the unfixed simulator the
    outcome IS deterministic at fixed parameters (always 0), but the
    deeper claim — that the outcome is independent of indexing speed —
    fails as soon as we permute latencies.

    EXPECTED on UNFIXED simulator: FAIL — the injected set varies as
    soon as the latency parameter is anything other than the magic
    fixed value, which proves the result is a function of timing
    rather than of the writes.

    Validates: Requirement 1.5
    """
    writes = [
        {
            "user_id": "alice",
            "owner_agent": "profile_agent",
            "sharing_policy": "shared",
            "content": "alice prefers dark mode",
        },
        {
            "user_id": "alice",
            "owner_agent": "task_agent",
            "sharing_policy": "shared",
            "content": "alice is debugging auth",
        },
    ]
    rng = random.Random(0xCAFE)
    outcomes: set[int] = set()
    samples: list[str] = []

    for i in range(iterations):
        # Latency drawn from the [0, 300] ms range called out in the
        # design doc — this is what the kernel sees in real Mem0
        # deployments depending on extraction-LLM warmth.
        latency = rng.uniform(0.0, 300.0)
        result, _ = _run_pipeline(
            writes=writes,
            user_id="alice",
            indexing_latency_ms=latency,
            inter_syscall_delay_ms=50.0,
        )
        outcomes.add(result.cross_agent_count)
        if len(samples) < 3 and result.cross_agent_count != 2:
            samples.append(
                f"iter={i}, latency={latency:.1f}ms, "
                f"delay=50ms, injected={result.cross_agent_count}"
            )

    # Property: outcome must be identical across all runs (i.e., a
    # single value in `outcomes`). On the unfixed simulator this set
    # contains both 0 and 2.
    passed = len(outcomes) == 1
    detail = (
        f"iterations={iterations}, distinct_outcomes={sorted(outcomes)}; "
        f"sample failures: {samples or 'none'}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (d) — Mixed sharing policy edge case (expected to FAIL)
# ---------------------------------------------------------------------------

def test_d_mixed_sharing_policy() -> tuple[bool, str]:
    """(d) Mixed sharing policy: 2 shared + 1 private write.

    EXPECTED on UNFIXED simulator: FAIL — usually injects 0 (race),
    sometimes 2 if delay > latency. Private must always be excluded.

    Validates: Requirement 1.2
    """
    writes = [
        {
            "user_id": "alice",
            "owner_agent": "profile_agent",
            "sharing_policy": "shared",
            "content": "alice prefers dark mode",
        },
        {
            "user_id": "alice",
            "owner_agent": "task_agent",
            "sharing_policy": "shared",
            "content": "alice is debugging auth",
        },
        {
            "user_id": "alice",
            "owner_agent": "notes_agent",
            "sharing_policy": "private",
            "content": "alice's private SSN: redacted",
        },
    ]
    result, _ = _run_pipeline(
        writes=writes,
        user_id="alice",
        indexing_latency_ms=200.0,
        inter_syscall_delay_ms=50.0,
    )
    no_private = all(m["sharing_policy"] != "private" for m in result.injected)
    expected_count = 2
    passed = result.cross_agent_count == expected_count and no_private
    detail = (
        f"latency=200ms, delay=50ms, expected_injected=2, "
        f"actual_injected={result.cross_agent_count}, "
        f"private_excluded={no_private}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (e) — Hypothesis PBT (expected to FAIL with shrunk counterexample)
# ---------------------------------------------------------------------------

# Strategy: 1–5 writes with random owner / sharing_policy / user_id.
_user_id_strategy = st.sampled_from(["alice", "bob", "carol", "dave"])
_owner_strategy = st.sampled_from(OWNER_AGENTS)
_policy_strategy = st.sampled_from(SHARING_POLICIES)

_write_strategy = st.fixed_dictionaries(
    {
        "user_id": _user_id_strategy,
        "owner_agent": _owner_strategy,
        "sharing_policy": _policy_strategy,
        "content": st.text(min_size=1, max_size=40),
    }
)

_writes_strategy = st.lists(_write_strategy, min_size=1, max_size=5)
_latency_strategy = st.floats(min_value=0.0, max_value=300.0)
_delay_strategy = st.floats(min_value=0.0, max_value=100.0)


@given(
    writes=_writes_strategy,
    target_user_id=_user_id_strategy,
    indexing_latency_ms=_latency_strategy,
    inter_syscall_delay_ms=_delay_strategy,
)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_e_hypothesis_property(
    writes: list[dict],
    target_user_id: str,
    indexing_latency_ms: float,
    inter_syscall_delay_ms: float,
) -> None:
    """(e) Hypothesis PBT — Property 1: post-call injected set equals
    the set of writes satisfying the bug-condition filter.

    For ANY write sequence + latency + delay, the cross-agent injected
    memory set MUST equal:
        { w | w.user_id == target_user_id
              AND w.sharing_policy == "shared"
              AND w.owner_agent != calling_agent }

    EXPECTED on UNFIXED simulator: FAILURE with hypothesis-shrunk
    counterexample showing a write enqueued but not visible at retrieval
    time.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """
    result, writes_with_ids = _run_pipeline(
        writes=writes,
        user_id=target_user_id,
        indexing_latency_ms=indexing_latency_ms,
        inter_syscall_delay_ms=inter_syscall_delay_ms,
    )

    expected_ids = _expected_cross_agent_set(
        writes_with_ids,
        user_id=target_user_id,
        calling_agent=CALLING_AGENT,
    )
    actual_ids = {m["id"] for m in result.injected}

    assert actual_ids == expected_ids, (
        f"injected set mismatch: "
        f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}; "
        f"latency={indexing_latency_ms:.1f}ms, "
        f"delay={inter_syscall_delay_ms:.1f}ms, "
        f"writes={[(w['owner_agent'], w['sharing_policy'], w['user_id']) for w in writes]}, "
        f"target_user_id={target_user_id}"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    """Run every sub-case. Each FAIL is a counterexample confirming the bug.

    EXPECTED OUTCOME on UNFIXED simulator: at least one (typically all
    five) sub-cases FAIL. Failure is the SUCCESS signal for an
    exploration test — it proves the write-barrier race exists.
    """
    print("=" * 78)
    print("Kernel write-barrier bug-condition exploration (Property 1)")
    print("Tests run against an UNFIXED kernel simulator.")
    print("FAILURE on the unfixed simulator IS the expected outcome — it")
    print("confirms the race documented in bugfix.md.")
    print("=" * 78)

    print("\n--- (a) Fast-inference race (200 ms latency, 50 ms delay) ---")
    passed_a, detail_a = test_a_fast_inference_race()
    record("a_fast_inference_race", passed_a, detail_a)

    print("\n--- (b) Slow-inference nondeterminism (50 iterations) ---")
    passed_b, detail_b = test_b_slow_inference_nondeterminism()
    record("b_slow_inference_nondeterminism", passed_b, detail_b)

    print("\n--- (c) Re-run determinism (50 iterations, randomized latency) ---")
    passed_c, detail_c = test_c_rerun_determinism()
    record("c_rerun_determinism", passed_c, detail_c)

    print("\n--- (d) Mixed sharing policy (2 shared + 1 private) ---")
    passed_d, detail_d = test_d_mixed_sharing_policy()
    record("d_mixed_sharing_policy", passed_d, detail_d)

    print("\n--- (e) Hypothesis PBT (200 examples) ---")
    try:
        test_e_hypothesis_property()
        record(
            "e_hypothesis_property",
            True,
            "All 200 hypothesis examples passed (no race surfaced)",
        )
    except AssertionError as e:
        msg = str(e).splitlines()[0]
        record(
            "e_hypothesis_property",
            False,
            f"Hypothesis shrunk a counterexample: {msg}",
        )
    except Exception as e:  # pragma: no cover - hypothesis infra error
        record(
            "e_hypothesis_property",
            False,
            f"Hypothesis raised non-assertion error: {type(e).__name__}: {e}",
        )
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — Bug-condition exploration")
    print("=" * 78)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed} (expected — each failure is a bug counterexample)")
    print(f"  Total:  {len(results)}")

    print("\nCounterexamples (each one demonstrates the write-barrier race):")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")

    print("=" * 78)

    # Exploration semantics:
    # - exit code 0 when failures exist (the bug was reproduced — success)
    # - exit code 1 when EVERY sub-case passed (the harness no longer
    #   reproduces the race; either the simulator drifted or someone
    #   accidentally fixed the unfixed-kernel model).
    return failed > 0


if __name__ == "__main__":
    bug_reproduced = run_all()
    if bug_reproduced:
        print("\nEXPLORATION RESULT: bug reproduced on unfixed simulator.")
        sys.exit(0)
    else:
        print(
            "\nEXPLORATION RESULT: bug NOT reproduced — investigate the simulator."
        )
        sys.exit(1)
