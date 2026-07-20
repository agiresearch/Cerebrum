"""Fix verification test for kernel-memory-write-barrier (Property 1).

Companion to ``tests/agents/test_kernel_write_barrier_bug_condition.py``.
That file uses an ``UnfixedKernelSimulator`` and is EXPECTED to FAIL on
all five sub-cases (a)–(e) — failure is the success signal that the
race exists.

This file uses a ``FixedKernelSimulator`` that mirrors the post-fix
``ContextInjector`` semantics from
``.kiro/specs/kernel-memory-write-barrier/design.md`` and
``.kiro/specs/kernel-memory-write-barrier/kernel_artifacts/context_injector.py``:

  * ``PendingWritesTracker`` — embedded mirror of
    ``aios/memory/pending_writes_tracker.py``.  Per-``user_id`` set of
    in-flight writes; ``record(...)`` on every ``add(...)`` and
    ``reconcile(...)`` on each barrier-loop probe.
  * ``shouldInvokeBarrier(...)`` — the four short-circuit branches from
    the design (auto_inject=false → 3.4; not cross-agent → 3.3; no
    pending writes → 3.1; only private/self pending → 3.5) plus the
    ``write_barrier.enabled=False`` operator override.
  * ``writeBarrier(...)`` — bounded poll on
    ``tracker.pending_for(user_id)``.  Each iteration advances the
    Mem0 fake's *virtual* clock by ``poll_interval_ms`` so writes
    eventually become visible.  On timeout, returns
    ``BarrierResult(status="timeout", ...)`` and proceeds (Property 3).

Re-runs the five sub-cases (a)–(e) from the bug-condition exploration
test against the fixed simulator and adds a sixth sub-case (f) that
verifies ``BarrierResult.status == "noop"`` and ``waited_ms == 0`` on
the happy path (preservation 3.1 / clause 3.6).

EXPECTED OUTCOME on the FIXED simulator: ALL six sub-cases PASS.

Repository scope
----------------
The kernel itself lives in https://github.com/agiresearch/AIOS, NOT in
this repo. This file lives in Cerebrum so the fix's behavioural
contract can be verified locally via the existing
``python tests/agents/...`` convention. The AIOS team should mirror
this test inside the kernel test suite at
``aios/tests/memory/test_write_barrier_fix_verification.py`` (the
embedded simulator can be replaced with the real ``ContextInjector``
+ ``Mem0Provider`` + ``PendingWritesTracker``).

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
"""

from __future__ import annotations

import os
import random
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

# Allow `python tests/agents/test_kernel_write_barrier_fix_verification.py`
# from the repo root.
sys.path.insert(0, ".")

# Make sibling test modules importable so we can reuse Mem0Fake without
# duplicating the fake's per-write indexing-latency model.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from test_kernel_write_barrier_bug_condition import (  # noqa: E402
    CALLING_AGENT,
    OWNER_AGENTS,
    SHARING_POLICIES,
    Mem0Fake,
    _expected_cross_agent_set,
)


# ---------------------------------------------------------------------------
# PendingWritesTracker — embedded mirror of aios/memory/pending_writes_tracker.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingWrite:
    """A write submitted to the simulator's ``add(...)`` but not yet visible.

    Mirrors ``aios.memory.pending_writes_tracker.PendingWrite``.
    """

    write_id: str
    user_id: str
    owner_agent: str
    sharing_policy: str


class PendingWritesTracker:
    """In-memory tracker keyed by ``user_id``.

    Mirrors ``aios.memory.pending_writes_tracker.PendingWritesTracker``.
    Threadsafe (the kernel runs syscalls concurrently); locking kept
    here even though this in-process test is single-threaded so the
    structural mirror of the kernel implementation is intact.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, PendingWrite]] = {}

    def record(
        self,
        *,
        user_id: str,
        owner_agent: str,
        sharing_policy: str,
    ) -> str:
        """Record a new pending write for ``user_id`` and return its id."""
        write_id = str(uuid.uuid4())
        pw = PendingWrite(
            write_id=write_id,
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
        )
        with self._lock:
            self._pending.setdefault(user_id, {})[write_id] = pw
        return write_id

    def acknowledge(self, write_id: str) -> None:
        """Drop ``write_id`` from the pending set if present."""
        with self._lock:
            for user_id, writes in list(self._pending.items()):
                if write_id in writes:
                    del writes[write_id]
                    if not writes:
                        del self._pending[user_id]
                    return

    def pending_for(self, user_id: str) -> list[PendingWrite]:
        """Return a list snapshot of pending writes for ``user_id``."""
        with self._lock:
            return list(self._pending.get(user_id, {}).values())

    def reconcile(
        self, user_id: str, visible_memories: list[dict[str, Any]]
    ) -> None:
        """Best-effort acknowledge pending writes whose metadata is visible.

        Matching key is the ``(owner_agent, sharing_policy)`` pair, the
        same heuristic used by the kernel artifact's tracker.  The
        Mem0 fake returns flat top-level metadata (no ``"metadata"``
        sub-dict) so we read ``owner_agent`` / ``sharing_policy``
        directly off each visible memory.
        """
        with self._lock:
            user_writes = self._pending.get(user_id)
            if not user_writes:
                return

            visible_signatures: set[tuple[str, str]] = set()
            for memory in visible_memories:
                meta = memory.get("metadata", {})
                owner = meta.get("owner_agent") or memory.get("owner_agent", "")
                policy = (
                    meta.get("sharing_policy") or memory.get("sharing_policy", "")
                )
                if owner and policy:
                    visible_signatures.add((owner, policy))

            to_remove = [
                wid
                for wid, pw in user_writes.items()
                if (pw.owner_agent, pw.sharing_policy) in visible_signatures
            ]
            for wid in to_remove:
                del user_writes[wid]
            if not user_writes:
                del self._pending[user_id]


# ---------------------------------------------------------------------------
# BarrierResult + FixedKernelSimulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarrierResult:
    """Outcome of a single ``writeBarrier(...)`` invocation.

    Mirrors ``BarrierResult`` from the kernel artifact:

    * ``"noop"``    — barrier had nothing pending at entry
    * ``"ok"``      — all pending writes drained within the timeout
    * ``"timeout"`` — bounded wait elapsed; retrieval proceeds anyway
    """

    status: str
    waited_ms: int
    residual_pending: int


@dataclass
class FixedInjectionResult:
    """Result of a simulated post-barrier cross-agent retrieval."""

    injected: list[dict[str, Any]]
    cross_agent_count: int
    barrier_result: BarrierResult | None
    barrier_invoked: bool


class FixedKernelSimulator:
    """Simulates the FIXED AIOS kernel's ``ContextInjector`` cross-agent path.

    Mirrors the post-fix pipeline from the design and the kernel
    artifact:

      1. ``add(...)``  — Mem0Provider.add + tracker.record
      2. ``llm_chat`` — advance virtual clock by inter-syscall delay,
         run gating predicate, optionally invoke bounded write barrier,
         then run cross-agent retrieval through the existing sharing
         filter (``sharing_policy=="shared"`` AND
         ``owner_agent != calling_agent``).

    The bounded barrier loop drives the Mem0 fake's *virtual* clock
    forward by ``poll_interval_ms`` per iteration so writes become
    visible deterministically (no real ``time.sleep`` and no
    wall-clock dependency).
    """

    def __init__(
        self,
        mem0: Mem0Fake,
        *,
        auto_inject: bool = True,
        cross_agent_scope: bool = True,
        write_barrier_enabled: bool = True,
        barrier_timeout_ms: int = 2000,
        barrier_poll_interval_ms: int = 25,
    ) -> None:
        self.mem0 = mem0
        self.tracker = PendingWritesTracker()
        self.auto_inject = auto_inject
        self.cross_agent_scope = cross_agent_scope
        self.write_barrier_enabled = write_barrier_enabled
        self.barrier_timeout_ms = barrier_timeout_ms
        self.barrier_poll_interval_ms = barrier_poll_interval_ms

    # -- write path ----------------------------------------------------

    def add(
        self,
        *,
        user_id: str,
        owner_agent: str,
        sharing_policy: str,
        content: str,
    ) -> tuple[str, str]:
        """Mirror ``Mem0Provider.add(...)``: enqueue + tracker.record.

        Returns ``(mem0_write_id, tracker_write_id)``.
        """
        mem0_write_id = self.mem0.add(
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
            content=content,
        )
        tracker_write_id = self.tracker.record(
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
        )
        return mem0_write_id, tracker_write_id

    # -- gating predicate ---------------------------------------------

    def should_invoke_barrier(
        self,
        *,
        user_id: str,
        calling_agent: str,
    ) -> bool:
        """Implements the design's ``shouldInvokeBarrier(...)`` predicate.

        Each ``False`` branch maps 1:1 to a preservation clause:

        | branch                                | preservation |
        | ------------------------------------- | ------------ |
        | ``write_barrier.enabled = False``     | operator override |
        | ``auto_inject = False``               | 3.4 |
        | not cross-agent scope                 | 3.3 |
        | no pending writes                     | 3.1 |
        | only private/self pending writes      | 3.5 |
        """
        if not self.write_barrier_enabled:
            return False
        if not self.auto_inject:
            return False
        if not self.cross_agent_scope:
            return False
        pending = self.tracker.pending_for(user_id)
        if len(pending) == 0:
            return False
        cross_agent_pending = [
            w
            for w in pending
            if w.sharing_policy == "shared" and w.owner_agent != calling_agent
        ]
        if len(cross_agent_pending) == 0:
            return False
        return True

    # -- bounded-wait barrier -----------------------------------------

    def write_barrier(self, *, user_id: str) -> BarrierResult:
        """Bounded visibility poll; advances the *virtual* clock per iter.

        Mirrors the design's ``writeBarrier(...)`` minus the optional
        ``flush()`` branch (the Mem0 fake does not expose a flush
        primitive — exactly the configuration the design's fallback
        path was specified for).
        """
        start_clock = self.mem0.clock_ms
        expected = len(self.tracker.pending_for(user_id))
        if expected == 0:
            # 3.6 happy path: barrier short-circuits when nothing pending.
            return BarrierResult(status="noop", waited_ms=0, residual_pending=0)

        while True:
            visible = self.mem0.get_all(user_id=user_id)
            self.tracker.reconcile(user_id, visible)

            residual = len(self.tracker.pending_for(user_id))
            if residual == 0:
                return BarrierResult(
                    status="ok",
                    waited_ms=int(self.mem0.clock_ms - start_clock),
                    residual_pending=0,
                )

            elapsed = self.mem0.clock_ms - start_clock
            if elapsed >= self.barrier_timeout_ms:
                return BarrierResult(
                    status="timeout",
                    waited_ms=int(elapsed),
                    residual_pending=residual,
                )

            # Drive the Mem0 fake's virtual clock forward so the next
            # poll iteration may see writes whose visible_at_ms has
            # been crossed.  In the live kernel this is a real
            # time.sleep; here we advance the simulated clock.
            self.mem0.advance_clock(self.barrier_poll_interval_ms)

    # -- pipeline entry point -----------------------------------------

    def llm_chat(
        self,
        *,
        user_id: str,
        calling_agent: str,
        inter_syscall_delay_ms: float,
    ) -> FixedInjectionResult:
        """Drive the post-fix injector pipeline.

        Steps (mirroring the kernel artifact's ``ContextInjector.inject``):
            1. Inter-syscall delay (advance virtual clock).
            2. Gating predicate ``shouldInvokeBarrier(...)``.
            3. Bounded ``writeBarrier(...)`` if predicate is True.
            4. Cross-agent retrieval via Mem0.get_all + sharing filter.
        """
        self.mem0.advance_clock(inter_syscall_delay_ms)

        barrier_result: BarrierResult | None = None
        invoked = self.should_invoke_barrier(
            user_id=user_id, calling_agent=calling_agent
        )
        if invoked:
            barrier_result = self.write_barrier(user_id=user_id)
            # On status="timeout" we deliberately do NOT raise — the
            # surrounding retrieval still runs against whatever Mem0
            # has committed (Property 3, Bounded Wait).

        visible = self.mem0.get_all(user_id=user_id)
        cross_agent = [
            m
            for m in visible
            if m["sharing_policy"] == "shared" and m["owner_agent"] != calling_agent
        ]
        return FixedInjectionResult(
            injected=cross_agent,
            cross_agent_count=len(cross_agent),
            barrier_result=barrier_result,
            barrier_invoked=invoked,
        )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

results: list[tuple[str, str, str]] = []
barrier_timings: list[tuple[str, BarrierResult | None]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


def _record_barrier(case: str, br: BarrierResult | None) -> None:
    barrier_timings.append((case, br))


def _run_fixed_pipeline(
    *,
    writes: list[dict[str, Any]],
    user_id: str,
    indexing_latency_ms: float,
    inter_syscall_delay_ms: float,
    calling_agent: str = CALLING_AGENT,
    barrier_timeout_ms: int = 2000,
    barrier_poll_interval_ms: int = 25,
) -> tuple[FixedInjectionResult, list[dict[str, Any]]]:
    """Drive the FixedKernelSimulator end-to-end."""
    mem0 = Mem0Fake(indexing_latency_ms=indexing_latency_ms)
    sim = FixedKernelSimulator(
        mem0,
        barrier_timeout_ms=barrier_timeout_ms,
        barrier_poll_interval_ms=barrier_poll_interval_ms,
    )

    writes_with_ids: list[dict[str, Any]] = []
    for w in writes:
        mem0_id, _ = sim.add(
            user_id=w["user_id"],
            owner_agent=w["owner_agent"],
            sharing_policy=w["sharing_policy"],
            content=w["content"],
        )
        writes_with_ids.append({**w, "write_id": mem0_id})

    result = sim.llm_chat(
        user_id=user_id,
        calling_agent=calling_agent,
        inter_syscall_delay_ms=inter_syscall_delay_ms,
    )
    return result, writes_with_ids


# ---------------------------------------------------------------------------
# Sub-case (a) — Fast-inference race, FIXED kernel
# ---------------------------------------------------------------------------


def test_a_fast_inference_race() -> tuple[bool, str]:
    """(a) FIXED Property 1: 200 ms latency, 50 ms inter-syscall delay.

    EXPECTED on FIXED simulator: PASS with injected count == 2.

    Validates: Requirements 2.1, 2.3
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
    result, _ = _run_fixed_pipeline(
        writes=writes,
        user_id="alice",
        indexing_latency_ms=200.0,
        inter_syscall_delay_ms=50.0,
    )
    _record_barrier("a_fast_inference_race", result.barrier_result)
    expected_count = 2
    actual_count = result.cross_agent_count
    passed = actual_count == expected_count
    detail = (
        f"latency=200ms, delay=50ms, expected_injected={expected_count}, "
        f"actual_injected={actual_count}, "
        f"barrier={_fmt_barrier(result.barrier_result)}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (b) — Slow-inference determinism, FIXED kernel
# ---------------------------------------------------------------------------


def test_b_slow_inference_determinism(iterations: int = 50) -> tuple[bool, str]:
    """(b) FIXED Property 1: 200 ms latency, randomized inter-syscall delay.

    Run ``iterations`` times.  All must produce injected count == 2.

    EXPECTED on FIXED simulator: PASS for every iteration regardless of
    delay distribution.

    Validates: Requirement 2.4
    """
    rng = random.Random(0xBEEF)  # same seed as exploration test
    successes = 0
    failures = 0
    sample_failure: str = ""
    statuses: dict[str, int] = {}
    max_waited_ms = 0

    for i in range(iterations):
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
        result, _ = _run_fixed_pipeline(
            writes=writes,
            user_id="alice",
            indexing_latency_ms=200.0,
            inter_syscall_delay_ms=delay,
        )
        if result.barrier_result is not None:
            statuses[result.barrier_result.status] = (
                statuses.get(result.barrier_result.status, 0) + 1
            )
            max_waited_ms = max(max_waited_ms, result.barrier_result.waited_ms)
        if result.cross_agent_count == 2:
            successes += 1
        else:
            failures += 1
            if not sample_failure:
                sample_failure = (
                    f"iter={i}, delay={delay:.1f}ms, "
                    f"injected={result.cross_agent_count}, expected=2, "
                    f"barrier={_fmt_barrier(result.barrier_result)}"
                )

    passed = failures == 0
    detail = (
        f"iterations={iterations}, successes={successes}, failures={failures}, "
        f"barrier_statuses={statuses}, max_waited_ms={max_waited_ms}; "
        f"first failure: {sample_failure or 'none'}"
    )
    _record_barrier(
        "b_slow_inference_determinism",
        BarrierResult(
            status="ok" if passed else "mixed",
            waited_ms=max_waited_ms,
            residual_pending=0,
        ),
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (c) — Re-run determinism, FIXED kernel
# ---------------------------------------------------------------------------


def test_c_rerun_determinism(iterations: int = 50) -> tuple[bool, str]:
    """(c) FIXED Property 4: invariant under latency permutation.

    Fix the write sequence; randomize indexing latency in [0, 300] ms;
    re-run 50 times.  All injected counts must be identical.

    EXPECTED on FIXED simulator: outcomes set has cardinality 1
    (always 2).

    Validates: Requirement 2.5
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
    rng = random.Random(0xCAFE)  # same seed as exploration test
    outcomes: set[int] = set()
    samples: list[str] = []
    max_waited_ms = 0

    for i in range(iterations):
        latency = rng.uniform(0.0, 300.0)
        result, _ = _run_fixed_pipeline(
            writes=writes,
            user_id="alice",
            indexing_latency_ms=latency,
            inter_syscall_delay_ms=50.0,
        )
        outcomes.add(result.cross_agent_count)
        if result.barrier_result is not None:
            max_waited_ms = max(max_waited_ms, result.barrier_result.waited_ms)
        if len(samples) < 3 and result.cross_agent_count != 2:
            samples.append(
                f"iter={i}, latency={latency:.1f}ms, "
                f"delay=50ms, injected={result.cross_agent_count}, "
                f"barrier={_fmt_barrier(result.barrier_result)}"
            )

    passed = len(outcomes) == 1 and 2 in outcomes
    detail = (
        f"iterations={iterations}, distinct_outcomes={sorted(outcomes)}, "
        f"max_waited_ms={max_waited_ms}; "
        f"sample failures: {samples or 'none'}"
    )
    _record_barrier(
        "c_rerun_determinism",
        BarrierResult(
            status="ok" if passed else "mixed",
            waited_ms=max_waited_ms,
            residual_pending=0,
        ),
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (d) — Mixed sharing policy, FIXED kernel
# ---------------------------------------------------------------------------


def test_d_mixed_sharing_policy() -> tuple[bool, str]:
    """(d) FIXED Property 1 + preservation 3.5: 2 shared + 1 private.

    EXPECTED on FIXED simulator: injected count == 2, private excluded.

    Validates: Requirement 2.2 + preservation 3.5
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
    result, _ = _run_fixed_pipeline(
        writes=writes,
        user_id="alice",
        indexing_latency_ms=200.0,
        inter_syscall_delay_ms=50.0,
    )
    _record_barrier("d_mixed_sharing_policy", result.barrier_result)
    no_private = all(m["sharing_policy"] != "private" for m in result.injected)
    expected_count = 2
    passed = result.cross_agent_count == expected_count and no_private
    detail = (
        f"latency=200ms, delay=50ms, expected_injected=2, "
        f"actual_injected={result.cross_agent_count}, "
        f"private_excluded={no_private}, "
        f"barrier={_fmt_barrier(result.barrier_result)}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (e) — Hypothesis PBT (Property 1, FIXED kernel)
# ---------------------------------------------------------------------------

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
    writes: list[dict[str, Any]],
    target_user_id: str,
    indexing_latency_ms: float,
    inter_syscall_delay_ms: float,
) -> None:
    """(e) FIXED Property 1 — post-call injected set equals the oracle.

    For ANY write sequence + latency + delay, the FixedKernelSimulator
    SHALL produce a cross-agent injected set equal to::

        { w | w.user_id == target_user_id
              AND w.sharing_policy == "shared"
              AND w.owner_agent != calling_agent }

    EXPECTED on FIXED simulator: 200 examples pass without falsification.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
    """
    result, writes_with_ids = _run_fixed_pipeline(
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
        f"target_user_id={target_user_id}, "
        f"barrier={_fmt_barrier(result.barrier_result)}"
    )


# ---------------------------------------------------------------------------
# Sub-case (f) — Barrier no-op on the happy path
# ---------------------------------------------------------------------------


def test_f_barrier_noop_on_empty_pending() -> tuple[bool, str]:
    """(f) FIXED preservation 3.1 / 3.6: barrier is a no-op when nothing pending.

    Two checks:

      1. ``llm_chat`` for a user with no prior writes must NOT invoke
         the barrier (``barrier_invoked == False``, ``barrier_result is
         None``) — this is the gating-predicate short-circuit.
      2. Direct invocation of ``write_barrier(user_id=...)`` against an
         empty tracker MUST return ``BarrierResult(status="noop",
         waited_ms=0, residual_pending=0)`` — this is the in-barrier
         defensive short-circuit at the top of the loop.

    Validates: preservation 3.1, 3.6
    """
    mem0 = Mem0Fake(indexing_latency_ms=200.0)
    sim = FixedKernelSimulator(mem0)

    # (1) Gate-level short-circuit: no writes for "alice" anywhere.
    result = sim.llm_chat(
        user_id="alice",
        calling_agent=CALLING_AGENT,
        inter_syscall_delay_ms=0.0,
    )
    gate_short_circuited = (
        not result.barrier_invoked and result.barrier_result is None
    )

    # (2) Defensive in-barrier short-circuit: call write_barrier
    # directly with an empty tracker.
    direct_br = sim.write_barrier(user_id="alice")
    _record_barrier("f_barrier_noop_on_empty_pending", direct_br)
    direct_is_noop = (
        direct_br.status == "noop"
        and direct_br.waited_ms == 0
        and direct_br.residual_pending == 0
    )

    passed = gate_short_circuited and direct_is_noop
    detail = (
        f"gate_short_circuited={gate_short_circuited} "
        f"(barrier_invoked={result.barrier_invoked}, "
        f"barrier_result={result.barrier_result}); "
        f"direct_barrier={_fmt_barrier(direct_br)}, "
        f"direct_is_noop={direct_is_noop}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_barrier(br: BarrierResult | None) -> str:
    if br is None:
        return "None (gate short-circuit)"
    return (
        f"BarrierResult(status={br.status!r}, "
        f"waited_ms={br.waited_ms}, residual_pending={br.residual_pending})"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all() -> bool:
    """Run every sub-case against the FixedKernelSimulator.

    EXPECTED OUTCOME on FIXED simulator: every sub-case PASSES — this
    confirms Property 1 (Bug Condition resolved) and the no-op
    preservation clauses 3.1 / 3.6 hold.
    """
    print("=" * 78)
    print("Kernel write-barrier fix verification (Property 1)")
    print("Tests run against the FixedKernelSimulator (post-fix kernel mirror).")
    print("EXPECTED OUTCOME: ALL sub-cases PASS — confirming the race is closed.")
    print("=" * 78)

    print("\n--- (a) Fast-inference race (200 ms latency, 50 ms delay) ---")
    passed_a, detail_a = test_a_fast_inference_race()
    record("a_fast_inference_race", passed_a, detail_a)

    print("\n--- (b) Slow-inference determinism (50 iterations) ---")
    passed_b, detail_b = test_b_slow_inference_determinism()
    record("b_slow_inference_determinism", passed_b, detail_b)

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
            "All 200 hypothesis examples passed (oracle matches simulator)",
        )
    except AssertionError as e:
        msg = str(e).splitlines()[0]
        record(
            "e_hypothesis_property",
            False,
            f"Hypothesis falsification: {msg}",
        )
    except Exception as e:  # pragma: no cover - hypothesis infra error
        record(
            "e_hypothesis_property",
            False,
            f"Hypothesis raised non-assertion error: {type(e).__name__}: {e}",
        )
        traceback.print_exc()

    print("\n--- (f) Barrier no-op on empty pending set ---")
    passed_f, detail_f = test_f_barrier_noop_on_empty_pending()
    record("f_barrier_noop_on_empty_pending", passed_f, detail_f)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — Fix verification")
    print("=" * 78)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results)}")

    print("\nBarrier outcomes per sub-case:")
    for name, br in barrier_timings:
        print(f"  - {name}: {_fmt_barrier(br)}")

    if failed:
        print("\nFAILURES (these would mean the fix is incomplete):")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")

    print("=" * 78)

    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    if all_passed:
        print(
            "\nVERIFICATION RESULT: Property 1 oracle matches FixedKernelSimulator "
            "across all generated cases."
        )
        sys.exit(0)
    else:
        print(
            "\nVERIFICATION RESULT: at least one sub-case failed — fix is incomplete."
        )
        sys.exit(1)
