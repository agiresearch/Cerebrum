"""Idempotency property test for kernel-memory-write-barrier (Task 4.3).

Property (Idempotency): calling ``writeBarrier(user_id)`` twice in
succession is observationally equivalent to calling it once.  After the
first invocation has drained (or short-circuited on) the pending writes
for ``user_id``, the second invocation:

  * MUST return ``BarrierResult(status="noop",
    residual_pending=0, waited_ms < poll_interval_ms)``
  * MUST NOT mutate ``PendingWritesTracker.pending_for(user_id)``
  * MUST NOT alter Mem0's visible state for ``user_id``
  * MUST NOT advance the simulator's virtual clock (no extra side effects)

This is a direct expression of the "no duplicate side effects, identical
retrieval input state" clause from
``.kiro/specs/kernel-memory-write-barrier/tasks.md`` task 4.3.

The test reuses the post-fix harness from
``test_kernel_write_barrier_fix_verification.py`` — specifically
``FixedKernelSimulator`` (which embeds ``PendingWritesTracker`` at
``sim.tracker``) and ``Mem0Fake`` from
``test_kernel_write_barrier_bug_condition.py`` — so the simulator
runtime is identical to the one used in tasks 4.1 and 4.2 and the
property is checked against the same code path.

Repository scope
----------------
The kernel itself lives in https://github.com/agiresearch/AIOS, NOT in
this repo. This file lives in Cerebrum so the post-fix idempotency
contract can be checked locally via the existing
``python tests/agents/...`` convention. The AIOS team should mirror
this test inside the kernel test suite at
``aios/tests/memory/test_write_barrier_idempotency.py`` (the embedded
simulator can be replaced with the real ``ContextInjector`` +
``Mem0Provider`` + ``PendingWritesTracker``).

**Validates: Requirements 2.2**
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# Allow running via `python tests/agents/test_kernel_write_barrier_idempotency.py`
# from the repository root.
sys.path.insert(0, ".")

# Make sibling test modules importable so we can reuse the FixedKernelSimulator
# without duplicating the post-fix kernel mirror.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from test_kernel_write_barrier_bug_condition import (  # noqa: E402
    CALLING_AGENT,
    OWNER_AGENTS,
    Mem0Fake,
)
from test_kernel_write_barrier_fix_verification import (  # noqa: E402
    BarrierResult,
    FixedKernelSimulator,
    PendingWritesTracker,  # noqa: F401  (imported per task instructions)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_visible(memories: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Project a Mem0Fake.get_all(...) result into a hashable, sorted form.

    Two ``get_all(user_id=...)`` results that contain the same memories
    in the same logical state will compare equal under this projection
    regardless of internal list ordering.
    """
    return sorted(
        (
            m["id"],
            m.get("user_id"),
            m.get("owner_agent"),
            m.get("sharing_policy"),
            m.get("content"),
        )
        for m in memories
    )


def _fmt(br: BarrierResult | None) -> str:
    if br is None:
        return "None (gate short-circuit)"
    return (
        f"BarrierResult(status={br.status!r}, "
        f"waited_ms={br.waited_ms}, residual_pending={br.residual_pending})"
    )


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
# Sub-case (a) — Idempotency with prior writes that drained successfully
# ---------------------------------------------------------------------------


def test_a_idempotency_after_drained_writes() -> tuple[bool, str]:
    """(a) Two cross-agent shared writes drain via the first barrier;
    the second barrier must be a no-op.

    Concrete scenario mirrors the fix-verification sub-case (a):
    200 ms indexing latency, 50 ms inter-syscall delay, two shared
    writes for ``alice`` from ``profile_agent`` and ``task_agent``.

    EXPECTED: first barrier status ∈ {ok, noop}; second barrier
    status="noop" with waited_ms < poll_interval_ms; tracker empty
    before and after; Mem0 visible state unchanged across the second
    call.

    Validates: Requirement 2.2
    """
    mem0 = Mem0Fake(indexing_latency_ms=200.0)
    sim = FixedKernelSimulator(mem0)

    user_id = "alice"
    sim.add(
        user_id=user_id,
        owner_agent="profile_agent",
        sharing_policy="shared",
        content="alice prefers dark mode",
    )
    sim.add(
        user_id=user_id,
        owner_agent="task_agent",
        sharing_policy="shared",
        content="alice is debugging the auth flow",
    )

    # First barrier inside the full pipeline (llm_chat -> gate -> writeBarrier).
    pipeline = sim.llm_chat(
        user_id=user_id,
        calling_agent=CALLING_AGENT,
        inter_syscall_delay_ms=50.0,
    )

    first_invoked = pipeline.barrier_invoked
    first_br = pipeline.barrier_result
    first_status_ok = (
        first_br is not None and first_br.status in {"ok", "noop"}
    )

    # Snapshot tracker + Mem0 + clock BEFORE the second call.
    pending_before = sim.tracker.pending_for(user_id)
    visible_before = _normalize_visible(mem0.get_all(user_id=user_id))
    clock_before = mem0.clock_ms

    # Second barrier — direct invocation.
    second_br = sim.write_barrier(user_id=user_id)

    # Snapshot AFTER the second call.
    pending_after = sim.tracker.pending_for(user_id)
    visible_after = _normalize_visible(mem0.get_all(user_id=user_id))

    checks = [
        ("first barrier was invoked via gate", first_invoked),
        ("first barrier status ∈ {ok, noop}", first_status_ok),
        ("pending_for empty before second", len(pending_before) == 0),
        ("second status == 'noop'", second_br.status == "noop"),
        ("second residual_pending == 0", second_br.residual_pending == 0),
        (
            f"second waited_ms ({second_br.waited_ms}) < "
            f"poll_interval_ms ({sim.barrier_poll_interval_ms})",
            second_br.waited_ms < sim.barrier_poll_interval_ms,
        ),
        ("pending_for empty after second", len(pending_after) == 0),
        ("Mem0 visible state unchanged", visible_before == visible_after),
        ("clock unchanged across no-op", mem0.clock_ms == clock_before),
    ]
    failures = [name for name, ok in checks if not ok]
    passed = not failures
    detail = (
        f"first={_fmt(first_br)}, second={_fmt(second_br)}, "
        f"pending_before={len(pending_before)}, "
        f"pending_after={len(pending_after)}, "
        f"visible_count={len(visible_after)}, "
        f"failures={failures}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (b) — Idempotency immediately after a "noop" barrier
# ---------------------------------------------------------------------------


def test_b_idempotency_after_noop_barrier() -> tuple[bool, str]:
    """(b) No writes ever submitted; both writeBarrier calls are observable
    no-ops with identical state.

    Scenario: ``llm_chat`` is called for a user with an empty pending
    set.  The gate predicate short-circuits (preservation 3.1) so no
    barrier is invoked through the pipeline.  We then call
    ``write_barrier(user_id=...)`` directly twice and assert the two
    invocations are observationally identical.

    EXPECTED: both direct invocations return
    ``BarrierResult(status="noop", waited_ms=0, residual_pending=0)``;
    tracker stays empty; Mem0 visible state stays empty; clock does
    not advance.

    Validates: Requirement 2.2 (preservation 3.1, 3.6)
    """
    mem0 = Mem0Fake(indexing_latency_ms=200.0)
    sim = FixedKernelSimulator(mem0)

    user_id = "alice"

    # Pipeline path: gate short-circuits because no pending writes for alice.
    pipeline = sim.llm_chat(
        user_id=user_id,
        calling_agent=CALLING_AGENT,
        inter_syscall_delay_ms=0.0,
    )
    gate_short_circuited = (
        not pipeline.barrier_invoked and pipeline.barrier_result is None
    )

    # First direct call — no-op short-circuit at the top of the loop.
    first_direct = sim.write_barrier(user_id=user_id)

    pending_before = sim.tracker.pending_for(user_id)
    visible_before = _normalize_visible(mem0.get_all(user_id=user_id))
    clock_before = mem0.clock_ms

    # Second direct call — must be observationally identical to the first.
    second_direct = sim.write_barrier(user_id=user_id)

    pending_after = sim.tracker.pending_for(user_id)
    visible_after = _normalize_visible(mem0.get_all(user_id=user_id))

    checks = [
        ("gate short-circuited (no first barrier via pipeline)",
         gate_short_circuited),
        ("first direct status == 'noop'", first_direct.status == "noop"),
        ("first direct waited_ms == 0", first_direct.waited_ms == 0),
        ("first direct residual_pending == 0",
         first_direct.residual_pending == 0),
        ("second direct status == 'noop'", second_direct.status == "noop"),
        ("second direct waited_ms == 0", second_direct.waited_ms == 0),
        ("second direct residual_pending == 0",
         second_direct.residual_pending == 0),
        (
            f"second waited_ms < poll_interval_ms "
            f"({sim.barrier_poll_interval_ms})",
            second_direct.waited_ms < sim.barrier_poll_interval_ms,
        ),
        ("pending_for empty before second", len(pending_before) == 0),
        ("pending_for empty after second", len(pending_after) == 0),
        ("Mem0 visible unchanged across second",
         visible_before == visible_after),
        ("clock unchanged across second", mem0.clock_ms == clock_before),
        ("first and second direct results equal",
         first_direct == second_direct),
    ]
    failures = [name for name, ok in checks if not ok]
    passed = not failures
    detail = (
        f"first_direct={_fmt(first_direct)}, "
        f"second_direct={_fmt(second_direct)}, "
        f"failures={failures}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Sub-case (c) — Hypothesis PBT: idempotency over random write sequences
# ---------------------------------------------------------------------------

# Strategy notes
# --------------
# * ``target_user_id`` is the user_id passed to ``llm_chat`` and the second
#   ``write_barrier`` call.
# * ``extras`` are 0–4 random writes with arbitrary user_ids; they may or
#   may not target the same user_id.  They mostly exist to stress the
#   tracker against unrelated state.
# * ``mandatory`` guarantees at least one cross-agent shared write exists
#   for ``target_user_id``, which guarantees the gate predicate triggers
#   so the *first* barrier is genuinely invoked through the pipeline
#   (otherwise this test would degenerate into the (b) sub-case).
# * ``indexing_latency_ms`` is bounded at 1500 ms so the worst-case
#   bounded-wait barrier (latency + small delay) stays comfortably below
#   the default 2000 ms timeout — the property is about idempotency,
#   not about bounded wait, so we keep first-barrier ``status="timeout"``
#   off the table.
# * ``inter_syscall_delay_ms`` covers both the "delay > latency" (writes
#   already visible) and "delay < latency" (barrier must poll) regimes.

_user_id_strategy = st.sampled_from(["alice", "bob", "carol", "dave"])
_owner_strategy = st.sampled_from(OWNER_AGENTS)
_policy_strategy = st.sampled_from(["shared", "private"])

_extra_write_strategy = st.fixed_dictionaries(
    {
        "user_id": _user_id_strategy,
        "owner_agent": _owner_strategy,
        "sharing_policy": _policy_strategy,
        "content": st.text(min_size=1, max_size=40),
    }
)

_mandatory_write_strategy = st.fixed_dictionaries(
    {
        "owner_agent": _owner_strategy,
        # mandatory write must be "shared" so the gate predicate triggers
        # — owner_agent != assistant_agent already holds because OWNER_AGENTS
        # excludes the calling agent.
        "sharing_policy": st.just("shared"),
        "content": st.text(min_size=1, max_size=40),
    }
)

_latency_strategy = st.floats(min_value=0.0, max_value=1500.0)
_delay_strategy = st.floats(min_value=0.0, max_value=100.0)


@given(
    extras=st.lists(_extra_write_strategy, min_size=0, max_size=4),
    mandatory=_mandatory_write_strategy,
    target_user_id=_user_id_strategy,
    indexing_latency_ms=_latency_strategy,
    inter_syscall_delay_ms=_delay_strategy,
)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
def test_c_hypothesis_idempotency(
    extras: list[dict[str, Any]],
    mandatory: dict[str, Any],
    target_user_id: str,
    indexing_latency_ms: float,
    inter_syscall_delay_ms: float,
) -> None:
    """(c) PBT — ``writeBarrier`` is idempotent across random inputs.

    For ANY write sequence containing at least one cross-agent shared
    write for ``target_user_id``, with indexing latencies under the
    barrier timeout, the following hold after running the full fixed
    pipeline once and then calling ``write_barrier(user_id=target)``
    a second time directly:

      * First barrier (inside ``llm_chat``) returned
        ``status ∈ {"ok", "noop"}`` (never ``"timeout"``).
      * Second barrier returns ``status="noop"`` with
        ``residual_pending=0`` and ``waited_ms < poll_interval_ms``.
      * ``tracker.pending_for(target_user_id)`` is empty before AND
        after the second call.
      * ``mem0.get_all(user_id=target_user_id)`` returns the same
        visible-memory set before AND after the second call.
      * The Mem0 fake's virtual clock does not advance during the
        second call (no duplicate side effects).

    Validates: Requirement 2.2
    """
    mem0 = Mem0Fake(indexing_latency_ms=indexing_latency_ms)
    sim = FixedKernelSimulator(mem0)

    # Submit all generated writes through the simulator's add(...) wrapper
    # so the tracker sees every record alongside the Mem0Fake enqueue.
    all_writes = list(extras) + [
        {
            "user_id": target_user_id,
            "owner_agent": mandatory["owner_agent"],
            "sharing_policy": mandatory["sharing_policy"],
            "content": mandatory["content"],
        }
    ]
    for w in all_writes:
        sim.add(
            user_id=w["user_id"],
            owner_agent=w["owner_agent"],
            sharing_policy=w["sharing_policy"],
            content=w["content"],
        )

    # First barrier via the full pipeline.
    pipeline = sim.llm_chat(
        user_id=target_user_id,
        calling_agent=CALLING_AGENT,
        inter_syscall_delay_ms=inter_syscall_delay_ms,
    )

    # The mandatory cross-agent shared write GUARANTEES the gate triggers.
    assert pipeline.barrier_invoked, (
        f"gate predicate should trigger when a shared cross-agent write "
        f"is pending for target_user_id={target_user_id!r}; "
        f"writes={all_writes}"
    )
    first_br = pipeline.barrier_result
    assert first_br is not None
    assert first_br.status in {"ok", "noop"}, (
        f"first barrier status must not be 'timeout' "
        f"(latency={indexing_latency_ms:.1f}ms, "
        f"timeout_ms={sim.barrier_timeout_ms}ms): {first_br}"
    )

    # Snapshot before second call.
    pending_before = sim.tracker.pending_for(target_user_id)
    visible_before = _normalize_visible(mem0.get_all(user_id=target_user_id))
    clock_before = mem0.clock_ms
    assert len(pending_before) == 0, (
        f"tracker should be drained for target_user_id after "
        f"status={first_br.status}: {pending_before}"
    )

    # Second barrier — direct call.
    second_br = sim.write_barrier(user_id=target_user_id)

    # Snapshot after.
    pending_after = sim.tracker.pending_for(target_user_id)
    visible_after = _normalize_visible(mem0.get_all(user_id=target_user_id))

    # Idempotency: the second call must be a fully observable no-op.
    assert second_br.status == "noop", (
        f"second barrier must return status='noop'; got {second_br}"
    )
    assert second_br.residual_pending == 0, (
        f"second barrier residual_pending must be 0; got {second_br}"
    )
    assert second_br.waited_ms < sim.barrier_poll_interval_ms, (
        f"second barrier waited_ms ({second_br.waited_ms}) must be < "
        f"poll_interval_ms ({sim.barrier_poll_interval_ms})"
    )
    assert len(pending_after) == 0, (
        f"tracker pending_for(target) must remain empty after second call; "
        f"got {pending_after}"
    )
    assert visible_before == visible_after, (
        f"Mem0 visible state must be unchanged across second call; "
        f"before={visible_before}, after={visible_after}"
    )
    assert mem0.clock_ms == clock_before, (
        f"clock must not advance during a no-op barrier; "
        f"before={clock_before}, after={mem0.clock_ms}"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all() -> bool:
    """Run every sub-case.  Each PASS confirms ``writeBarrier`` is idempotent.

    EXPECTED OUTCOME on FIXED simulator: ALL sub-cases (a), (b), (c) PASS.
    """
    print("=" * 78)
    print("Kernel write-barrier idempotency property test (Task 4.3)")
    print("Validates Requirement 2.2: writeBarrier(...) called twice is")
    print("observationally equivalent to calling it once.")
    print("=" * 78)

    print("\n--- (a) Idempotency with prior writes that drained successfully ---")
    passed_a, detail_a = test_a_idempotency_after_drained_writes()
    record("a_idempotency_after_drained_writes", passed_a, detail_a)

    print("\n--- (b) Idempotency immediately after a 'noop' barrier ---")
    passed_b, detail_b = test_b_idempotency_after_noop_barrier()
    record("b_idempotency_after_noop_barrier", passed_b, detail_b)

    print("\n--- (c) Hypothesis PBT (200 examples) ---")
    try:
        test_c_hypothesis_idempotency()
        record(
            "c_hypothesis_idempotency",
            True,
            "All 200 hypothesis examples passed; second writeBarrier "
            "always observably a no-op",
        )
    except AssertionError as e:
        msg = str(e).splitlines()[0]
        record(
            "c_hypothesis_idempotency",
            False,
            f"Hypothesis falsification: {msg}",
        )
    except Exception as e:  # pragma: no cover — hypothesis infra error
        record(
            "c_hypothesis_idempotency",
            False,
            f"Hypothesis raised non-assertion error: {type(e).__name__}: {e}",
        )
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY — Idempotency")
    print("=" * 78)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results)}")

    if failed:
        print("\nFAILURES (idempotency violations):")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")

    print("=" * 78)
    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    if all_passed:
        print(
            "\nIDEMPOTENCY RESULT: writeBarrier is idempotent across all "
            "generated cases."
        )
        sys.exit(0)
    else:
        print(
            "\nIDEMPOTENCY RESULT: at least one sub-case failed — fix is "
            "incomplete."
        )
        sys.exit(1)
