"""Preservation observation tests for the kernel write-barrier fix (Property 2).

Property 2: Preservation — Non-Buggy Inputs Unchanged

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

CRITICAL: These tests MUST PASS on the UNFIXED kernel — they capture
baseline behavior that the write-barrier fix must preserve. If any test
fails on unfixed code, the test logic is wrong (not the kernel).

GOAL: Capture the eight preservation clauses (3.1–3.8) as property-based
tests that pass on the UNFIXED kernel so the fix can be validated against
them later.

Methodology:
  1. Observe behavior on UNFIXED kernel first
  2. Write property tests asserting that behavior across the input domain
  3. Confirm all tests PASS on unfixed code
"""

import sys
sys.path.insert(0, ".")

import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Reuse Mem0Fake and UnfixedContextInjector from Task 1
# (Duplicated here to keep the test self-contained per Cerebrum conventions)
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
    relevance_score: float = 1.0


class Mem0Fake:
    """Fake Mem0 provider with configurable indexing latency.

    Simulates real Mem0 behavior where add() enqueues a write but the
    write is not immediately visible to get_all()/search() until an
    internal indexing pipeline completes.
    """

    def __init__(self, indexing_latency_ms: float = 200.0):
        self.indexing_latency_ms = indexing_latency_ms
        self._store: List[MemoryEntry] = []
        self._lock = threading.Lock()
        self._write_counter = 0

    def add(self, content: str, user_id: str, owner_agent: str,
            sharing_policy: str, memory_type: str = "profile") -> str:
        """Enqueue a memory write. Returns a write_id.

        The entry becomes visible after indexing_latency_ms.
        """
        self._write_counter += 1
        write_id = f"write_{self._write_counter}"

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

        return write_id

    def add_immediate(self, content: str, user_id: str, owner_agent: str,
                      sharing_policy: str, memory_type: str = "profile",
                      relevance_score: float = 1.0) -> str:
        """Add a memory that is immediately indexed (no latency).

        Used for testing preservation scenarios where writes are already
        indexed before retrieval (3.6, 3.7).
        """
        self._write_counter += 1
        write_id = f"write_{self._write_counter}"

        entry = MemoryEntry(
            content=content,
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
            memory_type=memory_type,
            indexed=True,
            indexed_at=time.time(),
            relevance_score=relevance_score,
        )
        with self._lock:
            self._store.append(entry)

        return write_id

    def get_all(self, user_id: str) -> List[MemoryEntry]:
        """Return only indexed (visible) entries for the given user_id."""
        with self._lock:
            return [
                e for e in self._store
                if e.user_id == user_id and e.indexed
            ]

    def get_all_shared_cross_agent(self, user_id: str,
                                    calling_agent: str) -> List[MemoryEntry]:
        """Return indexed shared entries from other agents for the given user_id."""
        with self._lock:
            return [
                e for e in self._store
                if e.user_id == user_id
                and e.indexed
                and e.sharing_policy == "shared"
                and e.owner_agent != calling_agent
            ]

    def get_own_memories(self, user_id: str,
                         owner_agent: str) -> List[MemoryEntry]:
        """Return indexed entries owned by the specified agent for the user_id."""
        with self._lock:
            return [
                e for e in self._store
                if e.user_id == user_id
                and e.indexed
                and e.owner_agent == owner_agent
            ]

    def search(self, user_id: str, query: str,
               calling_agent: str) -> List[MemoryEntry]:
        """Return indexed entries matching user_id (simplified search).

        In the real Mem0, this would do embedding similarity. Here we
        just return all indexed entries for the user_id from the calling agent.
        """
        with self._lock:
            return [
                e for e in self._store
                if e.user_id == user_id
                and e.indexed
                and e.owner_agent == calling_agent
            ]

    def reset(self):
        """Clear all entries."""
        with self._lock:
            self._store.clear()
            self._write_counter = 0


# ---------------------------------------------------------------------------
# Unfixed ContextInjector with Full Pipeline Simulation
# ---------------------------------------------------------------------------

@dataclass
class KernelConfig:
    """Kernel configuration for the ContextInjector."""
    auto_inject: bool = True
    auto_extract: bool = True
    relevance_threshold: float = 0.3
    max_injected_memories: int = 10
    max_memory_tokens: int = 2000
    barrier_poll_interval_ms: int = 25


class UnfixedContextInjector:
    """Simulates the UNFIXED kernel's ContextInjector behavior.

    The unfixed kernel does NOT wait for writes to be indexed before
    executing cross-agent retrieval. It immediately calls get_all() /
    search(), which returns only entries that Mem0 has already indexed.

    For preservation tests, this is fine — we are testing paths where
    the bug condition does NOT hold.
    """

    def __init__(self, mem0: Mem0Fake, config: KernelConfig | None = None):
        self.mem0 = mem0
        self.config = config or KernelConfig()
        self.barrier_invoked = False

    def inject(self, user_id: str, calling_agent: str,
               cross_agent: bool = True) -> List[MemoryEntry]:
        """Perform retrieval WITHOUT a write barrier.

        Args:
            user_id: The resolved user_id for retrieval.
            calling_agent: The agent that triggered llm_chat.
            cross_agent: Whether to include cross-agent memories.

        Returns:
            List of injected memories after filtering/ranking.
        """
        self.barrier_invoked = False

        if not self.config.auto_inject:
            return []

        if not cross_agent:
            # Single-agent retrieval — only own memories
            return self.mem0.get_own_memories(user_id, calling_agent)

        # Cross-agent retrieval (no write barrier on unfixed kernel)
        results = self.mem0.get_all_shared_cross_agent(user_id, calling_agent)

        # Apply ranking/threshold/token-cap pipeline
        results = self._apply_relevance_threshold(results)
        results = self._apply_max_memories(results)
        results = self._apply_token_cap(results)

        return results

    def _apply_relevance_threshold(self,
                                    memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Filter memories below relevance_threshold."""
        return [m for m in memories
                if m.relevance_score >= self.config.relevance_threshold]

    def _apply_max_memories(self,
                            memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Limit to max_injected_memories, sorted by relevance descending."""
        sorted_mems = sorted(memories, key=lambda m: m.relevance_score, reverse=True)
        return sorted_mems[:self.config.max_injected_memories]

    def _apply_token_cap(self,
                         memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Truncate memories to fit within max_memory_tokens.

        Simplified: estimate 1 token per 4 characters.
        """
        result = []
        total_tokens = 0
        for m in memories:
            tokens = len(m.content) // 4 + 1
            if total_tokens + tokens > self.config.max_memory_tokens:
                break
            result.append(m)
            total_tokens += tokens
        return result


# ---------------------------------------------------------------------------
# Conversation Extractor Simulation (for 3.8)
# ---------------------------------------------------------------------------

class ConversationExtractor:
    """Simulates the kernel's auto_extract behavior.

    Stores conversation memories under the resolved user_id when
    auto_extract is enabled.
    """

    def __init__(self, mem0: Mem0Fake, config: KernelConfig | None = None):
        self.mem0 = mem0
        self.config = config or KernelConfig()

    def auto_extract(self, user_id: str, assistant_agent: str,
                     conversation_content: str) -> str | None:
        """Extract and store a conversation memory.

        Returns the write_id if stored, None if auto_extract is disabled.
        """
        if not self.config.auto_extract:
            return None

        write_id = self.mem0.add(
            content=conversation_content,
            user_id=user_id,
            owner_agent=assistant_agent,
            sharing_policy="private",
            memory_type="conversation",
        )
        return write_id


# ---------------------------------------------------------------------------
# Test Results Tracking
# ---------------------------------------------------------------------------

results_log: List[Tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    results_log.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

_user_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
)
_agent_names = st.sampled_from([
    "profile_agent", "task_agent", "assistant_agent", "notes_agent", "custom_agent"
])
_sharing_policies = st.sampled_from(["shared", "private"])
_memory_types = st.sampled_from(["profile", "task_context", "conversation"])
_content = st.text(min_size=1, max_size=200)
_relevance_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# ---------------------------------------------------------------------------
# (a) 3.1 — No-pending-writes preservation
# ---------------------------------------------------------------------------

@given(user_id=_user_ids)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_a_no_pending_writes_preservation(user_id):
    """(a) 3.1 — No-pending-writes preservation.

    **Validates: Requirements 3.1**

    With an empty pending set for the user_id, cross-agent retrieval
    executes through its existing path without barrier-induced latency.
    The injected set is whatever get_all returns (empty, since no writes
    have been submitted).
    """
    mem0 = Mem0Fake(indexing_latency_ms=200.0)
    config = KernelConfig()
    injector = UnfixedContextInjector(mem0, config)

    # No writes submitted — empty pending set
    start_ms = time.time() * 1000

    injected = injector.inject(user_id=user_id, calling_agent="assistant_agent")

    elapsed_ms = (time.time() * 1000) - start_ms

    # Assert: no barrier invoked
    assert not injector.barrier_invoked, "Barrier should not be invoked with no pending writes"

    # Assert: latency is negligible (well below barrier_poll_interval_ms)
    assert elapsed_ms < config.barrier_poll_interval_ms, (
        f"Retrieval took {elapsed_ms:.1f} ms, expected < {config.barrier_poll_interval_ms} ms "
        f"(no barrier path active)"
    )

    # Assert: injected set matches get_all (should be empty since no writes)
    expected = mem0.get_all_shared_cross_agent(user_id, "assistant_agent")
    assert injected == expected, (
        f"Injected set should equal get_all result. "
        f"Got {len(injected)}, expected {len(expected)}"
    )


# ---------------------------------------------------------------------------
# (b) 3.2 — Standalone-write preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    owner_agent=_agent_names,
    sharing_policy=_sharing_policies,
    content=_content,
    memory_type=_memory_types,
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_b_standalone_write_preservation(user_id, owner_agent, sharing_policy,
                                          content, memory_type):
    """(b) 3.2 — Standalone-write preservation.

    **Validates: Requirements 3.2**

    Calling create_memory outside any retrieval context produces the
    same write side effects as the unfixed kernel: a write_id is returned,
    content and metadata are stored correctly.
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)  # Instant indexing for standalone write test

    # Call create_memory (standalone, no retrieval context)
    write_id = mem0.add(
        content=content,
        user_id=user_id,
        owner_agent=owner_agent,
        sharing_policy=sharing_policy,
        memory_type=memory_type,
    )

    # Wait for indexing to complete (instant in this case)
    time.sleep(0.01)

    # Assert: write_id is returned
    assert write_id is not None and write_id != "", (
        f"Expected a non-empty write_id, got {write_id!r}"
    )

    # Assert: stored content and metadata are correct
    all_entries = mem0.get_all(user_id)
    matching = [e for e in all_entries if e.content == content
                and e.owner_agent == owner_agent
                and e.sharing_policy == sharing_policy
                and e.memory_type == memory_type]

    assert len(matching) == 1, (
        f"Expected exactly 1 matching entry stored, found {len(matching)}. "
        f"Total entries for user_id={user_id!r}: {len(all_entries)}"
    )

    stored = matching[0]
    assert stored.content == content, f"Content mismatch: {stored.content!r} != {content!r}"
    assert stored.user_id == user_id, f"user_id mismatch: {stored.user_id!r} != {user_id!r}"
    assert stored.owner_agent == owner_agent
    assert stored.sharing_policy == sharing_policy
    assert stored.memory_type == memory_type


# ---------------------------------------------------------------------------
# (c) 3.3 — Single-agent retrieval preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    agent_name=_agent_names,
    contents=st.lists(_content, min_size=0, max_size=5),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_c_single_agent_retrieval_preservation(user_id, agent_name, contents):
    """(c) 3.3 — Single-agent retrieval preservation.

    **Validates: Requirements 3.3**

    Retrievals scoped only to the calling agent's own memories use the
    existing single-agent retrieval path without invoking the barrier.
    Result is identical to direct get_own_memories call.
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)
    config = KernelConfig()
    injector = UnfixedContextInjector(mem0, config)

    # Write memories owned by the agent (immediately indexed)
    for c in contents:
        mem0.add_immediate(
            content=c,
            user_id=user_id,
            owner_agent=agent_name,
            sharing_policy="private",
            memory_type="profile",
        )

    # Single-agent retrieval (cross_agent=False)
    injected = injector.inject(
        user_id=user_id,
        calling_agent=agent_name,
        cross_agent=False,
    )

    # Assert: barrier not invoked
    assert not injector.barrier_invoked, "Barrier should not be invoked for single-agent retrieval"

    # Assert: result matches direct own-memories query
    expected = mem0.get_own_memories(user_id, agent_name)
    assert len(injected) == len(expected), (
        f"Single-agent retrieval returned {len(injected)} entries, "
        f"expected {len(expected)} from get_own_memories"
    )

    # Assert: content sets match
    injected_contents = {e.content for e in injected}
    expected_contents = {e.content for e in expected}
    assert injected_contents == expected_contents, (
        f"Content mismatch between injected and expected"
    )


# ---------------------------------------------------------------------------
# (d) 3.4 — auto_inject=false preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    owner_agent=_agent_names,
    contents=st.lists(_content, min_size=1, max_size=5),
    sharing_policy=_sharing_policies,
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_d_auto_inject_false_preservation(user_id, owner_agent, contents,
                                           sharing_policy):
    """(d) 3.4 — auto_inject=false preservation.

    **Validates: Requirements 3.4**

    With auto_inject=false in kernel config, the ContextInjector is
    bypassed entirely. inject() is never effectively called, barrier
    never reached.
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)
    config = KernelConfig(auto_inject=False)
    injector = UnfixedContextInjector(mem0, config)

    # Write some memories (both shared and private)
    for c in contents:
        mem0.add_immediate(
            content=c,
            user_id=user_id,
            owner_agent=owner_agent,
            sharing_policy=sharing_policy,
        )

    # Trigger retrieval with auto_inject=false
    injected = injector.inject(
        user_id=user_id,
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # Assert: injector returns empty (bypassed)
    assert injected == [], (
        f"With auto_inject=false, inject() should return empty list, "
        f"got {len(injected)} entries"
    )

    # Assert: barrier not invoked
    assert not injector.barrier_invoked, "Barrier should not be invoked when auto_inject=false"


# ---------------------------------------------------------------------------
# (e) 3.5 — Sharing-policy preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    owner_agents=st.lists(_agent_names, min_size=1, max_size=5),
    contents=st.lists(_content, min_size=1, max_size=5),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_e_sharing_policy_preservation(user_id, owner_agents, contents):
    """(e) 3.5 — Sharing-policy preservation.

    **Validates: Requirements 3.5**

    Write sequences containing ONLY sharing_policy="private" memories.
    Cross-agent retrieval observes empty injection because private memories
    are excluded from cross-agent injection by the sharing-policy filter.
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)
    config = KernelConfig()
    injector = UnfixedContextInjector(mem0, config)

    # Write only private memories (immediately indexed)
    for i, c in enumerate(contents):
        agent = owner_agents[i % len(owner_agents)]
        mem0.add_immediate(
            content=c,
            user_id=user_id,
            owner_agent=agent,
            sharing_policy="private",  # All private
            memory_type="profile",
        )

    # Cross-agent retrieval
    injected = injector.inject(
        user_id=user_id,
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # Assert: empty injection (no shared pending writes, all private)
    assert injected == [], (
        f"With only private memories, cross-agent injection should be empty, "
        f"got {len(injected)} entries"
    )

    # Assert: barrier not invoked (no shared pending writes to wait for)
    assert not injector.barrier_invoked, (
        "Barrier should not be invoked when all pending writes are private"
    )


# ---------------------------------------------------------------------------
# (f) 3.6 — Already-indexed-writes preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    owner_agents=st.lists(
        st.sampled_from(["profile_agent", "task_agent", "notes_agent"]),
        min_size=1,
        max_size=5,
    ),
    contents=st.lists(_content, min_size=1, max_size=5),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_f_already_indexed_writes_preservation(user_id, owner_agents, contents):
    """(f) 3.6 — Already-indexed-writes preservation.

    **Validates: Requirements 3.6**

    Writes are already committed and indexed before retrieval. The
    injected set on the unfixed kernel is identical to what it would
    be post-fix (the barrier would be a no-op on this happy path).
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)
    config = KernelConfig()
    injector = UnfixedContextInjector(mem0, config)

    # Write shared memories that are IMMEDIATELY indexed (no race)
    written_contents = set()
    for i, c in enumerate(contents):
        agent = owner_agents[i % len(owner_agents)]
        # Only add as shared from non-assistant agents
        if agent != "assistant_agent":
            mem0.add_immediate(
                content=c,
                user_id=user_id,
                owner_agent=agent,
                sharing_policy="shared",
                memory_type="profile",
            )
            written_contents.add(c)

    # Cross-agent retrieval (writes already indexed — happy path)
    injected = injector.inject(
        user_id=user_id,
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # Assert: injected set matches direct cross-agent query
    expected = mem0.get_all_shared_cross_agent(user_id, "assistant_agent")
    assert len(injected) == len(expected), (
        f"Already-indexed retrieval: got {len(injected)}, expected {len(expected)}"
    )

    # Assert: content sets match
    injected_contents = {e.content for e in injected}
    expected_contents = {e.content for e in expected}
    assert injected_contents == expected_contents, (
        f"Content mismatch on already-indexed path"
    )


# ---------------------------------------------------------------------------
# (g) 3.7 — Ranking/threshold/token-cap preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    relevance_scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=15,
    ),
    relevance_threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    max_injected_memories=st.integers(min_value=1, max_value=20),
    max_memory_tokens=st.integers(min_value=10, max_value=5000),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_g_ranking_threshold_token_cap_preservation(
    user_id, relevance_scores, relevance_threshold,
    max_injected_memories, max_memory_tokens,
):
    """(g) 3.7 — Ranking/threshold/token-cap preservation.

    **Validates: Requirements 3.7**

    Pre-load Mem0 with mixed-relevance memories (already indexed, no race).
    Run retrieval through ranking → relevance_threshold →
    max_injected_memories → max_memory_tokens truncation.
    Assert the ranked/thresholded/truncated injected list matches the
    expected pipeline output.
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)
    config = KernelConfig(
        relevance_threshold=relevance_threshold,
        max_injected_memories=max_injected_memories,
        max_memory_tokens=max_memory_tokens,
    )
    injector = UnfixedContextInjector(mem0, config)

    # Pre-load memories with various relevance scores (already indexed)
    agents = ["profile_agent", "task_agent", "notes_agent"]
    for i, score in enumerate(relevance_scores):
        agent = agents[i % len(agents)]
        content = f"memory_content_{i}_" + "x" * 20  # Fixed-size content
        mem0.add_immediate(
            content=content,
            user_id=user_id,
            owner_agent=agent,
            sharing_policy="shared",
            memory_type="profile",
            relevance_score=score,
        )

    # Run retrieval through the full pipeline
    injected = injector.inject(
        user_id=user_id,
        calling_agent="assistant_agent",
        cross_agent=True,
    )

    # Manually compute expected result through the same pipeline
    all_shared = mem0.get_all_shared_cross_agent(user_id, "assistant_agent")

    # Step 1: relevance threshold
    above_threshold = [m for m in all_shared
                       if m.relevance_score >= relevance_threshold]

    # Step 2: sort by relevance descending, limit to max
    sorted_mems = sorted(above_threshold, key=lambda m: m.relevance_score, reverse=True)
    limited = sorted_mems[:max_injected_memories]

    # Step 3: token cap
    expected = []
    total_tokens = 0
    for m in limited:
        tokens = len(m.content) // 4 + 1
        if total_tokens + tokens > max_memory_tokens:
            break
        expected.append(m)
        total_tokens += tokens

    # Assert: injected list matches expected pipeline output
    assert len(injected) == len(expected), (
        f"Ranking/threshold/token-cap pipeline: got {len(injected)}, "
        f"expected {len(expected)} (threshold={relevance_threshold:.2f}, "
        f"max_memories={max_injected_memories}, max_tokens={max_memory_tokens})"
    )

    # Assert: order and content match
    for i, (got, exp) in enumerate(zip(injected, expected)):
        assert got.content == exp.content, (
            f"Mismatch at position {i}: got {got.content!r}, expected {exp.content!r}"
        )


# ---------------------------------------------------------------------------
# (h) 3.8 — Conversation-extractor preservation
# ---------------------------------------------------------------------------

@given(
    user_id=_user_ids,
    conversation_contents=st.lists(_content, min_size=1, max_size=5),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_h_conversation_extractor_preservation(user_id, conversation_contents):
    """(h) 3.8 — Conversation-extractor preservation.

    **Validates: Requirements 3.8**

    Auto-extracted conversation memories are stored under the resolved
    user_id per the existing conversation-extractor contract, independent
    of the barrier. The barrier has no interaction with the extract path.
    """
    mem0 = Mem0Fake(indexing_latency_ms=0.0)
    config = KernelConfig(auto_extract=True)
    extractor = ConversationExtractor(mem0, config)

    # Run auto_extract for each conversation
    write_ids = []
    for content in conversation_contents:
        write_id = extractor.auto_extract(
            user_id=user_id,
            assistant_agent="assistant_agent",
            conversation_content=content,
        )
        write_ids.append(write_id)

    # Wait for indexing
    time.sleep(0.01)

    # Assert: all writes returned valid write_ids
    for wid in write_ids:
        assert wid is not None and wid != "", (
            f"Expected non-empty write_id from auto_extract, got {wid!r}"
        )

    # Assert: stored conversation memories have correct user_id and content
    all_entries = mem0.get_all(user_id)
    conversation_entries = [e for e in all_entries if e.memory_type == "conversation"]

    assert len(conversation_entries) == len(conversation_contents), (
        f"Expected {len(conversation_contents)} conversation memories stored, "
        f"found {len(conversation_entries)}"
    )

    # Assert: content matches what was extracted
    stored_contents = {e.content for e in conversation_entries}
    expected_contents = set(conversation_contents)
    assert stored_contents == expected_contents, (
        f"Content mismatch: stored={stored_contents}, expected={expected_contents}"
    )

    # Assert: all stored under correct user_id and owner
    for entry in conversation_entries:
        assert entry.user_id == user_id, (
            f"user_id mismatch: {entry.user_id!r} != {user_id!r}"
        )
        assert entry.owner_agent == "assistant_agent", (
            f"owner_agent mismatch: {entry.owner_agent!r} != 'assistant_agent'"
        )
        assert entry.sharing_policy == "private", (
            f"Conversation memories should be private, got {entry.sharing_policy!r}"
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all():
    """Run all preservation observation tests and report results.

    These tests MUST PASS on the UNFIXED kernel — they capture baseline
    behavior that the write-barrier fix must preserve.
    """
    print("=" * 70)
    print("Preservation Observation Tests (Property 2)")
    print("Non-Buggy Inputs Unchanged")
    print("")
    print("EXPECTED: These tests PASS on the UNFIXED kernel.")
    print("They capture baseline behavior for post-fix validation.")
    print("=" * 70)

    test_funcs = [
        ("a_no_pending_writes_preservation_3_1", test_a_no_pending_writes_preservation),
        ("b_standalone_write_preservation_3_2", test_b_standalone_write_preservation),
        ("c_single_agent_retrieval_preservation_3_3", test_c_single_agent_retrieval_preservation),
        ("d_auto_inject_false_preservation_3_4", test_d_auto_inject_false_preservation),
        ("e_sharing_policy_preservation_3_5", test_e_sharing_policy_preservation),
        ("f_already_indexed_writes_preservation_3_6", test_f_already_indexed_writes_preservation),
        ("g_ranking_threshold_token_cap_preservation_3_7", test_g_ranking_threshold_token_cap_preservation),
        ("h_conversation_extractor_preservation_3_8", test_h_conversation_extractor_preservation),
    ]

    for name, func in test_funcs:
        print(f"\n--- ({name[0]}) {name} ---")
        try:
            func()
            record(name, True)
        except AssertionError as e:
            record(name, False, str(e)[:300])
        except Exception as e:
            record(name, False, f"Unexpected error: {traceback.format_exc()[:300]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in results_log if s == "PASS")
    failed = sum(1 for _, s, _ in results_log if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results_log)}")

    if failed == 0:
        print("\n  All preservation tests PASS — baseline behavior captured.")
        print("  These tests will validate that the write-barrier fix")
        print("  does not regress any of the eight preservation clauses (3.1–3.8).")
    else:
        print(f"\n  WARNING: {failed} test(s) FAILED — this is unexpected!")
        print("  Preservation tests should PASS on unfixed code.")
        print("  Review test logic for errors.")

    print("=" * 70)

    # Return True if ALL tests passed (expected for preservation)
    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    # Exit 0 if all preservation tests passed (expected)
    # Exit 1 if any failed (unexpected — test logic error)
    sys.exit(0 if all_passed else 1)
