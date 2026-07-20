"""Parse AIOS kernel stdout/stderr for memory injection log lines.

The AIOS kernel emits lines like:

    Injected 5 memories (3 own + 2 shared) for user_id=elena_mitchell from agents: ['profile_agent', 'task_agent']

or the simpler variant:

    Injected 3 memories for user_id=jordan_mitchell

This module extracts injection counts per user_id from captured kernel
output, providing an alternative verification path that bypasses the
harness-side audit query (which is unreliable due to kernel metadata
filtering limitations).

Usage::

    from benchmarks.shared_memory.kernel_log_parser import (
        parse_injection_lines,
        parse_injection_entries,
    )

    # Simple: user_id → total injected count
    counts = parse_injection_lines("kernel_stdout.log")
    # {"elena_mitchell": 5, "jordan_mitchell": 3}

    # Detailed: list of structured entries
    entries = parse_injection_entries("kernel_stdout.log")
    # [InjectionLogEntry(user_id="elena_mitchell", total_count=5, ...)]
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Pattern for detailed format:
# "Injected 5 memories (3 own + 2 shared) for agent=assistant_agent, user_id=elena_mitchell"
# Also matches legacy: "Injected 5 memories (3 own + 2 shared) for user_id=elena_mitchell from agents: [...]"
_DETAILED_PATTERN = re.compile(
    r"Injected\s+(\d+)\s+(?:memories|memory)\s+"
    r"\((\d+)\s+own\s*\+\s*(\d+)\s+shared\)\s+"
    r"for\s+(?:agent=\S+,\s*)?user_id=(\S+)"
    r"(?:\s+from\s+agents?:\s*\[([^\]]*)\])?"
)

# Pattern for simple format:
# "Injected 3 memories for user_id=jordan_mitchell"
# Also matches: "Injected 3 memories for agent=assistant_agent, user_id=jordan_mitchell"
_SIMPLE_PATTERN = re.compile(
    r"Injected\s+(\d+)\s+(?:memories|memory)\s+for\s+(?:agent=\S+,\s*)?user_id=(\S+)"
)


@dataclass
class InjectionLogEntry:
    """Structured representation of a single kernel injection log line.

    Attributes:
        user_id: The user identifier the memories were injected for.
        total_count: Total number of memories injected.
        own_count: Number of own-agent memories (from detailed format).
        shared_count: Number of shared cross-agent memories (from detailed format).
        source_agents: List of agents whose memories were injected (from detailed format).
        raw_line: The original log line that was parsed.
    """

    user_id: str
    total_count: int
    own_count: Optional[int] = None
    shared_count: Optional[int] = None
    source_agents: List[str] = field(default_factory=list)
    raw_line: str = ""


def _parse_agents_list(agents_str: str) -> List[str]:
    """Parse the agents list from the log line.

    Handles formats like: 'profile_agent', 'task_agent'

    Args:
        agents_str: The raw string between brackets in the log line.

    Returns:
        List of agent name strings, stripped of quotes and whitespace.
    """
    if not agents_str or not agents_str.strip():
        return []
    # Split by comma, strip quotes and whitespace from each
    agents = []
    for part in agents_str.split(","):
        cleaned = part.strip().strip("'\"")
        if cleaned:
            agents.append(cleaned)
    return agents


def _parse_line(line: str) -> Optional[InjectionLogEntry]:
    """Attempt to parse a single log line as an injection entry.

    Tries the detailed pattern first (more specific), then falls back
    to the simple pattern.

    Args:
        line: A single line from the kernel log.

    Returns:
        InjectionLogEntry if the line matches, None otherwise.
    """
    # Try detailed pattern first (more specific, avoids false match by simple)
    match = _DETAILED_PATTERN.search(line)
    if match:
        total = int(match.group(1))
        own = int(match.group(2))
        shared = int(match.group(3))
        user_id = match.group(4)
        agents_str = match.group(5) or ""
        agents = _parse_agents_list(agents_str)
        return InjectionLogEntry(
            user_id=user_id,
            total_count=total,
            own_count=own,
            shared_count=shared,
            source_agents=agents,
            raw_line=line.rstrip("\n"),
        )

    # Try simple pattern
    match = _SIMPLE_PATTERN.search(line)
    if match:
        total = int(match.group(1))
        user_id = match.group(2)
        return InjectionLogEntry(
            user_id=user_id,
            total_count=total,
            raw_line=line.rstrip("\n"),
        )

    return None


def parse_injection_entries(log_path: str) -> List[InjectionLogEntry]:
    """Parse all injection log entries from a kernel log file.

    Reads the file line by line and extracts all lines matching the
    kernel injection patterns. Handles encoding issues gracefully by
    using ``errors="replace"`` and skips lines that cannot be decoded.

    Args:
        log_path: Path to the captured kernel stdout/stderr file.

    Returns:
        List of InjectionLogEntry instances, one per matching log line,
        in the order they appeared in the file.

    Raises:
        FileNotFoundError: If log_path does not exist.
        OSError: If the file cannot be read.
    """
    entries: List[InjectionLogEntry] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            entry = _parse_line(line)
            if entry is not None:
                entries.append(entry)

    logger.debug(
        "Parsed %d injection entries from %s", len(entries), log_path
    )
    return entries


def parse_injection_lines(log_path: str) -> dict[str, int]:
    """Parse kernel log and return user_id → total injected count mapping.

    If a user_id appears multiple times in the log (e.g., across multiple
    injection events), the counts are summed.

    This is the primary API for cross-referencing kernel ground truth
    with harness-side audit results.

    Args:
        log_path: Path to the captured kernel stdout/stderr file.

    Returns:
        Dict mapping user_id to total injected memory count.
        Empty dict if no injection lines are found or the file is empty.

    Raises:
        FileNotFoundError: If log_path does not exist.
        OSError: If the file cannot be read.
    """
    entries = parse_injection_entries(log_path)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.user_id] = counts.get(entry.user_id, 0) + entry.total_count
    return counts
