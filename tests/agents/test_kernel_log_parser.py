"""Unit tests for benchmarks/shared_memory/kernel_log_parser.py.

Tests parsing of AIOS kernel injection log lines in both detailed and
simple formats, including edge cases like malformed lines, empty files,
encoding issues, and multiple entries for the same user_id.
"""

import os
import sys
import tempfile
import unittest

# Ensure the project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.shared_memory.kernel_log_parser import (
    InjectionLogEntry,
    parse_injection_entries,
    parse_injection_lines,
    _parse_line,
    _parse_agents_list,
)


class TestParseAgentsList(unittest.TestCase):
    """Tests for the _parse_agents_list helper."""

    def test_typical_agents_list(self):
        result = _parse_agents_list("'profile_agent', 'task_agent'")
        self.assertEqual(result, ["profile_agent", "task_agent"])

    def test_single_agent(self):
        result = _parse_agents_list("'profile_agent'")
        self.assertEqual(result, ["profile_agent"])

    def test_double_quoted_agents(self):
        result = _parse_agents_list('"profile_agent", "task_agent"')
        self.assertEqual(result, ["profile_agent", "task_agent"])

    def test_empty_string(self):
        result = _parse_agents_list("")
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        result = _parse_agents_list("   ")
        self.assertEqual(result, [])

    def test_no_quotes(self):
        result = _parse_agents_list("profile_agent, task_agent")
        self.assertEqual(result, ["profile_agent", "task_agent"])


class TestParseLine(unittest.TestCase):
    """Tests for parsing individual log lines."""

    def test_detailed_format(self):
        line = "Injected 5 memories (3 own + 2 shared) for user_id=elena_mitchell from agents: ['profile_agent', 'task_agent']"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, "elena_mitchell")
        self.assertEqual(entry.total_count, 5)
        self.assertEqual(entry.own_count, 3)
        self.assertEqual(entry.shared_count, 2)
        self.assertEqual(entry.source_agents, ["profile_agent", "task_agent"])

    def test_simple_format(self):
        line = "Injected 3 memories for user_id=jordan_mitchell"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, "jordan_mitchell")
        self.assertEqual(entry.total_count, 3)
        self.assertIsNone(entry.own_count)
        self.assertIsNone(entry.shared_count)
        self.assertEqual(entry.source_agents, [])

    def test_single_memory_singular(self):
        line = "Injected 1 memory for user_id=test_user"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, "test_user")
        self.assertEqual(entry.total_count, 1)

    def test_detailed_singular(self):
        line = "Injected 1 memory (1 own + 0 shared) for user_id=solo_user from agents: ['profile_agent']"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.total_count, 1)
        self.assertEqual(entry.own_count, 1)
        self.assertEqual(entry.shared_count, 0)

    def test_detailed_without_agents_list(self):
        line = "Injected 4 memories (2 own + 2 shared) for user_id=alice_wonder"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, "alice_wonder")
        self.assertEqual(entry.total_count, 4)
        self.assertEqual(entry.own_count, 2)
        self.assertEqual(entry.shared_count, 2)
        self.assertEqual(entry.source_agents, [])

    def test_line_with_timestamp_prefix(self):
        line = "2026-06-24 10:05:32 [INFO] Injected 5 memories (3 own + 2 shared) for user_id=elena_mitchell from agents: ['profile_agent', 'task_agent']"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, "elena_mitchell")
        self.assertEqual(entry.total_count, 5)

    def test_non_matching_line(self):
        line = "Starting AIOS kernel on port 8000..."
        entry = _parse_line(line)
        self.assertIsNone(entry)

    def test_empty_line(self):
        entry = _parse_line("")
        self.assertIsNone(entry)

    def test_partial_match_missing_user_id(self):
        line = "Injected 3 memories for something_else"
        entry = _parse_line(line)
        self.assertIsNone(entry)

    def test_user_id_with_run_suffix(self):
        line = "Injected 2 memories for user_id=elena_mitchell_20260624T100532"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, "elena_mitchell_20260624T100532")
        self.assertEqual(entry.total_count, 2)

    def test_raw_line_preserved(self):
        line = "Injected 3 memories for user_id=test_user\n"
        entry = _parse_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.raw_line, "Injected 3 memories for user_id=test_user")


class TestParseInjectionEntries(unittest.TestCase):
    """Tests for parse_injection_entries with real files."""

    def _write_temp_log(self, content: str) -> str:
        """Write content to a temporary file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".log")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_mixed_formats(self):
        content = """\
2026-06-24 10:00:01 [INFO] Starting memory injection...
Injected 5 memories (3 own + 2 shared) for user_id=elena_mitchell from agents: ['profile_agent', 'task_agent']
2026-06-24 10:00:02 [INFO] Processing next request...
Injected 3 memories for user_id=jordan_mitchell
2026-06-24 10:00:03 [INFO] Done.
"""
        path = self._write_temp_log(content)
        try:
            entries = parse_injection_entries(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].user_id, "elena_mitchell")
            self.assertEqual(entries[0].total_count, 5)
            self.assertEqual(entries[1].user_id, "jordan_mitchell")
            self.assertEqual(entries[1].total_count, 3)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = self._write_temp_log("")
        try:
            entries = parse_injection_entries(path)
            self.assertEqual(entries, [])
        finally:
            os.unlink(path)

    def test_no_matching_lines(self):
        content = """\
Starting AIOS kernel...
Loading model qwen2.5:7b...
Ready to serve requests.
"""
        path = self._write_temp_log(content)
        try:
            entries = parse_injection_entries(path)
            self.assertEqual(entries, [])
        finally:
            os.unlink(path)

    def test_multiple_entries_same_user(self):
        content = """\
Injected 2 memories for user_id=alice
Injected 3 memories for user_id=alice
"""
        path = self._write_temp_log(content)
        try:
            entries = parse_injection_entries(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].total_count, 2)
            self.assertEqual(entries[1].total_count, 3)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_injection_entries("/nonexistent/path/to/file.log")

    def test_encoding_issues_handled(self):
        """Test that binary/encoding issues are handled gracefully."""
        fd, path = tempfile.mkstemp(suffix=".log")
        with os.fdopen(fd, "wb") as f:
            f.write(b"Injected 2 memories for user_id=test_user\n")
            f.write(b"\xff\xfe Invalid UTF-8 bytes\n")
            f.write(b"Injected 1 memory for user_id=another_user\n")
        try:
            entries = parse_injection_entries(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].user_id, "test_user")
            self.assertEqual(entries[1].user_id, "another_user")
        finally:
            os.unlink(path)


class TestParseInjectionLines(unittest.TestCase):
    """Tests for parse_injection_lines (the main dict API)."""

    def _write_temp_log(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".log")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_basic_mapping(self):
        content = """\
Injected 5 memories (3 own + 2 shared) for user_id=elena_mitchell from agents: ['profile_agent', 'task_agent']
Injected 3 memories for user_id=jordan_mitchell
"""
        path = self._write_temp_log(content)
        try:
            counts = parse_injection_lines(path)
            self.assertEqual(counts, {
                "elena_mitchell": 5,
                "jordan_mitchell": 3,
            })
        finally:
            os.unlink(path)

    def test_summing_multiple_entries(self):
        content = """\
Injected 2 memories for user_id=alice
Injected 3 memories for user_id=alice
Injected 1 memory for user_id=bob
"""
        path = self._write_temp_log(content)
        try:
            counts = parse_injection_lines(path)
            self.assertEqual(counts, {
                "alice": 5,
                "bob": 1,
            })
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty_dict(self):
        path = self._write_temp_log("")
        try:
            counts = parse_injection_lines(path)
            self.assertEqual(counts, {})
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_injection_lines("/does/not/exist.log")


class TestIntegrationWithRunEvaluation(unittest.TestCase):
    """Integration tests for --kernel-logs CLI argument and orchestrator."""

    def test_build_arg_parser_has_kernel_logs(self):
        """Verify --kernel-logs is a valid CLI argument."""
        from benchmarks.shared_memory.run_evaluation import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args(["--kernel-logs", "/tmp/kernel.log"])
        self.assertEqual(args.kernel_logs, "/tmp/kernel.log")

    def test_build_arg_parser_default_kernel_logs_none(self):
        """Verify --kernel-logs defaults to None."""
        from benchmarks.shared_memory.run_evaluation import build_arg_parser
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.kernel_logs)


if __name__ == "__main__":
    unittest.main()
