"""Unit tests for backward-compatible search_memories calls with cross-agent parameters.

Validates Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 6.2
"""

import inspect
import unittest
from unittest.mock import patch, MagicMock

from cerebrum.memory.apis import search_memories, MemoryQuery


class TestSearchMemoriesBackwardCompat(unittest.TestCase):
    """Tests that search_memories remains backward-compatible after adding
    cross-agent filter parameters (user_id, sharing_policy)."""

    @patch("cerebrum.memory.apis.send_request")
    def test_default_params_no_new_args(self, mock_send):
        """search_memories("agent", "query") with no new params produces
        params dict {"content": "query", "k": 5}."""
        mock_send.return_value = {"response_class": "memory", "search_results": []}

        search_memories("agent", "query")

        mock_send.assert_called_once()
        _, query_obj, _ = mock_send.call_args[0]
        self.assertIsInstance(query_obj, MemoryQuery)
        self.assertEqual(query_obj.params, {"content": "query", "k": 5})

    @patch("cerebrum.memory.apis.send_request")
    def test_positional_args_no_type_error(self, mock_send):
        """Positional args search_memories("a", "q", 3, "http://url") do not
        raise TypeError."""
        mock_send.return_value = {"response_class": "memory", "search_results": []}

        # Should not raise
        search_memories("a", "q", 3, "http://url")

        mock_send.assert_called_once()
        _, query_obj, base_url = mock_send.call_args[0]
        self.assertEqual(base_url, "http://url")
        self.assertEqual(query_obj.params["k"], 3)

    @patch("cerebrum.memory.apis.send_request")
    def test_both_params_none_no_extra_keys(self, mock_send):
        """MemoryQuery with both user_id and sharing_policy as None has no
        user_id/sharing_policy keys in serialized params."""
        mock_send.return_value = {"response_class": "memory", "search_results": []}

        search_memories("a", "q", user_id=None, sharing_policy=None)

        _, query_obj, _ = mock_send.call_args[0]
        params = query_obj.model_dump()["params"]
        self.assertNotIn("user_id", params)
        self.assertNotIn("sharing_policy", params)

    @patch("cerebrum.memory.apis.send_request")
    def test_both_params_provided_included_in_params(self, mock_send):
        """search_memories with user_id and sharing_policy includes both keys
        in the params dict."""
        mock_send.return_value = {"response_class": "memory", "search_results": []}

        search_memories("a", "q", user_id="u1", sharing_policy="shared")

        _, query_obj, _ = mock_send.call_args[0]
        self.assertEqual(query_obj.params["user_id"], "u1")
        self.assertEqual(query_obj.params["sharing_policy"], "shared")

    def test_invalid_sharing_policy_raises_value_error(self):
        """search_memories with sharing_policy="invalid" raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            search_memories("a", "q", sharing_policy="invalid")
        self.assertIn("sharing_policy", str(ctx.exception))

    def test_empty_user_id_raises_value_error(self):
        """search_memories with user_id="" raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            search_memories("a", "q", user_id="")
        self.assertIn("user_id", str(ctx.exception))

    def test_whitespace_only_user_id_raises_value_error(self):
        """search_memories with user_id="   " raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            search_memories("a", "q", user_id="   ")
        self.assertIn("user_id", str(ctx.exception))

    def test_signature_includes_user_id_with_none_default(self):
        """search_memories signature includes user_id parameter with None default."""
        sig = inspect.signature(search_memories)
        self.assertIn("user_id", sig.parameters)
        param = sig.parameters["user_id"]
        self.assertEqual(param.default, None)
        # user_id should be keyword-only (after the *)
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)

    @patch("cerebrum.memory.apis.send_request")
    def test_user_id_is_stripped_before_sending(self, mock_send):
        """search_memories with user_id=" alex " strips whitespace to "alex"."""
        mock_send.return_value = {"response_class": "memory", "search_results": []}

        search_memories("agent", "query", user_id="  alex_chen  ")

        _, query_obj, _ = mock_send.call_args[0]
        self.assertEqual(query_obj.params["user_id"], "alex_chen")


if __name__ == "__main__":
    unittest.main()
