"""Unit tests for the --method CLI argument in run_evaluation.py.

Verifies that:
- --method accepts all five valid values
- --method defaults to kernel_shared when not specified
- Invalid method names are rejected with non-zero exit
- --condition phase1 and --condition phase2 still parse without error

Validates: Requirements 1.1, 1.4, 1.5
"""

import sys
sys.path.insert(0, ".")

import argparse

from benchmarks.shared_memory.run_evaluation import build_arg_parser


def test_method_accepts_kernel_shared():
    """--method kernel_shared is a valid choice."""
    parser = build_arg_parser()
    args = parser.parse_args(["--method", "kernel_shared"])
    assert args.method == "kernel_shared", (
        f"Expected method='kernel_shared', got {args.method!r}"
    )
    print("PASSED: --method kernel_shared accepted")


def test_method_accepts_naive_concat():
    """--method naive_concat is a valid choice."""
    parser = build_arg_parser()
    args = parser.parse_args(["--method", "naive_concat"])
    assert args.method == "naive_concat", (
        f"Expected method='naive_concat', got {args.method!r}"
    )
    print("PASSED: --method naive_concat accepted")


def test_method_accepts_vanilla_rag():
    """--method vanilla_rag is a valid choice."""
    parser = build_arg_parser()
    args = parser.parse_args(["--method", "vanilla_rag"])
    assert args.method == "vanilla_rag", (
        f"Expected method='vanilla_rag', got {args.method!r}"
    )
    print("PASSED: --method vanilla_rag accepted")


def test_method_accepts_mem0_default():
    """--method mem0_default is a valid choice."""
    parser = build_arg_parser()
    args = parser.parse_args(["--method", "mem0_default"])
    assert args.method == "mem0_default", (
        f"Expected method='mem0_default', got {args.method!r}"
    )
    print("PASSED: --method mem0_default accepted")


def test_method_accepts_all():
    """--method all is a valid choice."""
    parser = build_arg_parser()
    args = parser.parse_args(["--method", "all"])
    assert args.method == "all", (
        f"Expected method='all', got {args.method!r}"
    )
    print("PASSED: --method all accepted")


def test_method_defaults_to_kernel_shared():
    """When --method is not specified, it defaults to kernel_shared."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.method == "kernel_shared", (
        f"Expected default method='kernel_shared', got {args.method!r}"
    )
    print("PASSED: --method defaults to kernel_shared")


def test_invalid_method_rejected():
    """Invalid method names are rejected with SystemExit (non-zero exit)."""
    parser = build_arg_parser()
    try:
        parser.parse_args(["--method", "invalid_method"])
        assert False, "Expected SystemExit for invalid method name"
    except SystemExit as e:
        assert e.code != 0, (
            f"Expected non-zero exit code for invalid method, got {e.code}"
        )
    print("PASSED: invalid method name rejected with non-zero exit")


def test_invalid_method_rejected_random_string():
    """Another invalid method name is also rejected."""
    parser = build_arg_parser()
    try:
        parser.parse_args(["--method", "gpt4_magic"])
        assert False, "Expected SystemExit for invalid method name"
    except SystemExit as e:
        assert e.code != 0, (
            f"Expected non-zero exit code for invalid method, got {e.code}"
        )
    print("PASSED: random invalid method name rejected with non-zero exit")


def test_condition_phase1_parses():
    """--condition phase1 parses without error."""
    parser = build_arg_parser()
    args = parser.parse_args(["--condition", "phase1"])
    assert args.condition == "phase1", (
        f"Expected condition='phase1', got {args.condition!r}"
    )
    print("PASSED: --condition phase1 parses without error")


def test_condition_phase2_parses():
    """--condition phase2 parses without error."""
    parser = build_arg_parser()
    args = parser.parse_args(["--condition", "phase2"])
    assert args.condition == "phase2", (
        f"Expected condition='phase2', got {args.condition!r}"
    )
    print("PASSED: --condition phase2 parses without error")


def test_condition_and_method_can_coexist():
    """--condition and --method can both be specified without error."""
    parser = build_arg_parser()
    args = parser.parse_args(["--condition", "phase1", "--method", "naive_concat"])
    assert args.condition == "phase1"
    assert args.method == "naive_concat"
    print("PASSED: --condition and --method can coexist")


def test_all_five_valid_values():
    """All five valid method values are accepted by the parser."""
    parser = build_arg_parser()
    valid_values = ["kernel_shared", "naive_concat", "vanilla_rag", "mem0_default", "all"]
    for value in valid_values:
        args = parser.parse_args(["--method", value])
        assert args.method == value, (
            f"Expected method={value!r}, got {args.method!r}"
        )
    print("PASSED: all five valid method values accepted")


if __name__ == "__main__":
    test_method_accepts_kernel_shared()
    test_method_accepts_naive_concat()
    test_method_accepts_vanilla_rag()
    test_method_accepts_mem0_default()
    test_method_accepts_all()
    test_method_defaults_to_kernel_shared()
    test_invalid_method_rejected()
    test_invalid_method_rejected_random_string()
    test_condition_phase1_parses()
    test_condition_phase2_parses()
    test_condition_and_method_can_coexist()
    test_all_five_valid_values()
    print("\nAll CLI tests passed.")
