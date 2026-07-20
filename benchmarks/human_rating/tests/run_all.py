"""Run all human-rating test suites. Stops on first failure.

Usage:
    python benchmarks/human_rating/tests/run_all.py
"""

import subprocess
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

SUITES = [
    "benchmarks/human_rating/tests/test_schemas.py",
    "benchmarks/human_rating/tests/test_context_formatter.py",
    "benchmarks/human_rating/tests/test_trial_loader.py",
    "benchmarks/human_rating/tests/test_sampling.py",
    "benchmarks/human_rating/tests/test_blinding.py",
    "benchmarks/human_rating/tests/test_artifact_writer.py",
    "benchmarks/human_rating/tests/test_rate.py",
    "benchmarks/human_rating/tests/test_compile_ratings.py",
    "benchmarks/human_rating/tests/test_artifact_audit.py",
    "benchmarks/human_rating/tests/test_end_to_end.py",
]


def main():
    print("=" * 60)
    print("  Human Rating — Full Test Suite")
    print("=" * 60)

    passed = 0
    for suite in SUITES:
        name = suite.split("/")[-1].replace(".py", "")
        print(f"\n▶ {name}")
        result = subprocess.run(
            [sys.executable, suite],
            cwd=_project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED!")
            print(result.stdout[-500:] if result.stdout else "")
            print(result.stderr[-500:] if result.stderr else "")
            sys.exit(1)
        passed += 1
        print(f"  ✓ passed")

    print(f"\n{'=' * 60}")
    print(f"  ALL {passed} SUITES PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
