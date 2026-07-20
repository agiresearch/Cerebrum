"""Directory layout and path helpers for the human-rating workflow.

Runtime layout:
    human_rating_runs/
    └── <run_name>/
        ├── rater/
        │   ├── rating_queue.json
        │   └── ratings.jsonl
        ├── private/
        │   └── answer_key.json
        └── compiled/
            ├── item_level_results.csv
            ├── method_question_type_summary.csv
            ├── human_judge_agreement.json
            └── intra_rater_consistency.json

The rating command receives ONLY paths under rater/.
It must NOT scan parent directories, discover the answer key, accept an
answer-key argument, import compilation code, or print the private path.
"""

import os
from dataclasses import dataclass


# Default base directory (relative to project root)
DEFAULT_BASE_DIR = "human_rating_runs"


@dataclass(frozen=True)
class RunPaths:
    """All filesystem paths for a single human-rating run.

    Separates rater-visible artifacts from private answer-key artifacts.
    """

    run_dir: str

    @property
    def rater_dir(self) -> str:
        """Rater-visible directory."""
        return os.path.join(self.run_dir, "rater")

    @property
    def private_dir(self) -> str:
        """Private directory containing the answer key."""
        return os.path.join(self.run_dir, "private")

    @property
    def compiled_dir(self) -> str:
        """Directory for compiled output artifacts."""
        return os.path.join(self.run_dir, "compiled")

    # --- Rater-visible files ---

    @property
    def rating_queue_path(self) -> str:
        """Path to the blinded rating queue JSON."""
        return os.path.join(self.rater_dir, "rating_queue.json")

    @property
    def ratings_path(self) -> str:
        """Path to the append-only ratings JSONL file."""
        return os.path.join(self.rater_dir, "ratings.jsonl")

    # --- Private files ---

    @property
    def answer_key_path(self) -> str:
        """Path to the answer key JSON (never exposed to rater)."""
        return os.path.join(self.private_dir, "answer_key.json")

    # --- Compiled output files ---

    @property
    def item_level_csv_path(self) -> str:
        """Path to the flat item-level results CSV."""
        return os.path.join(self.compiled_dir, "item_level_results.csv")

    @property
    def method_summary_csv_path(self) -> str:
        """Path to the method × question_type summary CSV."""
        return os.path.join(self.compiled_dir, "method_question_type_summary.csv")

    @property
    def agreement_json_path(self) -> str:
        """Path to human-judge agreement metrics JSON."""
        return os.path.join(self.compiled_dir, "human_judge_agreement.json")

    @property
    def consistency_json_path(self) -> str:
        """Path to intra-rater consistency metrics JSON."""
        return os.path.join(self.compiled_dir, "intra_rater_consistency.json")


def get_run_paths(run_name: str, base_dir: str | None = None) -> RunPaths:
    """Construct RunPaths for a given run name.

    Args:
        run_name: Name of the evaluation run (directory name).
        base_dir: Override for the base directory. Defaults to
            DEFAULT_BASE_DIR relative to the current working directory.

    Returns:
        A RunPaths instance with all path properties configured.
    """
    base = base_dir if base_dir is not None else DEFAULT_BASE_DIR
    return RunPaths(run_dir=os.path.join(base, run_name))


def get_rater_paths(queue_path: str) -> tuple[str, str]:
    """Derive the ratings file path from the queue file path.

    The rating CLI receives only the queue path and derives the ratings
    path from it. This function encapsulates that logic.

    Args:
        queue_path: Path to the rating_queue.json file.

    Returns:
        Tuple of (queue_path, ratings_path).
    """
    rater_dir = os.path.dirname(queue_path)
    ratings_path = os.path.join(rater_dir, "ratings.jsonl")
    return queue_path, ratings_path
