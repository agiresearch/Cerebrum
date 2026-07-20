"""Data contracts for the human-rating evaluation workflow.

Protocol: mixed_personalization_source_grounded_v1

Design:
- 4 methods × 6 unique trials = 24 unique items + 6 duplicates = 30 total
- All questions are classified as mixed_personalization (jointly require
  profile and task information).
- Primary automated comparison score: integration_score.
- Displayed context: source-grounded reference context (NOT exact model-visible).
- Single 1–5 rating per item using the integration rubric.

Design invariants:
- The blinded queue schema CANNOT represent method, judge score, trial ID,
  or duplicate status.
- Ratings are immutable one-record-per-ID submissions.
- The answer key is never exposed to the rating CLI.
"""

from dataclasses import dataclass
from typing_extensions import Literal

# ---------------------------------------------------------------------------
# Valid domain values
# ---------------------------------------------------------------------------

VALID_SOURCE_METHODS = frozenset({
    "naive_concat",
    "vanilla_rag",
    "mem0_default",
    "kernel_shared",
})

VALID_RATINGS = frozenset({1, 2, 3, 4, 5})

# The only supported evaluation dimension for this protocol
EVALUATION_DIMENSION: Literal["integration"] = "integration"

# The only supported context provenance
CONTEXT_PROVENANCE: Literal["synthetic_source_context"] = "synthetic_source_context"

# Protocol constants
UNIQUE_ITEMS_PER_METHOD = 6
DUPLICATE_COUNT = 6
TOTAL_RATING_ITEMS = 30


# ---------------------------------------------------------------------------
# Source trial schema (internal, never exposed to rater)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceTrial:
    """Typed internal representation of a source benchmark trial.

    Validation requirements:
    - source_method must be one of VALID_SOURCE_METHODS.
    - model must be "gpt-4o".
    - original_trial_id must be nonempty.
    - evaluation_dimension must be "integration".
    - reference_context must be nonempty.
    - reference_context_provenance must be "synthetic_source_context".
    - is_exact_model_visible_context must be False.
    - question and response must be nonempty.
    - judge_score must equal integration_score and be 1–5.
    - All three judge dimension scores must be 1–5.
    """

    source_method: str
    original_trial_id: str
    model: str
    evaluation_dimension: Literal["integration"]
    reference_context: str
    reference_context_provenance: Literal["synthetic_source_context"]
    is_exact_model_visible_context: bool
    question: str
    response: str
    judge_score: int
    profile_usage_score: int
    task_usage_score: int
    integration_score: int

    def __post_init__(self) -> None:
        if self.source_method not in VALID_SOURCE_METHODS:
            raise ValueError(
                f"source_method must be one of {sorted(VALID_SOURCE_METHODS)}, "
                f"got {self.source_method!r}"
            )
        if self.model != "gpt-4o":
            raise ValueError(
                f"model must be 'gpt-4o', got {self.model!r}"
            )
        if not self.original_trial_id or not self.original_trial_id.strip():
            raise ValueError("original_trial_id must be nonempty")
        if self.evaluation_dimension != "integration":
            raise ValueError(
                f"evaluation_dimension must be 'integration', "
                f"got {self.evaluation_dimension!r}"
            )
        if not self.reference_context or not self.reference_context.strip():
            raise ValueError("reference_context must be nonempty")
        if self.reference_context_provenance != "synthetic_source_context":
            raise ValueError(
                f"reference_context_provenance must be 'synthetic_source_context', "
                f"got {self.reference_context_provenance!r}"
            )
        if self.is_exact_model_visible_context is not False:
            raise ValueError(
                "is_exact_model_visible_context must be False for this protocol"
            )
        if not self.question or not self.question.strip():
            raise ValueError("question must be nonempty")
        if not self.response or not self.response.strip():
            raise ValueError("response must be nonempty")
        for field_name, value in [
            ("judge_score", self.judge_score),
            ("profile_usage_score", self.profile_usage_score),
            ("task_usage_score", self.task_usage_score),
            ("integration_score", self.integration_score),
        ]:
            if not isinstance(value, int) or value < 1 or value > 5:
                raise ValueError(
                    f"{field_name} must be an integer 1–5, got {value!r}"
                )
        if self.judge_score != self.integration_score:
            raise ValueError(
                f"judge_score ({self.judge_score}) must equal "
                f"integration_score ({self.integration_score})"
            )


# ---------------------------------------------------------------------------
# Blinded rating queue item (rater-visible ONLY)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RatingQueueItem:
    """A single blinded item presented to the rater.

    The reference_context field shows the source user profile and task
    context. It is NOT the exact context visible to the model.

    This schema intentionally CANNOT represent:
    - source_method
    - model
    - original_trial_id
    - judge_score
    - duplicate status
    - source path
    """

    blinded_id: str
    reference_context: str
    question: str
    response: str


# ---------------------------------------------------------------------------
# Answer-key item (preprocessing + compilation ONLY)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnswerKeyItem:
    """Unblinded metadata for a single queue item.

    Used by sample_and_blind (writer) and compile_ratings (reader).
    Never imported or accessed by the rate command.
    """

    blinded_id: str
    source_method: str
    model: str
    original_trial_id: str
    judge_score: int
    judge_score_dimension: Literal["integration"]
    profile_usage_score: int
    task_usage_score: int
    integration_score: int
    duplicate_of: str | None


# ---------------------------------------------------------------------------
# Rating record (append-only session file)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RatingRecord:
    """A single human rating submission.

    One 1–5 score per item using the integration rubric (how well the
    response combines profile preferences and task context into a grounded
    recommendation).

    Validation requirements:
    - rating must be an integer 1–5.
    - Each blinded_id may appear at most once in the session.
    - rated_at must be nonempty.
    - No source method, judge score, or duplicate metadata.

    Ratings are immutable: once submitted, they cannot be overwritten.
    """

    blinded_id: str
    rating: int
    note: str | None
    flagged: bool
    rated_at: str

    def __post_init__(self) -> None:
        if not self.blinded_id or not self.blinded_id.strip():
            raise ValueError("blinded_id must be nonempty")
        if not isinstance(self.rating, int) or self.rating not in VALID_RATINGS:
            raise ValueError(
                f"rating must be an integer 1–5, got {self.rating!r}"
            )
        if not self.rated_at or not self.rated_at.strip():
            raise ValueError("rated_at must be nonempty")


# ---------------------------------------------------------------------------
# Compiled item-level record (flat export)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledRecord:
    """One flat row per rating instance in the compiled export.

    Joins the rating record with the answer-key metadata and computes
    human-vs-judge agreement on the integration dimension.
    """

    blinded_id: str
    rating: int
    note: str | None
    flagged: bool
    rated_at: str
    source_method: str
    model: str
    original_trial_id: str
    judge_score: int
    profile_usage_score: int
    task_usage_score: int
    integration_score: int
    duplicate_of: str | None
    is_duplicate: bool
    human_judge_difference: int
    exact_human_judge_match: bool
    within_one_human_judge: bool
