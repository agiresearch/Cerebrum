"""Centralized 1–5 rating rubric for human evaluation.

This rubric is extracted directly from the GPT-5.4 automated judge prompt
in ``benchmarks/shared_memory/judge.py``. It uses the same wording for
consistency between automated and human scoring.

The rubric covers three dimensions:
- Profile Usage: references to user profile attributes
- Task Usage: references to task context details
- Integration: combining profile + task into a grounded recommendation
"""

# ---------------------------------------------------------------------------
# Profile Usage rubric
# ---------------------------------------------------------------------------

PROFILE_USAGE_RUBRIC: dict[int, str] = {
    5: (
        "Correctly and specifically references multiple profile attributes "
        "(tools, language, style) in the recommendation, regardless of "
        "response length"
    ),
    4: "Correctly references most profile attributes",
    3: "References some profile attributes but misses key ones",
    2: "Vague or incorrect references to profile attributes",
    1: (
        "No evidence of profile knowledge; response could apply to any "
        "developer"
    ),
}

# ---------------------------------------------------------------------------
# Task Usage rubric
# ---------------------------------------------------------------------------

TASK_USAGE_RUBRIC: dict[int, str] = {
    5: (
        "Correctly and specifically references project goals, blockers, "
        "and next steps in the recommendation, regardless of response length"
    ),
    4: "Correctly references most task context details",
    3: "References some task context details but misses key ones",
    2: "Vague or incorrect references to task context",
    1: (
        "No evidence of task context knowledge; response is generic advice"
    ),
}

# ---------------------------------------------------------------------------
# Integration rubric
# ---------------------------------------------------------------------------

INTEGRATION_RUBRIC: dict[int, str] = {
    5: (
        "Seamlessly combines profile preferences and task context into a "
        "single grounded recommendation"
    ),
    4: "Combines both sources with minor gaps in integration",
    3: (
        "Addresses profile and task context separately without integrating "
        "them"
    ),
    2: (
        "Mentions both sources but the recommendation does not logically "
        "follow from them"
    ),
    1: (
        "No integration; response addresses at most one source or is "
        "entirely generic"
    ),
}

# ---------------------------------------------------------------------------
# Combined rubric (for the rating CLI general display)
# ---------------------------------------------------------------------------

RATING_RUBRIC: dict[int, str] = {
    5: (
        "Excellent personalization — correctly references multiple profile "
        "attributes AND task context details, seamlessly integrated into a "
        "single grounded recommendation"
    ),
    4: (
        "Good personalization — references most profile and task attributes "
        "with minor gaps"
    ),
    3: (
        "Moderate personalization — references some attributes but misses "
        "key ones, or addresses profile and task separately"
    ),
    2: (
        "Weak personalization — vague or incorrect references, or "
        "recommendation does not logically follow from context"
    ),
    1: (
        "No personalization — no evidence of profile or task knowledge; "
        "response is entirely generic"
    ),
}
