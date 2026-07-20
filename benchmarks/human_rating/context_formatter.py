"""Deterministic reference-context formatter for the human-rating interface.

Formats stored synthetic profile and task context fields into a consistent
display string for the rating CLI. This is labeled as "Reference User and
Task Context" — NOT as exact model-visible context.

The formatter:
- Uses only fields stored in the result record.
- Preserves list ordering.
- Handles missing optional values.
- Produces deterministic output.
- Does NOT import benchmark prompt-building code.
- Does NOT claim byte equivalence with the model prompt.
"""


def format_reference_context(
    synthetic_profile: dict | None,
    synthetic_task_context: dict | None,
) -> str:
    """Format synthetic profile and task context into a reference display string.

    Args:
        synthetic_profile: Dict with keys: user_name, preferred_tools,
            preferred_language, response_style. May be None.
        synthetic_task_context: Dict with keys: current_project,
            active_experiment, goals, blockers, next_steps. May be None.

    Returns:
        Formatted reference context string. Empty string if both inputs
        are None.

    Raises:
        ValueError: If both inputs are None (trial has no context to display).
    """
    if synthetic_profile is None and synthetic_task_context is None:
        raise ValueError(
            "Cannot format reference context: both synthetic_profile and "
            "synthetic_task_context are None."
        )

    parts: list[str] = []

    if synthetic_profile is not None:
        parts.append("USER PROFILE")
        parts.append(f"  Name: {synthetic_profile.get('user_name', '')}")
        tools = synthetic_profile.get("preferred_tools", [])
        parts.append(f"  Preferred tools: {', '.join(tools) if tools else '(none)'}")
        parts.append(f"  Preferred language: {synthetic_profile.get('preferred_language', '')}")
        parts.append(f"  Response style: {synthetic_profile.get('response_style', '')}")

    if synthetic_task_context is not None:
        if parts:
            parts.append("")  # Blank separator line
        parts.append("TASK CONTEXT")
        parts.append(f"  Current project: {synthetic_task_context.get('current_project', '')}")
        parts.append(f"  Active experiment: {synthetic_task_context.get('active_experiment', '')}")

        goals = synthetic_task_context.get("goals", [])
        if goals:
            parts.append("  Goals:")
            for goal in goals:
                parts.append(f"    - {goal}")

        blockers = synthetic_task_context.get("blockers", [])
        if blockers:
            parts.append("  Blockers:")
            for blocker in blockers:
                parts.append(f"    - {blocker}")

        next_steps = synthetic_task_context.get("next_steps", [])
        if next_steps:
            parts.append("  Next steps:")
            for step in next_steps:
                parts.append(f"    - {step}")

    return "\n".join(parts)
