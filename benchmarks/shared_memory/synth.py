"""Synthetic data generation for the shared memory evaluation harness.

Uses LLM calls via ``llm_chat_with_json_output`` to produce unique
synthetic profiles, task contexts, and follow-up queries for each trial.
"""

import json
import logging
import uuid
from typing import Dict, Any, List

from cerebrum.llm.apis import llm_chat, llm_chat_with_json_output
from cerebrum.config.config_manager import config

from benchmarks.shared_memory.models import (
    SyntheticProfile,
    SyntheticTaskContext,
    SyntheticTrialData,
)

logger = logging.getLogger(__name__)


def _robust_llm_json_call(agent_name, messages, kernel_url, response_format, llms):
    """Call llm_chat_with_json_output with fallback to llm_chat.

    Some kernel configurations return None for response_message when using
    the chat_with_json_output action type. This helper falls back to plain
    llm_chat and extracts JSON from the free-form text response.
    """
    llm_response = llm_chat_with_json_output(
        agent_name=agent_name,
        messages=messages,
        base_url=kernel_url,
        response_format=response_format,
        llms=llms,
    )

    raw = llm_response["response"]["response_message"]
    if raw is not None:
        return raw

    # Fallback to plain llm_chat
    logger.debug("llm_chat_with_json_output returned None, falling back to llm_chat.")
    llm_response = llm_chat(
        agent_name=agent_name,
        messages=messages,
        base_url=kernel_url,
        llms=llms,
    )
    raw = llm_response["response"]["response_message"]
    if raw is None:
        raise RuntimeError(
            f"LLM returned None response_message on both attempts. "
            f"llms={llms} kernel={kernel_url}"
        )
    return raw


def _unwrap_nested(data: dict, required_keys: List[str]) -> dict:
    """Unwrap a potentially nested LLM response to find the expected keys."""
    if all(k in data for k in required_keys):
        return data
    for value in data.values():
        if isinstance(value, dict) and all(k in value for k in required_keys):
            return value
    return data


def _validate_vague_query(
    query: str,
    profile: SyntheticProfile,
    task_context: SyntheticTaskContext,
) -> bool:
    """Return True if the query does not contain forbidden profile/task literals.

    Args:
        query: The generated follow-up query.
        profile: The synthetic profile for this trial.
        task_context: The synthetic task context for this trial.

    Returns:
        True if the query is acceptably vague (no forbidden terms found).
    """
    query_lower = query.lower()
    forbidden = [
        profile.preferred_language.lower(),
        task_context.current_project.lower(),
    ] + [t.lower() for t in profile.preferred_tools]
    return not any(term in query_lower for term in forbidden if term)


class SyntheticDataGenerator:
    """Generates synthetic trial data (profile, task context, query) via LLM."""

    def __init__(self, agent_name: str = "eval_harness", llms: list | None = None, run_id: str | None = None):
        """Initialise the generator.

        Args:
            agent_name: Agent identity used for SDK LLM calls.
            llms: Optional model override, e.g. [{"name": "qwen2.5:7b", "backend": "ollama"}].
            run_id: Optional run-specific discriminator appended to user_id
                for namespace isolation. When None, a UUID4 short suffix is
                generated automatically to prevent residual state pollution
                from prior runs.
        """
        self.agent_name = agent_name
        self.kernel_url = config.get_kernel_url()
        self.llms = llms
        self.run_id = run_id or uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------
    # Profile generation
    # ------------------------------------------------------------------

    def generate_profile(self, trial_index: int) -> SyntheticProfile:
        """Generate a synthetic user profile for a single trial.

        Args:
            trial_index: Zero-based trial number, included in the prompt
                to encourage diversity across trials.

        Returns:
            A validated ``SyntheticProfile`` instance.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    "object with exactly these keys: user_name, "
                    "preferred_tools, preferred_language, response_style. "
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a realistic software developer profile for "
                    f"trial #{trial_index}. Return a JSON object with:\n"
                    f'- "user_name": a realistic full name\n'
                    f'- "preferred_tools": a list of 2-5 developer tool names\n'
                    f'- "preferred_language": a programming language\n'
                    f'- "response_style": one of "concise", "detailed", '
                    f'"casual", or "formal"'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_profile",
                "schema": {
                    "type": "object",
                    "properties": {
                        "user_name": {"type": "string"},
                        "preferred_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "preferred_language": {"type": "string"},
                        "response_style": {"type": "string"},
                    },
                    "required": [
                        "user_name",
                        "preferred_tools",
                        "preferred_language",
                        "response_style",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        raw = _robust_llm_json_call(
            self.agent_name, messages, self.kernel_url, response_format, self.llms
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        data = _unwrap_nested(data, ["user_name", "preferred_tools", "preferred_language", "response_style"])
        return SyntheticProfile(**data)

    # ------------------------------------------------------------------
    # Task-context generation
    # ------------------------------------------------------------------

    def generate_task_context(
        self, trial_index: int, profile: SyntheticProfile
    ) -> SyntheticTaskContext:
        """Generate a synthetic task context informed by the profile.

        Args:
            trial_index: Zero-based trial number for diversity.
            profile: The previously generated profile for this trial.

        Returns:
            A validated ``SyntheticTaskContext`` instance.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    "object with exactly these keys: current_project, "
                    "active_experiment, goals, blockers, next_steps. "
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a realistic working context for a developer "
                    f"named {profile.user_name} who uses "
                    f"{profile.preferred_language} for trial #{trial_index}. "
                    f"Return a JSON object with:\n"
                    f'- "current_project": name of the project\n'
                    f'- "active_experiment": what they are currently testing\n'
                    f'- "goals": list of 2-4 goal strings\n'
                    f'- "blockers": list of 0-2 blocker strings\n'
                    f'- "next_steps": list of 2-4 next step strings'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_task_context",
                "schema": {
                    "type": "object",
                    "properties": {
                        "current_project": {"type": "string"},
                        "active_experiment": {"type": "string"},
                        "goals": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "current_project",
                        "active_experiment",
                        "goals",
                        "blockers",
                        "next_steps",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        llm_response = llm_chat_with_json_output(
            agent_name=self.agent_name,
            messages=messages,
            base_url=self.kernel_url,
            response_format=response_format,
            llms=self.llms,
        )

        raw = llm_response["response"]["response_message"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        data = _unwrap_nested(data, ["current_project", "active_experiment", "goals", "blockers", "next_steps"])
        return SyntheticTaskContext(**data)

    # ------------------------------------------------------------------
    # Follow-up query generation
    # ------------------------------------------------------------------

    def generate_vague_query(
        self,
        profile: SyntheticProfile,
        task_context: SyntheticTaskContext,
    ) -> str:
        """Generate an intentionally vague follow-up query.

        The query asks for a recommendation, prioritization, or next action
        without restating profile or task facts. It only becomes answerable
        when the assistant has access to both memory sources.

        Args:
            profile: The synthetic profile for this trial.
            task_context: The synthetic task context for this trial.

        Returns:
            A plain-text vague query string.
        """
        forbidden_terms = [
            profile.preferred_language,
            task_context.current_project,
        ] + list(profile.preferred_tools)
        forbidden_str = ", ".join(f'"{t}"' for t in forbidden_terms if t)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    'object with exactly one key: "query". '
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate a short, intentionally vague follow-up question "
                    "that a developer might ask their AI assistant. The question "
                    "should ask for a recommendation, prioritization, or next "
                    "action — something like "
                    '"Which of my pending tasks should I tackle first?" or '
                    '"What\'s the most impactful thing I could do right now?" or '
                    '"How should I prioritize what\'s on my plate?"\n\n'
                    "The developer has several pending tasks/options to choose "
                    "from. The query should implicitly reference choosing among "
                    "them or prioritizing, without naming the specific options.\n\n"
                    "CRITICAL RULES:\n"
                    "- The query MUST be vague and general on its own.\n"
                    "- The query MUST NOT mention any specific tools, "
                    "programming languages, project names, or task details.\n"
                    "- The query should only become answerable when the "
                    "assistant has access to the user's profile and task context.\n"
                    f"- Do NOT include any of these words or phrases in the "
                    f"query: {forbidden_str}\n\n"
                    'Return JSON: {"query": "your question here"}'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "follow_up_query",
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        max_attempts = 4  # 1 initial + 3 retries
        last_query = ""

        for attempt in range(max_attempts):
            llm_response = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                base_url=self.kernel_url,
                response_format=response_format,
                llms=self.llms,
            )

            raw = llm_response["response"]["response_message"]
            data = json.loads(raw) if isinstance(raw, str) else raw
            data = _unwrap_nested(data, ["query"])
            last_query = data["query"]

            if _validate_vague_query(last_query, profile, task_context):
                return last_query

            logger.warning(
                "Vague query validation failed (attempt %d/%d): %s",
                attempt + 1,
                max_attempts,
                last_query,
            )

        # Accept best-effort after exhausting retries
        logger.warning("Accepting query after %d failed validations: %s", max_attempts, last_query)
        return last_query

    # ------------------------------------------------------------------
    # Plausible actions generation
    # ------------------------------------------------------------------

    def generate_plausible_actions(
        self, profile: SyntheticProfile, task_context: SyntheticTaskContext
    ) -> List[str]:
        """Generate 3-4 plausible next actions given a profile and task context.

        The actions represent credible things the developer could do next.
        Correct prioritization among them depends on combining profile
        preferences with the task context.

        Args:
            profile: The synthetic profile for this trial.
            task_context: The synthetic task context for this trial.

        Returns:
            A list of 3-4 plausible action strings.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data generator. You MUST return a flat JSON "
                    'object with exactly one key: "actions". '
                    "Do NOT nest the object inside another key."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given this developer profile and task context, generate "
                    "3-4 plausible next actions the developer could take. Each "
                    "action should be a credible thing they could do next. The "
                    "correct prioritization among them should depend on "
                    "combining the profile preferences with the task context.\n\n"
                    f"Profile: {profile.model_dump_json()}\n"
                    f"Task Context: {task_context.model_dump_json()}\n\n"
                    'Return JSON: {"actions": ["action 1", "action 2", "action 3"]}'
                ),
            },
        ]

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "plausible_actions",
                "schema": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["actions"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        llm_response = llm_chat_with_json_output(
            agent_name=self.agent_name,
            messages=messages,
            base_url=self.kernel_url,
            response_format=response_format,
            llms=self.llms,
        )

        raw = llm_response["response"]["response_message"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        data = _unwrap_nested(data, ["actions"])
        return data["actions"]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def generate_trial_data(self, trial_index: int) -> SyntheticTrialData:
        """Generate all synthetic data for a single trial.

        Orchestrates profile → task context → plausible actions → follow-up
        query generation, and derives a stable user_id from the profile scoped
        to the current run to prevent residual state pollution.

        The user_id format is ``{name}_{run_id}_t{trial_index}`` where name
        is the lowercase underscore-separated profile name, run_id is the
        run-specific discriminator, and trial_index guarantees uniqueness
        even when the LLM generates duplicate names within a single run.

        Args:
            trial_index: Zero-based trial number.

        Returns:
            A ``SyntheticTrialData`` bundle with profile, task context,
            follow-up query, plausible actions, and run-scoped user_id.
        """
        profile = self.generate_profile(trial_index)
        task_context = self.generate_task_context(trial_index, profile)
        plausible_actions = self.generate_plausible_actions(profile, task_context)
        follow_up_query = self.generate_vague_query(profile, task_context)
        base_name = profile.user_name.lower().replace(" ", "_")
        user_id = f"{base_name}_{self.run_id}_t{trial_index}"

        return SyntheticTrialData(
            profile=profile,
            task_context=task_context,
            follow_up_query=follow_up_query,
            plausible_actions=plausible_actions,
            user_id=user_id,
        )
