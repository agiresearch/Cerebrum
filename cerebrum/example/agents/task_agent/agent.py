import os
import json
import logging

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.memory.apis import create_memory, update_memory, search_memories
from cerebrum.config.config_manager import config

from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    FIELD_SHARING_POLICY,
    MEMORY_TYPE_TASK_CONTEXT,
    POLICY_PRIVATE,
    POLICY_SHARED,
)

aios_kernel_url = config.get_kernel_url()

logger = logging.getLogger(__name__)


class TaskAgent:
    """Agent that extracts and stores working context from user input.

    Analyzes user input to identify short- to medium-term working context
    such as current project, active experiment, goals, blockers, and
    next steps. Uses llm_chat_with_json_output for structured extraction
    and upserts task context memories via the kernel memory layer.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.messages = []
        self.rounds = 0

    def load_config(self) -> dict:
        """Load agent configuration from config.json in the agent's directory."""
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, "config.json")

        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def run(self, task_input: str) -> dict:
        """Extract and store working context from input.

        Args:
            task_input: User input text to extract task context from.

        Returns:
            Dict with agent_name, result, and rounds.
        """
        try:
            # Extract structured task context from input
            context_data = self._extract_task_context(task_input)
            self.rounds += 1

            # Upsert task context memory using explicit request-scoped user_id only
            user_id = getattr(self, 'user_id', None)
            if not user_id or not str(user_id).strip():
                logger.warning(
                    "Skipping memory write: no explicit user_id provided to TaskAgent"
                )
                result_summary = (
                    f"Extracted task context (memory write skipped — no user_id): "
                    f"project={context_data.get('current_project', '')}, "
                    f"experiment={context_data.get('active_experiment', '')}, "
                    f"goals={context_data.get('goals', [])}, "
                    f"blockers={context_data.get('blockers', [])}, "
                    f"next_steps={context_data.get('next_steps', [])}"
                )
                return {
                    "agent_name": self.agent_name,
                    "result": result_summary,
                    "rounds": self.rounds,
                }
            user_id = str(user_id).strip()
            memory_ids = self._upsert_task_memory(user_id, context_data)

            result_summary = (
                f"Extracted task context: "
                f"project={context_data.get('current_project', '')}, "
                f"experiment={context_data.get('active_experiment', '')}, "
                f"goals={context_data.get('goals', [])}, "
                f"blockers={context_data.get('blockers', [])}, "
                f"next_steps={context_data.get('next_steps', [])}. "
                f"Memory IDs: {memory_ids}"
            )

            return {
                "agent_name": self.agent_name,
                "result": result_summary,
                "rounds": self.rounds,
            }

        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "result": f"Error: {e}",
                "rounds": self.rounds,
            }

    def _extract_task_context(self, task_input: str) -> dict:
        """Use llm_chat_with_json_output to extract structured task context.

        Args:
            task_input: Raw user input text.

        Returns:
            Dict with keys: current_project, active_experiment,
            goals, blockers, next_steps.
        """
        system_instruction = "".join(self.config.get("description", []))
        self.messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": task_input},
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_context",
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

        response = llm_chat_with_json_output(
            agent_name=self.agent_name,
            messages=self.messages,
            base_url=aios_kernel_url,
            llms=getattr(self, 'llms', None),
            response_format=response_format,
        )

        response_message = response["response"]["response_message"]
        if isinstance(response_message, str):
            context_data = json.loads(response_message)
        else:
            context_data = response_message

        return context_data

    def _upsert_task_memory(self, user_id: str, context_data: dict) -> list:
        """Search for existing task context memories, update or create.

        Args:
            user_id: Identifier for the user this memory pertains to.
            context_data: Extracted task context dict to store.

        Returns:
            List of memory IDs that were created or updated.
        """
        memory_ids = []
        content = json.dumps(context_data)

        # Search for existing task context memories
        search_response = search_memories(
            agent_name=self.agent_name,
            query=f"task context {user_id}",
            base_url=aios_kernel_url,
            user_id=user_id,
        )

        existing_results = []
        if search_response and isinstance(search_response, dict):
            resp = search_response.get("response", {})
            if resp and isinstance(resp, dict):
                existing_results = resp.get("search_results", []) or []

        # Filter for task_context-type memories owned by this agent
        matching = [
            r for r in existing_results
            if r.get("metadata", {}).get("memory_type") == MEMORY_TYPE_TASK_CONTEXT
        ]

        if matching:
            # Update existing task context memory
            for mem in matching:
                memory_id = mem.get("memory_id", mem.get("id", ""))
                if memory_id:
                    update_memory(
                        agent_name=self.agent_name,
                        memory_id=memory_id,
                        content=content,
                        base_url=aios_kernel_url,
                    )
                    memory_ids.append(memory_id)
        else:
            # Create new task context memory
            metadata = build_memory_metadata(
                owner_agent=self.agent_name,
                user_id=user_id,
                memory_type=MEMORY_TYPE_TASK_CONTEXT,
                sharing_policy=POLICY_SHARED if getattr(self, 'share_memory', False) else POLICY_PRIVATE,
            )
            create_response = create_memory(
                agent_name=self.agent_name,
                content=content,
                metadata=metadata,
                base_url=aios_kernel_url,
            )
            if create_response and isinstance(create_response, dict):
                resp = create_response.get("response", {})
                if resp and isinstance(resp, dict):
                    mid = resp.get("memory_id", "")
                    if mid:
                        memory_ids.append(mid)

        return memory_ids

    def share_memory(self, memory_id: str) -> None:
        """Phase 2: Mark a task context memory as shared.

        Sets the sharing_policy metadata to "shared" on an already-stored
        memory so that other agents can discover it via search_memories.

        Args:
            memory_id: ID of the memory to share.
        """
        update_memory(
            agent_name=self.agent_name,
            memory_id=memory_id,
            metadata={FIELD_SHARING_POLICY: POLICY_SHARED},
            base_url=aios_kernel_url,
        )

    def revoke_sharing(self, memory_id: str) -> None:
        """Phase 2: Revoke sharing on a task context memory.

        Sets the sharing_policy metadata back to "private" so that the
        memory is no longer visible to other agents.

        Args:
            memory_id: ID of the memory to make private.
        """
        update_memory(
            agent_name=self.agent_name,
            memory_id=memory_id,
            metadata={FIELD_SHARING_POLICY: POLICY_PRIVATE},
            base_url=aios_kernel_url,
        )
