import os
import json

from cerebrum.llm.apis import llm_chat
from cerebrum.config.config_manager import config

aios_kernel_url = config.get_kernel_url()


class AssistantAgent:
    """A personalized assistant agent that helps users with queries.

    Issues plain llm_chat calls. The kernel handles all memory operations:
    - auto_extract stores conversation turns as memories automatically
    - auto_inject retrieves and injects relevant memories into context
    """

    def __init__(self, agent_name: str, llms: list | None = None):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.messages = []
        self.rounds = 0
        self.llms = llms  # Optional model override, e.g. [{"name": "qwen2.5:7b", "backend": "ollama"}]

    def load_config(self) -> dict:
        """Load agent configuration from config.json in the agent's directory."""
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, "config.json")

        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def run(self, task_input: str) -> dict:
        """Process user query and return a result dictionary.

        Args:
            task_input: The user's query string.

        Returns:
            A dict with agent_name, result, and rounds.
        """
        try:
            # Build system instruction from config description
            system_instruction = "".join(self.config.get("description", []))
            self.messages.append({"role": "system", "content": system_instruction})

            # Append user query
            self.messages.append({"role": "user", "content": task_input})

            # Call LLM — kernel handles memory injection and extraction
            response = llm_chat(
                agent_name=self.agent_name,
                messages=self.messages,
                base_url=aios_kernel_url,
                llms=self.llms,
                user_id=getattr(self, "user_id", None),
            )

            result_text = response["response"]["response_message"] if response else ""
            self.messages.append({"role": "assistant", "content": result_text})
            self.rounds += 1

            return {
                "agent_name": self.agent_name,
                "result": result_text,
                "rounds": self.rounds,
            }

        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "result": f"Error: {e}",
                "rounds": self.rounds,
            }
