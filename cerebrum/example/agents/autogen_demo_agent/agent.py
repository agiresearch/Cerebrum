from autogen import ConversableAgent

from cerebrum.agents.base import BaseAgent
from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func


class AutoGenAgent(BaseAgent):

    def __init__(self, agent_name, task_input, config_):
        super().__init__(agent_name, task_input, config_)

        self.task_input = task_input
        self.rounds = 0

        # prepare autogen
        prepare_framework(FrameworkType.AutoGen)

    def run(self):
        # set aios request function
        set_request_func(self.send_request, self.agent_name)

        cathy = ConversableAgent(
            "cathy",
            system_message="Your name is Cathy and you are a teacher. You will try to teach a student how to "
                           "solve problem.",
            human_input_mode="NEVER",  # Never ask for human input.
        )

        joe = ConversableAgent(
            "joe",
            system_message="Your name is Joe and you are a student.",
            human_input_mode="NEVER",  # Never ask for human input.
        )

        # Let the assistant start the conversation.  It will end when the user types exit.
        final_result = joe.initiate_chat(cathy, message=self.task_input, max_turns=3)
        chat_history = final_result.chat_history

        self.rounds += 1
        return {
            "agent_name": self.agent_name,
            "result": chat_history,
            "rounds": self.rounds,
        }
