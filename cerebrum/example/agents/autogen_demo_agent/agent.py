from autogen import ConversableAgent

from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func
from cerebrum.utils.communication import send_request


class AutoGenAgent:

    def __init__(self, agent_name):
        self.agent_name = agent_name

        # prepare autogen
        prepare_framework(FrameworkType.AutoGen)

    def run(self, task):
        # set aios request function
        set_request_func(send_request, self.agent_name)

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
        final_result = joe.initiate_chat(cathy, message=task, max_turns=3)
        # chat_history = final_result.chat_history

        return final_result
