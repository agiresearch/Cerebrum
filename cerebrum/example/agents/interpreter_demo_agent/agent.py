from interpreter import interpreter

from cerebrum.utils.communication import send_request
from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func


class OpenInterpreterAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name

        # prepare open-interpreter
        prepare_framework(FrameworkType.OpenInterpreter)

    def run(self, task):
        # set aios request function
        set_request_func(send_request, self.agent_name)
        final_result = interpreter.chat(task)

        return final_result


