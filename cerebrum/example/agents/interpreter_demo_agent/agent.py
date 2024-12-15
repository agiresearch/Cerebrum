from interpreter import interpreter

from cerebrum.agents.base import BaseAgent
from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func


class OpenInterpreterAgent(BaseAgent):
    def __init__(self, agent_name, task_input, config_):
        super().__init__(agent_name, task_input, config_)

        self.task_input = task_input
        self.rounds = 0

        # prepare open-interpreter
        prepare_framework(FrameworkType.OpenInterpreter)

    def run(self):
        # set aios request function
        set_request_func(self.send_request, self.agent_name)
        final_result = interpreter.chat(self.task_input)
        self.rounds += 1
        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": self.rounds,
        }


