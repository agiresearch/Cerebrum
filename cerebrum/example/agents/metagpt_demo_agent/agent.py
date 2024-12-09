from cerebrum.agents.base import BaseAgent
from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func

from metagpt.software_company import generate_repo, ProjectRepo


class MetaGPTAgent(BaseAgent):
    """
    Use `export METAGPT_PROJECT_ROOT=<PATH>` sepcify project location
    """

    def __init__(self, agent_name, task_input, config_):
        super().__init__(agent_name, task_input, config_)

        self.task_input = task_input
        self.rounds = 0

        # prepare open-interpreter
        prepare_framework(FrameworkType.MetaGPT)

    def run(self):
        # set aios request function
        set_request_func(self.send_request, self.agent_name)
        repo: ProjectRepo = generate_repo(self.task_input)  # Example: Create a 2048 game

        final_result = str(repo)
        self.rounds += 1
        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": self.rounds,
        }
