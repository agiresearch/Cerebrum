from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func
from cerebrum.utils.communication import send_request

from metagpt.software_company import generate_repo, ProjectRepo


class MetaGPTAgent:
    """
    Use `export METAGPT_PROJECT_ROOT=<PATH>` sepcify project location
    """

    def __init__(self, agent_name):
        self.agent_name = agent_name

        # prepare open-interpreter
        prepare_framework(FrameworkType.MetaGPT)

    def run(self, task):
        # set aios request function
        set_request_func(send_request, self.agent_name)
        repo: ProjectRepo = generate_repo(task)  # Example: Create a 2048 game

        final_result = str(repo)
        return final_result
