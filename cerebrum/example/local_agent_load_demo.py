from pathlib import Path
from cerebrum.manager.agent import AgentManager

manager = AgentManager(base_url='https://app.aios.foundation')

agent = manager.load_agent(local=True, path=f"/Users/rama2r/Cerebrum/example/agents/academic_agent")

print(agent)