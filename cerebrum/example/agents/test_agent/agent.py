from cerebrum.llm.apis import llm_chat_with_tool_call_output
import argparse

from cerebrum.config.config_manager import config

aios_kernel_url = config.get_kernel_url()

from cerebrum.interface import AutoTool

class TestAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.messages = []

    def run(self, task_input):
        self.messages.append({"role": "user", "content": task_input})
        
        tool_names = ["demo_author/arxiv"]
        
        tools = AutoTool.from_batch_preload(tool_names)["tools"]
        breakpoint()
        
        tool_call_response = llm_chat_with_tool_call_output(
            agent_name=self.agent_name,
            messages=self.messages,
            base_url=aios_kernel_url,
            tools=tools
        )
        
        tool_calls = tool_call_response["response"]["tool_calls"]
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_params = tool_call["parameters"]
            
            tool = AutoTool.from_preloaded(tool_name)
            tool_response = tool.run(tool_params)
        
        final_result = tool_response["response"]["response_message"]
        return final_result
        
def main():
    parser = argparse.ArgumentParser(description="Run test agent")
    parser.add_argument("--task_input", type=str, required=True, help="Task input for the agent")
    args = parser.parse_args()

    agent = TestAgent("test_agent")
    agent.run(args.task_input)

if __name__ == "__main__":
    main()
