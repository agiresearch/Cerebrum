from cerebrum.manager.agent import AgentManager
from cerebrum.manager.tool import ToolManager
from cerebrum.runtime.process import LLMProcessor, RunnableAgent

from .. import config
 
class AutoAgent:
    AGENT_MANAGER = AgentManager('https://my.aios.foundation')
 
    @classmethod
    def from_preloaded(cls, agent_name: str):
        _client = config.global_client

        return RunnableAgent(_client, agent_name)


class AutoLLM:
    @classmethod
    def from_dynamic(cls):
        return LLMProcessor(config.global_client)


class AutoTool:
    TOOL_MANAGER = ToolManager('https://app.aios.foundation')

    @classmethod
    def from_preloaded(cls, tool_string: str):
        try:
            # Parse tool string
            if '/' in tool_string:
                author, name = tool_string.split('/')
            else:
                # If no '/', treat as local tool
                author = "local"
                name = tool_string
            
            tool_path = cls.TOOL_MANAGER.local_tools_dir / name
            if tool_path.exists():
                config = cls.TOOL_MANAGER.load_local_tool(name)
                version = config["meta"]["version"]
                return cls.TOOL_MANAGER.load_tool(local=True, name=name, version=version)[0]()
            
            author, name, version = cls.TOOL_MANAGER.download_tool(author, name)
            return cls.TOOL_MANAGER.load_tool(author, name, version)[0]()
        except Exception as e:
            print(f"Error loading tool {tool_string}: {str(e)}")
            raise
    
    @classmethod
    def from_batch_preload(cls, tool_names: list[str]):
        response = {
             'tools': [],
             'tool_info': []
        }

        for tool_name in tool_names:
             tool = AutoTool.from_preloaded(tool_name)

             response['tools'].append(tool.get_tool_call_format())
             response['tool_info'].append(
                {
                    "name": tool.get_tool_call_format()["function"]["name"],
                    "description": tool.get_tool_call_format()["function"]["description"],
                }
             )


        return response
