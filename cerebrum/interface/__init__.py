from cerebrum.manager.agent import AgentManager
from cerebrum.manager.tool import ToolManager
from cerebrum.runtime.process import LLMProcessor, RunnableAgent
from cerebrum.config.config_manager import config

from .. import config as cerebrum_config
 
class AutoAgent:
    hub_url = config.get('manager', 'agent_hub_url')
    print(f"[DEBUG] Using Agent Hub URL: {hub_url}")
    AGENT_MANAGER = AgentManager(hub_url)
 
    @classmethod
    def from_preloaded(cls, agent_name: str):
        _client = cerebrum_config.global_client

        return RunnableAgent(_client, agent_name)


class AutoLLM:
    @classmethod
    def from_dynamic(cls):
        return LLMProcessor(cerebrum_config.global_client)


class AutoTool:
    hub_url = config.get('manager', 'tool_hub_url')
    print(f"[DEBUG] Using Tool Hub URL: {hub_url}")
    TOOL_MANAGER = ToolManager(hub_url)

    @classmethod
    def from_preloaded(cls, tool_name: str):
        n_slash = tool_name.count('/')
        if n_slash == 1: # load from author/name
        # if tool_name.split('/')[0] != 'core':
            try:
                author, name, version = cls.TOOL_MANAGER.download_tool(
                    author=tool_name.split('/')[0],
                    name=tool_name.split('/')[1]
                )
                tool, _ = cls.TOOL_MANAGER.load_tool(author, name, version)
            except:
                print('reload',tool_name.split('/')[1])
                tool, _ = cls.TOOL_MANAGER.load_tool(local=True, name=tool_name.split('/')[1])
        else:
            tool, _ = cls.TOOL_MANAGER.load_tool(local=True, name=tool_name)
        
        #return tool instance, not class
        return tool()
    
    @classmethod
    def from_batch_preload(cls, tool_names: list[str]):
        response = {
             'tools': [],
             'tool_info': []
        }

        for tool_name in tool_names:
             print('tool name', tool_name)
             tool = AutoTool.from_preloaded(tool_name)

             response['tools'].append(tool.get_tool_call_format())
             response['tool_info'].append(
                {
                    "name": tool.get_tool_call_format()["function"]["name"],
                    "description": tool.get_tool_call_format()["function"]["description"],
                }
             )


        return response
