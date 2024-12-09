import importlib
import sys
from pathlib import Path
from typing import List
import os
import traceback

class AutoTool:
    TOOL_MANAGER = ToolManager('https://app.aios.foundation')

    @classmethod
    def debug_paths(cls):
        """Print current Python paths and package locations"""
        print("===== Debug Information =====")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path:")
        for p in sys.path:
            print(f"  - {p}")
        print(f"Cerebrum package location: {importlib.__file__}")
        print("===========================")

    @classmethod
    def from_preloaded(cls, tool_string: str):
        """Get tool instance from preloaded tools"""
        cls.debug_paths()  # Add debug info
        try:
            # Parse tool string
            if '/' in tool_string:
                author, name = tool_string.split('/')
            else:
                # If no '/', treat as local tool
                author = "local"
                name = tool_string

            print(f"Loading tool: {name} from author: {author}")  # Debug info

            # Try loading tool
            tool_path = cls.TOOL_MANAGER.local_tools_dir / name
            if tool_path.exists():
                # If local tool exists, load directly
                config = cls.TOOL_MANAGER.load_local_tool(name)
                version = config["meta"]["version"]
                print(f"Loaded local tool config: {config}")  # Debug info
            else:
                # Only try downloading if local tool doesn't exist
                _, _, version = cls.TOOL_MANAGER.download_tool(author, name)

            print(f"Tool path: {tool_path}")  # Debug info
            
            if not tool_path.exists():
                raise FileNotFoundError(f"Tool path does not exist: {tool_path}")

            sys.path.insert(0, str(tool_path))
            
            try:
                module = importlib.import_module("tool")
                tool_class = getattr(module, config["build"]["module"])
                return tool_class()
            finally:
                sys.path.pop(0)
            
        except Exception as e:
            print(f"Error loading tool {tool_string}: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")  # Add full error trace
            raise

    @classmethod
    def from_batch_preload(cls, tool_strings: List[str]):
        """Batch preload tools"""
        response = {
            'tools': [],
            'tool_info': []
        }

        for tool_string in tool_strings:
            try:
                tool = cls.from_preloaded(tool_string)
                tool_format = tool.get_tool_call_format()
                
                response['tools'].append(tool_format)
                response['tool_info'].append({
                    "name": tool_format["function"]["name"],
                    "description": tool_format["function"]["description"],
                })
            except Exception as e:
                print(f"Error preloading tool {tool_string}: {str(e)}")
                continue

        return response 