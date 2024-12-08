import json
import os
from pathlib import Path
from typing import Tuple
import traceback

class ToolManager:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.cache_dir = Path(os.path.expanduser("~")) / ".cache" / "cerebrum" / "tools"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_tools = {}
        
        # Set up local tools directory
        current_file = Path(__file__).resolve()
        self.local_tools_dir = current_file.parent.parent / "example" / "tools"
        print(f"Initialized ToolManager with local tools directory: {self.local_tools_dir}")

    def _version_to_path(self, version):
        """Convert version string to path-safe format"""
        if version is None or version == "latest":
            return "latest"
        return str(version).replace(".", "-")

    def load_local_tool(self, name: str):
        """Load tool from local directory"""
        try:
            tool_path = self.local_tools_dir / name
            if not tool_path.exists():
                raise FileNotFoundError(f"Tool {name} not found in local directory")
            
            config_path = tool_path / "config.json"
            with open(config_path) as f:
                config = json.load(f)
                
            return config
            
        except Exception as e:
            print(f"Error loading local tool {name}: {str(e)}")
            raise

    def download_tool(self, author: str, name: str, version: str = None) -> Tuple[str, str, str]:
        """Download tool from registry if not in cache"""
        print(f"Attempting to download tool: {author}/{name} version: {version}")
        try:
            # First check if it's a local tool
            tool_path = self.local_tools_dir / name
            if tool_path.exists():
                config = self.load_local_tool(name)
                actual_version = config["meta"]["version"]
                print(f"Using local tool: {name} version: {actual_version}")
                return author, name, actual_version
            
            # If not local tool, check cache
            tool_key = f"{author}/{name}"
            if tool_key in self.loaded_tools:
                cached_version = self.loaded_tools[tool_key]
                print(f"Using cached tool: {tool_key} version: {cached_version}")
                return author, name, cached_version

            # If neither local nor cached, set version and create cache
            actual_version = version if version else "latest"
            print(f"Using version: {actual_version}")
            
            cache_path = self._get_cache_path(author, name, actual_version)
            print(f"Cache path: {cache_path}")
            
            if not cache_path.exists():
                print(f"Creating new tool cache at: {cache_path}")
                os.makedirs(cache_path.parent, exist_ok=True)
                self._create_empty_tool_package(cache_path)
            
            self.loaded_tools[tool_key] = actual_version
            return author, name, actual_version
            
        except Exception as e:
            print(f"Error downloading tool {author}/{name}: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise

    def _create_empty_tool_package(self, cache_path: Path):
        """Create an empty tool package for testing"""
        try:
            with open(cache_path, 'wb') as f:
                f.write(b'{"files": {}, "config": {}}')
        except Exception as e:
            print(f"Error creating empty tool package: {str(e)}")
            raise

    def parse_tool_string(self, tool_string: str) -> Tuple[str, str, str]:
        """Parse tool string in format 'author/name[@version]'"""
        try:
            if '@' in tool_string:
                base, version = tool_string.split('@')
            else:
                base, version = tool_string, None
                
            if '/' in base:
                author, name = base.split('/')
            else:
                raise ValueError(f"Invalid tool string format: {tool_string}")
                
            return author, name, version
        except Exception as e:
            print(f"Error parsing tool string '{tool_string}': {str(e)}")
            raise

    def load_tool(self, author: str, name: str, version: str = None):
        """Load a tool from cache or download it"""
        try:
            actual_version = version if version else "latest"
            tool_key = f"{author}/{name}"
            
            if tool_key in self.loaded_tools:
                return self.loaded_tools[tool_key]
            
            author, name, version = self.download_tool(author, name, actual_version)
            return version
            
        except Exception as e:
            print(f"Error loading tool {author}/{name}: {str(e)}")
            raise

    def _get_cache_path(self, author: str, name: str, version: str) -> Path:
        """Get local cache path for tool"""
        if version is None:
            version = "latest"
        version_path = self._version_to_path(version)
        return self.cache_dir / author / name / f"{version_path}.tool"