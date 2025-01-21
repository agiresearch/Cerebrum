import importlib
import os
import json
import base64
import subprocess
import sys
from typing import List, Dict
import requests
from pathlib import Path
import platformdirs
import importlib.util
import uuid
import traceback
import logging
import re

from cerebrum.manager.package import AgentPackage
from cerebrum.utils.manager import get_newest_version

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = Path(platformdirs.user_cache_dir("cerebrum"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    def _version_to_path(self, version: str) -> str:
        return version.replace('.', '-')

    def _path_to_version(self, path_version: str) -> str:
        return path_version.replace('-', '.')

    def package_agent(self, folder_path: str) -> Dict:
        try:
            logger.debug(f"\n{'='*50}")
            logger.debug(f"Packaging agent from folder: {folder_path}")
            logger.debug(f"Folder exists: {os.path.exists(folder_path)}")
            logger.debug(f"Folder contents: {os.listdir(folder_path)}")
            
            agent_files = self._get_agent_files(folder_path)
            logger.debug(f"\nCollected agent files: {list(agent_files.keys())}")
            
            config_path = os.path.join(folder_path, "config.json")
            logger.debug(f"\nLooking for config at: {config_path}")
            logger.debug(f"Config exists: {os.path.exists(config_path)}")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    metadata = json.load(f)
            else:
                logger.error(f"Config file not found at {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            logger.debug(f"\nMetadata loaded: {json.dumps(metadata, indent=2)}")
            
            # Modify here: Extract only the required fields from meta and build, instead of overwriting the entire configuration
            meta_info = metadata.get("meta", {})
            build_info = metadata.get("build", {})
            result = metadata.copy()  # Keep the complete configuration, including the tools field
            
            # Only update specific fields at the top level
            top_level_updates = {
                "author": meta_info.get('author'),
                "name": metadata.get("name"),
                "version": meta_info.get('version'),
                "license": meta_info.get('license'),
                "entry": build_info.get('entry'),
                "module": build_info.get('module')
            }
            result.update(top_level_updates)
            
            # Add file content
            files_list = []
            for name, content in agent_files.items():
                files_list.append({
                    "path": name,
                    "content": base64.b64encode(content).decode('utf-8')
                })
            result["files"] = files_list
            
            return result
            
        except Exception as e:
            logger.error(f"\n{'='*50}")
            logger.error(f"Failed to package agent")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            logger.error(f"{'='*50}\n")
            raise

    def upload_agent(self, payload: Dict):
        response = requests.post(f"{self.base_url}/cerebrum/upload", json=payload)
        response.raise_for_status()
        print(f"Agent {payload.get('author')}/{payload.get('name')} (v{payload.get('version')}) uploaded successfully.")

    def download_agent(self, author: str, name: str, version: str | None = None) -> tuple[str, str, str]:
        if version is None:
            cached_versions = self._get_cached_versions(author, name)
            version = get_newest_version(cached_versions)

        try:
            cache_path = self._get_cache_path(author, name, version)
        except:
            cache_path = None

        if cache_path is not None and cache_path.exists():
            print(f"Using cached version of {author}/{name} (v{version})")
            return author, name, version

        if version is None:
            params = {
                "author": author,
                "name": name,
            }
        else: 
            params = {
                "author": author,
                "name": name,
                "version": version
            }
        
        response = requests.get(f"{self.base_url}/cerebrum/download", params=params)
        response.raise_for_status()
        agent_data = response.json()

        actual_version = agent_data.get('version', version)
        cache_path = self._get_cache_path(author, name, actual_version)

        self._save_agent_to_cache(agent_data, cache_path)
        print(
            f"Agent {author}/{name} (v{actual_version}) downloaded and cached successfully.")

        if not self.check_reqs_installed(cache_path):
            self.install_agent_reqs(cache_path)

        return author, name, actual_version

    def _get_cached_versions(self, author: str, name: str) -> List[str]:
        agent_dir = self.cache_dir / author / name
        if agent_dir.exists():
            return [self._path_to_version(v.stem) for v in agent_dir.glob("*.agent") if v.is_file()]
        return []

    def _get_cache_path(self, author: str, name: str, version: str) -> Path:
        return self.cache_dir / author / name / f"{self._version_to_path(version)}.agent"
    
    def _get_random_cache_path(self):
        """
        Creates a randomly named folder inside the cache/cerebrum directory and returns its path.
        Uses platformdirs for correct cross-platform cache directory handling.
        """
        # Get the user cache directory using platformdirs
        cache_dir = platformdirs.user_cache_dir(appname="cerebrum")
        
        # Generate a random UUID for the folder name
        random_name = str(uuid.uuid4())
        
        # Create the full path
        random_folder_path = os.path.join(cache_dir, random_name)
        
        # Create the directory and any necessary parent directories
        os.makedirs(random_folder_path, exist_ok=True)
        
        return Path(random_folder_path) / f"local.agent"


    def _save_agent_to_cache(self, agent_data: Dict, cache_path: Path):
        agent_package = AgentPackage(cache_path)
        agent_package.metadata = {
            "author": agent_data["author"],
            "name": agent_data["name"],
            "version": agent_data["version"],
            "license": agent_data["license"],
            "entry": agent_data["entry"],
            "module": agent_data["module"]
        }
        agent_package.files = {file["path"]: base64.b64decode(file["content"]) for file in agent_data["files"]}
        agent_package.save()

        # Ensure the cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saved agent to cache: {cache_path}")

    def _get_agent_files(self, folder_path: str) -> Dict[str, bytes]:
        try:
            logger.debug(f"\nCollecting files from {folder_path}")
            files = {}
            for root, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    if filename.endswith(('.py', '.txt', '.json')):
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, folder_path)
                        logger.debug(f"Processing file: {relative_path}")
                        
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            files[relative_path] = content
                            logger.debug(f"Added file: {relative_path} ({len(content)} bytes)")
            
            logger.debug(f"Collected {len(files)} files: {list(files.keys())}")
            return files
        except Exception as e:
            logger.error(f"Error collecting files from {folder_path}: {str(e)}")
            raise

    def _get_agent_metadata(self, folder_path: str) -> Dict:
        try:
            config_path = os.path.join(folder_path, "config.json")
            logger.debug(f"\nReading metadata from {config_path}")
            
            if not os.path.exists(config_path):
                logger.error(f"Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                content = f.read()
                logger.debug(f"Config content:\n{content}")
                metadata = json.loads(content)
                logger.debug(f"Parsed metadata: {json.dumps(metadata, indent=2)}")
                return metadata
        except Exception as e:
            logger.error(f"Error reading metadata from {config_path}: {str(e)}")
            raise

    def list_available_agents(self) -> List[Dict[str, str]]:
        response = requests.get(f"{self.base_url}/cerebrum/get_all_agents")
        response.raise_for_status()

        response: dict = response.json()

        agent_list = []

        for v in list(response.values())[:-1]:
            agent_list.append({
                "agent": "/".join([v["author"], v["name"], v['version']])
            })

        return agent_list

    def check_agent_updates(self, author: str, name: str, current_version: str) -> bool:
        response = requests.get(f"{self.base_url}/cerebrum/check_updates", params={
            "author": author,
            "name": name,
            "current_version": current_version
        })
        response.raise_for_status()
        return response.json()["update_available"]

    def check_reqs_installed(self, agent_path: Path) -> bool:
        agent_package = AgentPackage(agent_path)
        agent_package.load()
        reqs_content = agent_package.files.get("meta_requirements.txt")
        if not reqs_content:
            return True  # No requirements file, consider it as installed

        try:
            result = subprocess.run(
                ['conda', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            result = subprocess.run(
                ['pip', 'list', '--format=freeze'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        reqs = [line.strip().split("==")[0] for line in reqs_content.decode('utf-8').splitlines() if line.strip() and not line.startswith("#")]

        output = result.stdout.decode('utf-8')
        installed_packages = [line.split()[0]
                              for line in output.splitlines() if line]

        return all(req in installed_packages for req in reqs)

    def install_agent_reqs(self, agent_path: Path):
        agent_package = AgentPackage(agent_path)
        agent_package.load()
        reqs_content = agent_package.files.get("meta_requirements.txt")
        if not reqs_content:
            print("No meta_requirements.txt found. Skipping dependency installation.")
            return

        temp_reqs_path = self.cache_dir / "temp_requirements.txt"
        with open(temp_reqs_path, "wb") as f:
            f.write(reqs_content)

        log_path = agent_path.with_suffix('.log')

        print(f"Installing dependencies for agent. Writing to {log_path}")

        with open(log_path, "a") as f:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(temp_reqs_path)
            ], stdout=f, stderr=f)

        temp_reqs_path.unlink()  # Remove temporary requirements file

    def _check_and_install_dependencies(self, agent_path: str) -> None:
        """Check and install dependencies for agent"""
        # First check meta_requirements.txt
        req_file = os.path.join(agent_path, "meta_requirements.txt")
        
        if not os.path.exists(req_file):
            # For built-in agents, also check requirements.txt
            req_file = os.path.join(agent_path, "requirements.txt")
        
        if os.path.exists(req_file):
            logger.info(f"Installing dependencies from {req_file}")
            try:
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    req_file,
                    "--quiet"
                ])
                logger.info("Dependencies installed successfully")
            except Exception as e:
                logger.error(f"Failed to install dependencies: {str(e)}")
                raise
        else:
            logger.warning(f"No requirements file found at {req_file}")

    def is_builtin_agent(self, name: str) -> bool:
        """Check if an agent is a built-in example agent"""
        builtin_paths = [
            os.path.join(os.path.dirname(self.base_path), "cerebrum", "example", "agents", name),
            os.path.join(os.path.dirname(self.base_path), "example", "agents", name),
        ]
        return any(os.path.exists(path) for path in builtin_paths)

    def _get_builtin_agent_path(self, name: str) -> str:
        """Get the path for a built-in agent"""
        # Try both possible paths for built-in agents
        possible_paths = [
            os.path.join(os.path.dirname(self.base_path), "cerebrum", "example", "agents", name),
            os.path.join(os.path.dirname(self.base_path), "example", "agents", name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
            
        raise FileNotFoundError(f"Built-in agent '{name}' not found in any of the expected paths: {possible_paths}")

    def load_agent(self, author: str = '', name: str = '', version: str | None = None,
                   local: bool = False, path: str | None = None):
        try:
            logger.debug(f"\n{'='*50}")
            logger.debug("Starting agent loading process")
            logger.debug(f"Parameters: author={author}, name={name}, version={version}")
            logger.debug(f"local={local}, path={path}")
            
            if local:
                # For local agents, check and install dependencies first
                self._check_and_install_dependencies(path)
                
                # Handle relative paths
                if not os.path.isabs(path):
                    # Try multiple possible base paths
                    possible_paths = [
                        path,  # Original path
                        os.path.join(os.getcwd(), path),  # Relative to current working directory
                        os.path.join(os.path.dirname(self.base_path), path),  # Relative to base_path
                        os.path.join(os.path.dirname(self.base_path), "cerebrum", path),  # Relative to cerebrum directory
                    ]
                    
                    logger.debug("Trying possible paths:")
                    for try_path in possible_paths:
                        logger.debug(f"Checking path: {try_path}")
                        logger.debug(f"Path exists: {os.path.exists(try_path)}")
                        if os.path.exists(try_path):
                            path = try_path
                            logger.debug(f"Found valid path: {path}")
                            break
                    else:
                        logger.error("No valid path found. Tried:")
                        for try_path in possible_paths:
                            logger.error(f"  - {try_path}")
                        raise FileNotFoundError(f"Could not find agent at any of the attempted paths")
                
                logger.debug(f"Final resolved path: {path}")
                
                # Package and save local agent
                logger.debug("\nPackaging local agent")
                logger.debug(f"Agent path: {path}")
                logger.debug(f"Path exists: {os.path.exists(path)}")
                logger.debug(f"Path contents: {os.listdir(path) if os.path.exists(path) else 'PATH NOT FOUND'}")
                
                local_agent_data = self.package_agent(path)
                logger.debug(f"Local agent data: {json.dumps(local_agent_data, indent=2)}")
                
                random_path = self._get_random_cache_path()
                logger.debug(f"Generated cache path: {random_path}")
                
                self._save_agent_to_cache(local_agent_data, random_path)
                agent_path = f"{random_path}"
                logger.debug(f"Saved agent to cache: {agent_path}")
                
            elif self.is_builtin_agent(name):
                # Handle built-in agent
                path = self._get_builtin_agent_path(name)
                logger.info(f"Loading built-in agent from {path}")
                self._check_and_install_dependencies(path)
                
                # Package and save built-in agent
                logger.debug("\nPackaging built-in agent")
                local_agent_data = self.package_agent(path)
                random_path = self._get_random_cache_path()
                self._save_agent_to_cache(local_agent_data, random_path)
                agent_path = f"{random_path}"
                
            else:
                # Handle remote agent
                if version is None:
                    cached_versions = self._get_cached_versions(author, name)
                    version = get_newest_version(cached_versions)

                agent_path = self._get_cache_path(author, name, version)
                
                if not agent_path.exists():
                    print(f"Agent {author}/{name} (v{version}) not found in cache. Downloading...")
                    self.download_agent(author, name, version)

            # Common loading code for all agent types
            logger.debug(f"\nLoading agent package from: {agent_path}")
            agent_package = AgentPackage(agent_path)
            agent_package.load()

            entry_point = agent_package.get_entry_point()
            module_name = agent_package.get_module_name()
            logger.debug(f"Entry point: {entry_point}")
            logger.debug(f"Module name: {module_name}")
            logger.debug(f"Package files: {list(agent_package.files.keys())}")

            temp_dir = self.cache_dir / "temp" / f"{author}_{name}_{version}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"\nCreated temp directory: {temp_dir}")
            logger.debug(f"Temp dir exists: {temp_dir.exists()}")

            # Extract files and print content
            for filename, content in agent_package.files.items():
                file_path = temp_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(content)
                if filename.endswith('.py'):
                    logger.debug(f"\nFile content for {filename}:")
                    try:
                        logger.debug(f"{file_path.read_text()}")
                    except Exception as e:
                        logger.error(f"Error reading {filename}: {e}")

            if str(temp_dir) not in sys.path:
                sys.path.insert(0, str(temp_dir))
            logger.debug(f"\nPython path: {sys.path}")

            module_path = temp_dir / entry_point
            logger.debug(f"\nLoading module from: {module_path}")
            logger.debug(f"Module path exists: {module_path.exists()}")
            
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if spec is None:
                raise ImportError(f"Failed to create module spec for {module_name}")
                
            module = importlib.util.module_from_spec(spec)
            logger.debug(f"Created module object: {module}")
            logger.debug(f"Module dict before exec: {dir(module)}")
            
            if spec.loader is None:
                raise ImportError(f"Module spec has no loader for {module_name}")
                
            spec.loader.exec_module(module)
            logger.debug(f"Module dict after exec: {dir(module)}")
            logger.debug(f"Available module attributes: {[attr for attr in dir(module) if not attr.startswith('__')]}")

            try:
                logger.debug(f"\nAttempting to get class from module {module.__name__}")
                # Prefer to use the build.class configuration
                class_name = agent_package.get_config().get("build", {}).get("class")
                if not class_name:  # If no class is specified, use the module name
                    class_name = module_name
                
                logger.debug(f"Looking for class: {class_name}")
                agent_class = getattr(module, class_name)
                logger.debug(f"Successfully got agent class: {agent_class}")
                logger.debug(f"Agent class type: {type(agent_class)}")
            except AttributeError as e:
                logger.error(f"\nFailed to get {class_name} from module {module.__name__}")
                logger.error(f"Module contents: {dir(module)}")
                logger.error(f"Module file path: {module.__file__}")
                if hasattr(module, '__all__'):
                    logger.error(f"Module __all__: {module.__all__}")
                raise AttributeError(
                    f"Module '{module.__name__}' has no attribute '{class_name}'. "
                    f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('__')]}"
                ) from e

            config = agent_package.get_config()
            logger.debug(f"\nAgent config: {json.dumps(config, indent=2)}")
            logger.debug(f"{'='*50}\n")
            
            return agent_class, config
            
        except Exception as e:
            logger.error(f"\n{'='*50}")
            logger.error("Failed to load agent")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"{'='*50}\n")
            raise


if __name__ == '__main__':
    manager = AgentManager('https://app.aios.foundation/')
    agent = manager.download_agent('example', 'academic_agent')
    print(agent)