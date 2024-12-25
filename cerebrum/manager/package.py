import json
from pathlib import Path
import zipfile
import os
from typing import Dict

class AgentPackage:
    def __init__(self, path: Path):
        self.path = path
        self.metadata = {}
        self.files = {}

    def load(self):
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            self.metadata = json.loads(zip_ref.read('metadata.json').decode('utf-8'))
            for file_info in zip_ref.infolist():
                if file_info.filename != 'metadata.json':
                    self.files[file_info.filename] = zip_ref.read(file_info.filename)

    def save(self):
        directory = os.path.dirname(self.path)

        os.makedirs(directory, exist_ok=True)

        with zipfile.ZipFile(self.path, 'w') as zip_ref:
            zip_ref.writestr('metadata.json', json.dumps(self.metadata))
            for filename, content in self.files.items():
                zip_ref.writestr(filename, content)

    def get_entry_point(self):
        return self.metadata.get('entry', 'agent.py')

    def get_module_name(self):
        return self.metadata.get('module', 'Agent')
    
    def get_config(self) -> Dict:
        if "config.json" not in self.files:
            raise FileNotFoundError("Config file not found in package")
        
        config_content = self.files["config.json"]
        config = json.loads(config_content.decode('utf-8'))
        
        # Maintain the integrity of the original configuration
        result = config.copy()
        
        # Extract required fields from meta and build, but do not override original fields
        meta_info = config.get("meta", {})
        build_info = config.get("build", {})
        
        # Only add top-level fields, do not override existing fields
        if "author" not in result:
            result["author"] = meta_info.get("author")
        if "version" not in result:
            result["version"] = meta_info.get("version")
        if "license" not in result:
            result["license"] = meta_info.get("license")
        if "entry" not in result:
            result["entry"] = build_info.get("entry")
        if "module" not in result:
            result["module"] = build_info.get("module")
            
        return result

class ToolPackage:
    def __init__(self, path: Path):
        self.path = path
        self.metadata = {}
        self.files = {}

    def load(self):
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            self.metadata = json.loads(zip_ref.read('metadata.json').decode('utf-8'))
            for file_info in zip_ref.infolist():
                if file_info.filename != 'metadata.json':
                    self.files[file_info.filename] = zip_ref.read(file_info.filename)

    def save(self):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        
        with zipfile.ZipFile(self.path, 'w') as zip_ref:
            zip_ref.writestr('metadata.json', json.dumps(self.metadata))
            for filename, content in self.files.items():
                zip_ref.writestr(filename, content)

    def get_entry_point(self) -> str:
        return self.metadata.get('entry', 'tool.py')

    def get_module_name(self) -> str:
        return self.metadata.get('module', 'Tool')
    
    def get_config(self) -> dict:
        return self.metadata
