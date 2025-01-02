import os
import yaml
from typing import Any, Dict
from pathlib import Path

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._config = {}
        self._load_default_config()
        self._load_environment_variables()
    
    def _load_default_config(self):
        """Load default configuration from YAML file"""
        config_path = Path(__file__).parent / "default.yaml"
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load default config: {e}")
            self._config = {}
    
    def _load_environment_variables(self):
        """Override configuration with environment variables"""
        env_mappings = {
            'CEREBRUM_KERNEL_URL': ('kernel', 'base_url'),
            'CEREBRUM_KERNEL_TIMEOUT': ('kernel', 'timeout'),
            'CEREBRUM_MEMORY_LIMIT': ('client', 'memory_limit'),
            'CEREBRUM_MAX_WORKERS': ('client', 'max_workers'),
            'CEREBRUM_ROOT_DIR': ('client', 'root_dir'),
            'CEREBRUM_AGENT_HUB_URL': ('manager', 'agent_hub_url'),
            'CEREBRUM_TOOL_HUB_URL': ('manager', 'tool_hub_url'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested_value(self._config, config_path, value)
    
    def _set_nested_value(self, d: Dict, path: tuple, value: Any):
        """Set a value in nested dictionary using a path tuple"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        current = self._config
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
            if current is None:
                return default
        return current

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if '.' in key:
                path = tuple(key.split('.'))
                self._set_nested_value(self._config, path, value)
            else:
                self._config[key] = value

config = ConfigManager() 