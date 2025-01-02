from cerebrum.client import Cerebrum
from cerebrum.config.config_manager import config

class Config:
    def __init__(self):
        self._global_client: Cerebrum = None
        self._base_url = config.get('kernel', 'base_url')
        self._timeout = config.get('kernel', 'timeout', default=30)
   
    @property
    def global_client(self):
        if not self._global_client:
            raise ValueError("Client not set. Call config.client = Cerebrum Client")
        return self._global_client
       
    @global_client.setter 
    def global_client(self, value):
        self._global_client = value

    def configure(self, **kwargs):
        """Configure multiple settings at once"""
        config.update(**kwargs)
        for key, value in kwargs.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)

config = Config()
