from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
from .communication import MemoryQuery, MemoryItem

@dataclass
class MemoryConfig:
    provider: str
    config: Dict[str, Any]

@dataclass
class MemoryLayer:
    memory_limit: int = 104857600  # 100MB
    eviction_k: int = 10
    custom_eviction_policy: Optional[str] = None
    agent_name: str = "default_agent"
    vector_store_config: Optional[MemoryConfig] = None
    llm_config: Optional[MemoryConfig] = None
    embedder_config: Optional[MemoryConfig] = None
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a new memory item through system call"""
        query = MemoryQuery(
            messages=[],  # Required by the model but not used for memory operations
            operation_type="add",
            text=text,
            metadata=metadata
        )
        
        # Use existing syscall infrastructure
        from aios.hooks.syscall import useSysCall
        syscall = useSysCall()
        result = syscall.memory(self.agent_name, query)
        return result["response"]
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories through system call"""
        memory_query = MemoryQuery(
            messages=[],
            operation_type="search",
            query=query,
            limit=limit
        )
        
        # Use existing syscall infrastructure
        from aios.hooks.syscall import useSysCall
        syscall = useSysCall()
        result = syscall.memory(self.agent_name, memory_query)
        return result["response"]
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item through system call"""
        memory_query = MemoryQuery(
            messages=[],
            operation_type="delete",
            memory_id=memory_id
        )
        
        # Use existing syscall infrastructure
        from aios.hooks.syscall import useSysCall
        syscall = useSysCall()
        result = syscall.memory(self.agent_name, memory_query)
        return result["response"]["status"] == "success"