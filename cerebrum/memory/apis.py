from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from typing_extensions import Literal

from cerebrum.utils.communication import Query, Response, send_request, aios_kernel_url

class MemoryQuery(Query):
    """
    Query class for memory operations.
    
    Attributes:
        action_type (Literal): Type of memory operation to perform
        params (Dict): Parameters specific to each action type
    """
    query_class: str = "memory"
    action_type: Literal["create", "read", "update", "delete", "search"]
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class MemoryResponse(Response):
    """
    Response class for memory operations.
    
    Attributes:
        memory_id (Optional[str]): The ID of the created/updated memory
        content (Optional[str]): The content of the memory for read operations
        metadata (Optional[Dict]): Memory metadata (keywords, context, etc.)
        search_results (Optional[List]): List of search results
        success (bool): Whether the operation was successful
        error (Optional[str]): Error message if any
        status_code (int): HTTP status code of the response
    """
    response_class: str = "memory"
    memory_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    success: bool = False
    error: Optional[str] = None
    status_code: int = 200

    class Config:
        arbitrary_types_allowed = True

def create_memory(agent_name: str, 
                 content: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 base_url: str = aios_kernel_url) -> MemoryResponse:
    """Create a new memory note.
    
    Args:
        agent_name: Name of the agent to handle the request
        content: Content of the memory
        metadata: Optional metadata (keywords, context, tags, etc.)
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse containing the created memory ID
        
    Example:
        >>> # Create a memory with content and metadata
        >>> metadata = {"tags": ["important", "meeting"], "priority": "high"}
        >>> response = create_memory("agent1", "Meeting notes: Discussed Q1 goals", metadata)
        >>> print(response.memory_id)  # "mem_123abc"
        >>> print(response.success)    # True
    """
    query = MemoryQuery(
        action_type="create",
        params={"content": content, "metadata": metadata or {}}
    )
    return send_request(agent_name, query, base_url)

def read_memory(agent_name: str, 
                memory_id: str,
                base_url: str = aios_kernel_url) -> MemoryResponse:
    """Read a memory note by ID.
    
    Args:
        agent_name: Name of the agent to handle the request
        memory_id: ID of the memory to read
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse containing the memory content and metadata
        
    Example:
        >>> # Read a memory by its ID
        >>> response = read_memory("agent1", "mem_123abc")
        >>> print(response.content)    # "Meeting notes: Discussed Q1 goals"
        >>> print(response.metadata)   # {"tags": ["important", "meeting"], "priority": "high"}
    """
    query = MemoryQuery(
        action_type="read",
        params={"memory_id": memory_id}
    )
    return send_request(agent_name, query, base_url)

def update_memory(agent_name: str,
                 memory_id: str,
                 content: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 base_url: str = aios_kernel_url) -> MemoryResponse:
    """Update an existing memory note.
    
    Args:
        agent_name: Name of the agent to handle the request
        memory_id: ID of the memory to update
        content: Optional new content
        metadata: Optional new metadata
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse indicating success/failure
        
    Example:
        >>> # Update memory content and add a new tag
        >>> new_metadata = {"tags": ["important", "meeting", "updated"], "priority": "high"}
        >>> response = update_memory(
        ...     "agent1",
        ...     "mem_123abc",
        ...     content="Updated meeting notes: Added action items",
        ...     metadata=new_metadata
        ... )
        >>> print(response.success)    # True
    """
    params = {"memory_id": memory_id}
    if content is not None:
        params["content"] = content
    if metadata is not None:
        params["metadata"] = metadata
        
    query = MemoryQuery(
        action_type="update",
        params=params
    )
    return send_request(agent_name, query, base_url)

def delete_memory(agent_name: str,
                 memory_id: str,
                 base_url: str = aios_kernel_url) -> MemoryResponse:
    """Delete a memory note.
    
    Args:
        agent_name: Name of the agent to handle the request
        memory_id: ID of the memory to delete
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse indicating success/failure
        
    Example:
        >>> # Delete a memory by its ID
        >>> response = delete_memory("agent1", "mem_123abc")
        >>> print(response.success)    # True
    """
    query = MemoryQuery(
        action_type="delete",
        params={"memory_id": memory_id}
    )
    return send_request(agent_name, query, base_url)

def search_memories(agent_name: str,
                   query: str,
                   k: int = 5,
                   base_url: str = aios_kernel_url) -> MemoryResponse:
    """Search for memories using a hybrid retrieval approach.
    
    Args:
        agent_name: Name of the agent to handle the request
        query: Search query text
        k: Maximum number of results to return
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse containing search results
        
    Example:
        >>> # Search for memories about meetings
        >>> response = search_memories("agent1", "meeting goals", limit=2)
        >>> for result in response.search_results:
        ...     print(f"Memory ID: {result['memory_id']}")
        ...     print(f"Content: {result['content']}")
        ...     print(f"Score: {result['score']}")
        # Memory ID: mem_123abc
        # Content: Meeting notes: Discussed Q1 goals
        # Score: 0.92
    """
    query = MemoryQuery(
        action_type="search",
        params={"query": query, "k": k}
    )
    return send_request(agent_name, query, base_url)