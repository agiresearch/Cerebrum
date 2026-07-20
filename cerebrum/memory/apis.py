"""Memory API module for AIOS kernel memory operations.

Provides functions to create, read, update, delete, and search agent memories
through the AIOS kernel.

Kernel-Side Memory Configuration:
    The following fields are configured in the kernel's ``config.yaml`` under
    the ``memory`` section. They are **not** configurable through the SDK.

    memory.provider : str
        Memory backend to use. Accepts ``"in-house"``, ``"mem0"``, or ``"zep"``.
    memory.auto_extract : bool
        When true, the kernel automatically stores conversation turns as
        memories after each chat LLM call.
    memory.auto_inject : bool
        When true, the kernel retrieves and injects relevant memories before
        each chat LLM call.
    memory.relevance_threshold : float
        Minimum similarity score a memory must meet to be eligible for
        injection.
    memory.max_injected_memories : int
        Maximum number of memories injected per LLM call.
    memory.max_memory_tokens : int
        Token budget for the injected memory block.

    ``memory.mem0.*`` and ``memory.zep.*`` contain provider-specific kernel
    configuration and are not set from the SDK.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from typing_extensions import Literal

from cerebrum.utils.communication import Query, Response, send_request
from cerebrum.config.config_manager import config

aios_kernel_url = config.get_kernel_url()

class MemoryQuery(Query):
    """
    Query class for memory operations.
    
    Attributes:
        action_type (Literal): Type of memory operation to perform
        params (Dict): Parameters specific to each action type
    """
    query_class: str = "memory"
    operation_type: Literal["add_memory", "get_memory", "update_memory", "remove_memory", "retrieve_memory", "add_agentic_memory","retrieve_memory_raw"]
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
    # status_code: int = 200

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
        metadata: Optional metadata (keywords, context, tags, etc.).
            Provider-specific keys can be passed through this dict.
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse containing the created memory ID
        
    Example:
        >>> # Create a memory with content and metadata
        >>> metadata = {"tags": ["important", "meeting"], "priority": "high"}
        >>> response = create_memory("agent1", "Meeting notes: Discussed Q1 goals", metadata)
        >>> print(response.memory_id)  # "mem_123abc"
        >>> print(response.success)    # True

    Provider-Specific Metadata Keys:
        mem0:
            ``user_id`` (str): Scopes memory to a specific user.
            ``agent_id`` (str): Scopes memory to a specific agent.
            Falls back to kernel config defaults if not provided.
        zep:
            ``session_id`` (str): Scopes memory to a session.
            ``user_id`` (str): Scopes memory to a user.
            Falls back to kernel config defaults if not provided.
        in-house:
            No provider-specific metadata keys required.
    """
    query = MemoryQuery(
        operation_type="add_memory",
        params={"content": content, "metadata": metadata or {}}
    )
    return send_request(agent_name, query, base_url)

def create_agentic_memory(agent_name: str, 
                 content: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 base_url: str = aios_kernel_url) -> MemoryResponse:
    """Create a new agentic memory note.
    
    Args:
        agent_name: Name of the agent to handle the request
        content: Content of the memory
        metadata: Optional metadata (keywords, context, tags, etc.).
            Provider-specific keys can be passed through this dict.
        base_url: Base URL for the API server
        
    Returns:
        MemoryResponse containing the created memory ID
        
    Example:
        >>> # Create a memory with content and metadata
        >>> metadata = {"tags": ["important", "meeting"], "priority": "high"}
        >>> response = create_agentic_memory("agent1", "Meeting notes: Discussed Q1 goals", metadata)
        >>> print(response.memory_id)  # "mem_123abc"
        >>> print(response.success)    # True

    Provider-Specific Metadata Keys:
        mem0:
            ``user_id`` (str): Scopes memory to a specific user.
            ``agent_id`` (str): Scopes memory to a specific agent.
            Falls back to kernel config defaults if not provided.
        zep:
            ``session_id`` (str): Scopes memory to a session.
            ``user_id`` (str): Scopes memory to a user.
            Falls back to kernel config defaults if not provided.
        in-house:
            No provider-specific metadata keys required.
    """
    query = MemoryQuery(
        operation_type="add_agentic_memory",
        params={"content": content, "metadata": metadata or {}}
    )
    return send_request(agent_name, query, base_url)

def get_memory(agent_name: str, 
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
        operation_type="get_memory",
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
        metadata: Optional new metadata.
            Provider-specific keys can be passed through this dict.
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

    Provider-Specific Metadata Keys:
        mem0:
            ``user_id`` (str): Scopes memory to a specific user.
            ``agent_id`` (str): Scopes memory to a specific agent.
            Falls back to kernel config defaults if not provided.
        zep:
            ``session_id`` (str): Scopes memory to a session.
            ``user_id`` (str): Scopes memory to a user.
            Falls back to kernel config defaults if not provided.
        in-house:
            No provider-specific metadata keys required.
    """
    params = {"memory_id": memory_id}
    if metadata is not None:
        params["metadata"] = metadata
    if content is not None:
        params["content"] = content
    else:
        params["content"] = None
    query = MemoryQuery(
        operation_type="update_memory",
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
        operation_type="remove_memory",
        params={"memory_id": memory_id}
    )
    return send_request(agent_name, query, base_url)

def search_memories(agent_name: str,
                   query: str,
                   k: int = 5,
                   base_url: str = aios_kernel_url,
                   *,
                   user_id: Optional[str] = None,
                   sharing_policy: Optional[str] = None) -> MemoryResponse:
    """Search for memories using a hybrid retrieval approach.
    
    Args:
        agent_name: Name of the agent to handle the request
        query: Search query text
        k: Maximum number of results to return
        base_url: Base URL for the API server
        user_id: Optional user ID for cross-agent memory retrieval.
            When provided, the kernel searches across all agents' memories
            scoped to this user instead of restricting to ``agent_name``.
            Must be a non-empty string or ``None`` (default).
        sharing_policy: Optional sharing policy filter. Accepted values
            are ``"shared"``, ``"private"``, or ``None`` (default).
            When provided, the kernel filters results to memories whose
            metadata contains the matching ``sharing_policy`` value.
        
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

    Kernel Contract:
        The kernel interprets ``user_id`` and ``sharing_policy`` in the
        ``params`` dict to determine the search scope:

        Neither ``user_id`` nor ``sharing_policy``:
            Default agent-scoped search. The kernel restricts results to
            memories owned by ``agent_name`` (existing behavior).
        ``user_id`` only:
            Search across all agents' memories scoped to that user. The
            kernel bypasses the agent-name scope and returns memories
            matching the given ``user_id``.
        ``sharing_policy`` only:
            Search within the default agent scope, but filter results to
            memories whose metadata ``sharing_policy`` matches the
            provided value.
        Both ``user_id`` and ``sharing_policy``:
            Cross-agent search. The kernel bypasses the agent-name scope
            and returns memories matching both the ``user_id`` AND the
            ``sharing_policy`` metadata filter.

    Provider-Specific Metadata Keys:
        These keys can be used to scope search results when passed via
        the ``metadata`` parameter of memory creation functions.

        mem0:
            ``user_id`` (str): Scopes memory to a specific user.
            ``agent_id`` (str): Scopes memory to a specific agent.
            Falls back to kernel config defaults if not provided.
        zep:
            ``session_id`` (str): Scopes memory to a session.
            ``user_id`` (str): Scopes memory to a user.
            Falls back to kernel config defaults if not provided.
        in-house:
            No provider-specific metadata keys required.
    """
    if sharing_policy is not None and sharing_policy not in ("shared", "private"):
        raise ValueError(
            f"sharing_policy must be 'shared', 'private', or None; got {sharing_policy!r}"
        )
    if user_id is not None and not user_id.strip():
        raise ValueError("user_id must be a non-empty string or None")

    # Normalize: strip whitespace from explicit user_id
    resolved_user_id = str(user_id).strip() if user_id is not None else None

    params: Dict[str, Any] = {"content": query, "k": k}
    if resolved_user_id is not None:
        params["user_id"] = resolved_user_id
    if sharing_policy is not None:
        params["sharing_policy"] = sharing_policy

    query_obj = MemoryQuery(
        operation_type="retrieve_memory",
        params=params,
    )
    return send_request(agent_name, query_obj, base_url)