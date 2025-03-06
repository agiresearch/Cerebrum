from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

from cerebrum.utils.communication import Query, send_request, Response, aios_kernel_url

class StorageQuery(Query):
    """
    Query class for storage operations.
    
    Attributes:
        query_class: Identifier for storage queries
        params: Dictionary containing operation parameters
        operation_type: Type of storage operation to perform
        
    Example:
        ```python
        # Create a query to write a file
        query = StorageQuery(
            params={"file_path": "test.txt", "content": "Hello"},
            operation_type="write"
        )
        
        # Create a query to search files
        query = StorageQuery(
            params={"query_text": "python code", "n": 5},
            operation_type="retrieve"
        )
        ```
    """
    query_class: str = "storage"
    params: Dict[str, Union[str, Any]]
    operation_type: str = Field(default="text")

    class Config:
        arbitrary_types_allowed = True

class StorageResponse(Response):
    """
    Response class for storage operations.
    
    Attributes:
        response_class: Identifier for storage responses
        response_message: Operation result message
        finished: Whether operation completed
        error: Error message if any
        status_code: HTTP status code
        
    Example:
        ```python
        # Successful file write response
        response = StorageResponse(
            response_message="File written successfully",
            finished=True,
            status_code=200
        )
        
        # Error response
        response = StorageResponse(
            response_message="File not found",
            finished=True,
            error="FileNotFoundError",
            status_code=404
        )
        ```
    """
    response_class: str = "storage"
    response_message: Optional[str] = None
    finished: bool = False
    error: Optional[str] = None
    status_code: int = 200

# Storage APIs
def mount(
        agent_name: str, 
        root_dir: str,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Mount a storage directory for an agent.
    
    Args:
        agent_name: Name of the agent
        root_dir: Directory path to mount
        base_url: API base URL
        
    Returns:
        StorageResponse object
        
    Example:
        ```python
        # Mount a storage directory
        response = mount("agent1", "/data/storage")
        if response.finished:
            print("Storage mounted successfully")
        else:
            print(f"Mount failed: {response.error}")
        ```
    """
    query = StorageQuery(
        params={"root_dir": root_dir},
        operation_type="mount"
    )
    return send_request(agent_name, query, base_url)

def retrieve_file(
        agent_name: str, 
        query_text: str,
        n: int,
        keywords: List[str] = None,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Retrieve files matching query criteria.
    
    Args:
        agent_name: Name of the agent
        query_text: Search query text
        n: Number of results to return
        keywords: Optional list of keywords
        base_url: API base URL
        
    Returns:
        StorageResponse with matching files
        
    Example:
        ```python
        # Search for Python files
        response = retrieve(
            "agent1",
            "python code",
            n=5,
            keywords=["function", "class"]
        )
        
        if response.finished:
            files = response.response_message
            for file in files:
                print(f"Found file: {file}")
        ```
    """
    params = {
        "query_text": query_text,
        "n": n,
        "keywords": keywords
    }
    query = StorageQuery(
        params=params,
        operation_type="retrieve"
    )
    return send_request(agent_name, query, base_url)
    
def create_file(
        agent_name: str, 
        file_path: str,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Create a new empty file.
    
    Args:
        agent_name: Name of the agent
        file_path: Path where to create the file
        base_url: API base URL
        
    Returns:
        StorageResponse object
        
    Example:
        ```python
        # Create a new Python file
        response = create_file("agent1", "src/new_module.py")
        if response.finished:
            print(f"File created at {file_path}")
        else:
            print(f"Failed to create file: {response.error}")
        ```
    """
    query = StorageQuery(
        params={"file_path": file_path},
        operation_type="create_file"
    )
    return send_request(agent_name, query, base_url)

def create_dir(
        agent_name: str, 
        dir_path: str,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Create a new directory.
    
    Args:
        agent_name: Name of the agent
        dir_path: Path where to create the directory
        base_url: API base URL
        
    Returns:
        StorageResponse object
        
    Example:
        ```python
        # Create a new project directory
        response = create_dir("agent1", "projects/new_project")
        if response.finished:
            print(f"Directory created at {dir_path}")
        else:
            print(f"Failed to create directory: {response.error}")
        ```
    """
    query = StorageQuery(
        params={"dir_path": dir_path},
        operation_type="create_dir"
    )
    return send_request(agent_name, query, base_url)

def write_file(
        agent_name: str, 
        file_path: str,
        content: str,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Write content to a file.
    
    Args:
        agent_name: Name of the agent
        file_path: Path to the file
        content: Content to write
        base_url: API base URL
        
    Returns:
        StorageResponse object
        
    Example:
        ```python
        # Write a Python function to a file
        content = '''
        def greet(name):
            return f"Hello, {name}!"
        '''
        response = write_file("agent1", "src/greetings.py", content)
        if response.finished:
            print("Function written successfully")
        ```
    """
    query = StorageQuery(
        params={"file_path": file_path, "content": content},
        operation_type="write"
    )
    return send_request(agent_name, query, base_url)

def rollback_file(
        agent_name: str, 
        file_path: str,
        n: int,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Roll back a file to a previous version.
    
    Args:
        agent_name: Name of the agent
        file_path: Path to the file
        n: Number of versions to roll back
        base_url: API base URL
        
    Returns:
        StorageResponse object
        
    Example:
        ```python
        # Roll back to previous version
        response = roll_back("agent1", "src/module.py", n=1)
        if response.finished:
            print("File rolled back successfully")
        else:
            print(f"Rollback failed: {response.error}")
        ```
    """
    query = StorageQuery(
        params={"file_path": file_path, "n": n},
        operation_type="rollback"
    )
    return send_request(agent_name, query, base_url)

def share_file(
        agent_name: str, 
        file_path: str,
        base_url: str = aios_kernel_url
    ) -> StorageResponse:
    """
    Share a file with other agents.
    
    Args:
        agent_name: Name of the agent sharing the file
        file_path: Path to the file to share
        base_url: API base URL
        
    Returns:
        StorageResponse object
        
    Example:
        ```python
        # Share a configuration file
        response = share_file("agent1", "config/shared_settings.json")
        if response.finished:
            print(f"File {file_path} is now shared")
        else:
            print(f"Failed to share file: {response.error}")
        ```
    """
    query = StorageQuery(
        params={"file_path": file_path},
        operation_type="share"
    )
    return send_request(agent_name, query, base_url)
