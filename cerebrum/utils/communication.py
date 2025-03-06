from pydantic import BaseModel, Field
from typing_extensions import Literal

import requests
from typing import Dict, Any, List

aios_kernel_url = "http://localhost:8000"

class Query(BaseModel):
    """
    Base class for all query types in the AIOS system.
    
    This class serves as the foundation for specialized query classes like LLMQuery,
    MemoryQuery, StorageQuery, and ToolQuery. It defines the minimum structure required
    for a valid query within the AIOS ecosystem.
    
    Attributes:
        query_class: Identifier for the query type, must be one of 
                     ["llm", "memory", "storage", "tool"]
    """
    query_class: Literal["llm", "memory", "storage", "tool"]
    
class Response(BaseModel):
    """
    Base class for all response types in the AIOS system.
    
    This class serves as the foundation for specialized response classes like LLMResponse,
    MemoryResponse, StorageResponse, and ToolResponse. It defines the minimum structure
    required for a valid response within the AIOS ecosystem.
    
    Attributes:
        response_class: Identifier for the response type, must be one of 
                        ["llm", "memory", "storage", "tool"]
    """
    response_class: Literal["llm", "memory", "storage", "tool"]

def post(base_url: str, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a POST request to the specified API endpoint.
    
    Args:
        base_url: Base URL of the API server
        endpoint: API endpoint path
        data: JSON-serializable dictionary to be sent in the request body
        
    Returns:
        Parsed JSON response as a dictionary
        
    Raises:
        requests.exceptions.HTTPError: If the request fails
    """
    response = requests.post(f"{base_url}{endpoint}", json=data)
    response.raise_for_status()
    return response.json()

def send_request(agent_name: str, query: Query, base_url: str = aios_kernel_url):
    """
    Send a query to the AIOS kernel on behalf of an agent.
    
    Args:
        agent_name: Name of the agent making the request
        query: Query object containing the request details
        base_url: Base URL of the AIOS kernel
        
    Returns:
        Parsed JSON response from the AIOS kernel
    """
    query_type = query.query_class
    result = post(base_url, "/query", {
        'query_type': query_type,
        'agent_name': agent_name,
        'query_data': query.model_dump()})

    return result