from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from typing_extensions import Literal

from cerebrum.utils.communication import Query, Response, send_request, aios_kernel_url

class LLMQuery(Query):
    """
    Query class for LLM operations.
    
    This class represents the input structure for performing various LLM actions
    such as chatting, using tools, or operating on files.
    
    Attributes:
        query_class: Identifier for LLM queries, always set to "llm"
        llms: Optional list of LLM configurations with format:
            [
                {
                    "name": str,           # Name of the LLM (e.g., "gpt-4")
                    "temperature": float,  # Sampling temperature (0.0-2.0)
                    "max_tokens": int,     # Maximum tokens to generate
                    "top_p": float,        # Nucleus sampling parameter (0.0-1.0)
                    "frequency_penalty": float,  # Frequency penalty (-2.0-2.0)
                    "presence_penalty": float    # Presence penalty (-2.0-2.0)
                }
            ]
        messages: List of message dictionaries with format:
            [
                {
                    "role": str,      # One of ["system", "user", "assistant"]
                    "content": str,   # The message content
                    "name": str,      # Optional name for the message sender
                    "function_call": dict,  # Optional function call details
                    "tool_calls": list     # Optional tool call details
                }
            ]
        tools: Optional list of available tools with format:
            [
                {
                    "name": str,        # Tool identifier
                    "description": str, # Tool description
                    "parameters": {     # Tool parameters schema
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string"},
                            "param2": {"type": "number"}
                        },
                        "required": ["param1"]
                    }
                }
            ]
        action_type: Type of action to perform, one of:
            - "chat": Simple conversation
            - "tool_use": Using external tools
            - "operate_file": File operations
        message_return_type: Desired format of the response
        response_format: Specific JSON format of the response, e.g., 
            {
                "type": "json_schema", 
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "context": {
                                "type": "string"
                            },
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["keywords", "context", "tags"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }
    
    Examples:
        ```python
        # Simple chat query
        query = LLMQuery(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is Python?"
                }
            ],
            action_type="chat"
        )
        
        # Tool use query with specific LLM configuration
        query = LLMQuery(
            llms=[{
                "name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 500
            }],
            messages=[
                {
                    "role": "user",
                    "content": "Calculate 2 + 2"
                }
            ],
            tools=[{
                "name": "calculator",
                "description": "Performs basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"]
                        },
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    },
                    "required": ["operation", "numbers"]
                }
            }],
            action_type="tool_use"
        )
        ```
    """
    query_class: str = "llm"
    llms: Optional[List[Dict[str, Any]]] = Field(default=None)
    messages: List[Dict[str, Union[str, Any]]]
    tools: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    action_type: Literal["chat", "tool_use", "operate_file"] = Field(default="chat")
    message_return_type: Literal["text", "json"] = Field(default="text")

    class Config:
        arbitrary_types_allowed = True

class LLMResponse(Response):
    """
    Response class for LLM operations.
    
    This class represents the output structure after performing LLM actions.
    
    Attributes:
        response_class: Identifier for LLM responses, always "llm"
        response_message: Generated response text
        tool_calls: List of tool calls made during processing, format:
            [
                {
                    "name": str,        # Tool name
                    "parameters": dict,  # Parameters used
                    "result": Any       # Tool execution result
                }
            ]
        finished: Whether processing completed successfully
        error: Error message if any
        status_code: HTTP status code
        
    Examples:
        ```python
        # Successful chat response
        response = LLMResponse(
            response_message="Python is a high-level programming language...",
            finished=True,
            status_code=200
        )
        
        # Tool use response with calculator
        response = LLMResponse(
            response_message=None,
            tool_calls=[{
                "name": "calculator",
                "parameters": {
                    "operation": "add",
                    "numbers": [2, 2]
                },
                "result": 4
            }],
            finished=True,
            status_code=200
        )
        ```
    """
    response_class: str = "llm"
    response_message: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finished: bool = False
    error: Optional[str] = None
    status_code: int = 200

    class Config:
        arbitrary_types_allowed = True

def llm_chat(
        agent_name: str, 
        messages: List[Dict[str, Any]], 
        base_url: str = aios_kernel_url,
        llms: List[Dict[str, Any]] = None
    ) -> LLMResponse:
    """
    Perform a chat interaction with the LLM.
    
    Args:
        agent_name: Name of the agent making the request
        messages: List of message dictionaries with format:
            [
                {
                    "role": "system"|"user"|"assistant",
                    "content": str,
                    "name": str  # Optional
                }
            ]
        base_url: API base URL
        llms: Optional list of LLM configurations
        
    Returns:
        LLMResponse containing the generated response
        
    Examples:
        ```python
        response = llm_chat(
            "agent1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Explain quantum computing."
                }
            ],
            llms=[{
                "name": "gpt-4",
                "temperature": 0.7
            }]
        )
        ```
    """
    query = LLMQuery(
        llms=llms,
        messages=messages,
        tools=None,
        action_type="chat"
    )
    return send_request(agent_name, query, base_url)

def llm_chat_with_json_output(
        agent_name: str, 
        messages: List[Dict[str, Any]], 
        base_url: str = aios_kernel_url,
        llms: List[Dict[str, Any]] = None,
        response_format: Dict[str, Dict] = None
    ) -> LLMResponse:
    """
    Perform a chat interaction with the LLM and return a JSON-formatted output.
    
    Args:
        agent_name: Name of the agent making the request
        messages: List of message dictionaries with format:
            [
                {
                    "role": "system"|"user"|"assistant",
                    "content": str,
                    "name": str  # Optional
                }
            ]
        base_url: API base URL
        llms: Optional list of LLM configurations
        response_format: JSON schema specifying the required output format
        
    Returns:
        LLMResponse containing the generated JSON response
        
    Examples:
        ```python
        response = llm_chat_with_json_output(
            "agent1",
            messages=[
                {
                    "role": "system",
                    "content": "Extract keywords from the user query."
                },
                {
                    "role": "user",
                    "content": "Analyze the impact of climate change on biodiversity."
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "keywords",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            }
        )
        ```
    """
    query = LLMQuery(
        llms=llms,
        messages=messages,
        tools=None,
        message_return_type="json",
        action_type="chat"
        # response_format=response_format
    )
    return send_request(agent_name, query, base_url)

def llm_call_tool(
        agent_name: str, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]], 
        base_url: str = aios_kernel_url,
        llms: List[Dict[str, Any]] = None
    ) -> LLMResponse:
    """
    Use LLM to call tools based on user input.
    
    Args:
        agent_name: Name of the agent making the request
        messages: List of message dictionaries with format:
            [
                {
                    "role": "system"|"user"|"assistant",
                    "content": str,
                    "name": str,  # Optional
                    "tool_calls": [  # Optional, for assistant messages
                        {
                            "tool": str,  # Tool name
                            "parameters": dict  # Tool parameters
                        }
                    ]
                }
            ]
        tools: List of available tools with format:
            [
                {
                    "name": str,  # Tool identifier
                    "description": str,  # Tool description
                    "parameters": {  # JSON Schema for parameters
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                }
            ]
        base_url: API base URL
        llms: Optional list of LLM configurations with format:
            [
                {
                    "name": str,  # e.g., "gpt-4"
                    "temperature": float,  # 0.0-2.0
                    "max_tokens": int
                }
            ]
        
    Returns:
        LLMResponse containing tool calls and results
        
    Examples:
        ```python
        # Calculator tool example
        response = llm_call_tool(
            "agent1",
            messages=[{
                "role": "user",
                "content": "What is 15 * 7?"
            }],
            tools=[{
                "name": "calculator",
                "description": "Performs basic arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["multiply"]
                        },
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    }
                }
            }]
        )
        
        # Weather and summary example
        response = llm_call_tool(
            "agent1",
            messages=[{
                "role": "user",
                "content": "What's the weather like and give me a summary?"
            }],
            tools=[
                {
                    "name": "weather_api",
                    "description": "Gets current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                },
                {
                    "name": "text_summarizer",
                    "description": "Summarizes text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "max_length": {"type": "integer"}
                        }
                    }
                }
            ],
            llms=[{
                "name": "gpt-4",
                "temperature": 0.7
            }]
        )
        ```
    """
    query = LLMQuery(
        llms=llms,
        messages=messages,
        tools=tools,
        action_type="tool_use"
    )
    return send_request(agent_name, query, base_url)

def llm_operate_file(
        agent_name: str, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]], 
        base_url: str = aios_kernel_url,
        llms: List[Dict[str, Any]] = None
    ) -> LLMResponse:
    """
    Use LLM to perform file operations.
    
    Args:
        agent_name: Name of the agent making the request
        messages: List of message dictionaries with format:
            [
                {
                    "role": "system"|"user"|"assistant",
                    "content": str,
                    "name": str,  # Optional
                    "file_operations": [  # Optional, for assistant messages
                        {
                            "operation": str,  # e.g., "write", "modify"
                            "file_path": str,
                            "content": str
                        }
                    ]
                }
            ]
        tools: List of file operation tools with format:
            [
                {
                    "name": str,  # e.g., "file_writer", "code_modifier"
                    "description": str,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "file_path": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }
            ]
        base_url: API base URL
        llms: Optional list of LLM configurations
        
    Returns:
        LLMResponse containing file operation results
        
    Examples:
        ```python
        # Create a Python script
        response = llm_operate_file(
            "agent1",
            messages=[{
                "role": "user",
                "content": "Create a script that sorts a list of numbers"
            }],
            tools=[{
                "name": "file_writer",
                "description": "Creates or modifies files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                        "format": {"type": "string", "enum": ["python", "text"]}
                    }
                }
            }]
        )
        
        # Modify existing code with error handling
        response = llm_operate_file(
            "agent1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code improvement assistant."
                },
                {
                    "role": "user",
                    "content": "Add error handling to the sort function in sort.py"
                }
            ],
            tools=[{
                "name": "code_modifier",
                "description": "Modifies existing code files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "modifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["add", "modify", "delete"]},
                                    "location": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }],
            llms=[{
                "name": "gpt-4",
                "temperature": 0.3  # Lower temperature for code modifications
            }]
        )
        ```
    """
    query = LLMQuery(
        llms=llms,
        messages=messages,
        tools=tools,
        action_type="operate_file"
    )
    return send_request(agent_name, query, base_url)