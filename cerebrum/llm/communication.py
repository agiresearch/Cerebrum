from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from typing_extensions import Literal  

class Request(BaseModel):
    pass

class LLMQuery(Request):
    """
    Query class represents the input structure for performing various actions.
    
    Attributes:
        messages (List[Dict[str, Union[str, Any]]]): A list of dictionaries where each dictionary
            represents a message containing 'role' and 'content' or other key-value pairs.
        tools (Optional[List[Dict[str, Any]]]): An optional list of JSON-like objects (dictionaries) 
            representing tools and their parameters. Default is an empty list.
        action_type (Literal): A string that must be one of "message_llm", "call_tool", or "operate_file".
            This restricts the type of action the query performs.
        message_return_type (str): The type of the response message. Default is "text".
    """
    messages: List[Dict[str, Union[str, Any]]]  # List of message dictionaries, each containing role and content.
    tools: Optional[List[Dict[str, Any]]] = Field(default_factory=list)  # List of JSON-like objects (dictionaries) representing tools.
    action_type: Literal["chat", "tool_use", "operate_file"] = Field(default="chat")  # Restrict the action_type to specific choices.
    message_return_type: str = Field(default="text")  # Type of the return message, default is "text".

    class Config:
        arbitrary_types_allowed = True  # Allows the use of arbitrary types such as Any and Dict.

class Response(BaseModel):
    """
    Response class represents the output structure after performing actions.
    
    Attributes:
        response_message (Optional[str]): The generated response message.
        tool_calls (Optional[List[Dict[str, Any]]]): Tool calls made during processing.
        finished (bool): Whether the processing is complete.
        error (Optional[str]): Error message if any.
        status_code (int): HTTP status code of the response.
    """
    response_message: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finished: bool = False
    error: Optional[str] = None
    status_code: int = 200

    class Config:
        arbitrary_types_allowed = True