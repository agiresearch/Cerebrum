from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

class MemoryItem(BaseModel):
    id: str
    memory: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

class MemoryQuery(BaseModel):
    messages: List[Dict[str, Union[str, Any]]]
    operation_type: str  # One of: "add", "search", "delete"
    text: Optional[str] = None  # For add operation
    query: Optional[str] = None  # For search operation
    memory_id: Optional[str] = None  # For delete operation
    limit: Optional[int] = Field(default=5)  # For search operation
    metadata: Optional[Dict[str, Any]] = None  # For add operation

    class Config:
        arbitrary_types_allowed = True
