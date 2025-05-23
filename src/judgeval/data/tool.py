from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional
import warnings

class Tool(BaseModel):
    tool_name: str
    parameters: Optional[Dict[str, Any]] = None
    agent_name: Optional[str] = None
    
    @field_validator('tool_name')
    def validate_tool_name(cls, v):
        if not v:
            warnings.warn("Tool name is empty or None", UserWarning)
        return v
    
    @field_validator('parameters')
    def validate_parameters(cls, v):
        if v is not None and not isinstance(v, dict):
            warnings.warn(f"Parameters should be a dictionary, got {type(v)}", UserWarning)
        return v
    
    @field_validator('agent_name')
    def validate_agent_name(cls, v):
        if v is not None and not isinstance(v, str):
            warnings.warn(f"Agent name should be a string, got {type(v)}", UserWarning)
        return v