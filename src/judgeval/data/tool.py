from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional, List
import warnings

class Tool(BaseModel):
    tool_name: str
    parameters: Optional[Dict[str, Any]] = None
    agent_name: Optional[str] = None
    result_dependencies: Optional[List[Dict[str, Any]]] = None
    action_dependencies: Optional[List[Dict[str, Any]]] = None
    require_all: Optional[bool] = None
    
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
    
    @field_validator('result_dependencies')
    def validate_result_dependencies(cls, v):
        if v is not None and not isinstance(v, list):
            warnings.warn(f"Result dependencies should be a list, got {type(v)}", UserWarning)
        return v
    
    @field_validator('action_dependencies')
    def validate_action_dependencies(cls, v):
        if v is not None and not isinstance(v, list):
            warnings.warn(f"Action dependencies should be a list, got {type(v)}", UserWarning)
        return v

    @field_validator('require_all')
    def validate_require_all(cls, v):
        if v is not None and not isinstance(v, bool):
            warnings.warn(f"Require all should be a boolean, got {type(v)}", UserWarning)
        return v