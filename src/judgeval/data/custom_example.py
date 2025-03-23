from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
import json
import warnings

# Brainstorming what are the requirements for the fields?
class CustomExample(BaseModel):
    name: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    example_id: str = Field(default_factory=lambda: str(uuid4()))
    example_index: Optional[int] = None
    timestamp: Optional[str] = None
    trace_id: Optional[str] = None
    
    model_config = {
        "extra": "allow",  # Allow extra fields with any types
    }
    
    def __init__(self, **data):
        if 'example_id' not in data:
            data['example_id'] = str(uuid4())
        # Set timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)
        
    @field_validator('additional_metadata', mode='before')
    @classmethod
    def validate_additional_metadata(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError(f"Additional metadata must be a dictionary or None but got {v} of type {type(v)}")
        return v
    
    @field_validator('example_index', mode='before')
    @classmethod
    def validate_example_index(cls, v):
        if v is not None and not isinstance(v, int):
            raise ValueError(f"Example index must be an integer or None but got {v} of type {type(v)}")
        return v
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def validate_timestamp(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError(f"Timestamp must be a string or None but got {v} of type {type(v)}")
        return v
    
    @field_validator('trace_id', mode='before')
    @classmethod
    def validate_trace_id(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError(f"Trace ID must be a string or None but got {v} of type {type(v)}")
        return v
        
    def to_dict(self):
        return self.model_dump()
    
    def __str__(self):
        return str(self.model_dump())
    
    def model_dump(self, **kwargs):
        """
        Custom serialization that handles special cases for fields that might fail standard serialization.
        """
        data = super().model_dump(**kwargs)
        
        # Get all fields including custom ones
        all_fields = self.__dict__
        
        for field_name, value in all_fields.items():
            try:
                # Check if the field has its own serialization method
                if hasattr(value, 'to_dict'):
                    data[field_name] = value.to_dict()
                elif hasattr(value, 'model_dump'):
                    data[field_name] = value.model_dump()
                # Field is already in data from super().model_dump()
                elif field_name in data:
                    continue
                else:
                    # Try standard JSON serialization
                    json.dumps(value)
                    data[field_name] = value
            except (TypeError, OverflowError, ValueError):
                # Handle non-serializable objects
                try:
                    # Try converting to string
                    data[field_name] = str(value)
                except Exception as _:
                    # If all else fails, store as None and optionally warn
                    warnings.warn(f"Could not serialize field {field_name}, setting to None")
                    data[field_name] = None
        
        return data
    
    