from judgeval.data import CustomExample
from pydantic import field_validator

class QodoExample(CustomExample):
    code: str
    original_code: str
    
    def __init__(self, **data):
        super().__init__(**data)
        
    @field_validator('code', 'original_code', mode='before')
    @classmethod
    def validate_code(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError(f"Code must be a string or None but got {v} of type {type(v)}")
        return v
    
    def to_dict(self):
        return {
            "code": self.code,
            "original_code": self.original_code,
            **super().to_dict()
        }
    
    def model_dump(self, **kwargs):
        """
        Custom serialization that handles special cases for fields that might fail standard serialization.
        """
        data = super().model_dump(**kwargs)
        
        # Do any additional serialization here
        data["code"] = self.code
        data["original_code"] = self.original_code
        
        return data




