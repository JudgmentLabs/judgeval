"""
`judgeval` JSON correctness scorer

TODO add link to docs page for this scorer

"""


# External imports
from pydantic import BaseModel, Field
# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from judgeval.data import ExampleParams

class JSONCorrectnessScorer(APIJudgmentScorer):
    json_schema: BaseModel = Field(None, exclude=True)
    
    def __init__(self, threshold: float, json_schema: BaseModel):
        super().__init__(
            threshold=threshold, 
            score_type=APIScorer.JSON_CORRECTNESS,
            required_params=[
                ExampleParams.INPUT,
                ExampleParams.ACTUAL_OUTPUT,
            ]
        )
        object.__setattr__(self, 'json_schema', json_schema)
    
    def to_dict(self):
        base_dict = super().to_dict()  # Get the parent class's dictionary
        base_dict["kwargs"] = {
            "json_schema": self.json_schema.model_json_schema()
        }
        return base_dict

    @property
    def __name__(self):
        return "JSON Correctness"
