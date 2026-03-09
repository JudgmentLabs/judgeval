def get_binary_scorer_template(scorer_name: str) -> str:
    return f"""
from judgeval.v1.judges import Judge
from judgeval.v1.hosted.responses import BinaryResponse
from judgeval.v1.data.example import Example

class {scorer_name}(Judge[BinaryResponse]):
    async def score(self, data: Example) -> BinaryResponse:
        return BinaryResponse(value=True, reason="Test")
"""


def get_categorical_scorer_template(scorer_name: str) -> str:
    return f"""
from judgeval.v1.judges import Judge
from judgeval.v1.hosted.responses import CategoricalResponse
from judgeval.v1.data.example import Example

class {scorer_name}(Judge[CategoricalResponse]):
    async def score(self, data: Example) -> CategoricalResponse:
        return CategoricalResponse(value="Test", reason="Test")
"""


def get_numeric_scorer_template(scorer_name: str) -> str:
    return f"""
from judgeval.v1.judges import Judge
from judgeval.v1.hosted.responses import NumericResponse
from judgeval.v1.data.example import Example

class {scorer_name}(Judge[NumericResponse]):
    async def score(self, data: Example) -> NumericResponse:
        return NumericResponse(value=1.0, reason="Test")
"""
