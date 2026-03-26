def get_binary_scorer_template(scorer_name: str) -> str:
    return f"""
from judgeval.judges import Judge
from judgeval.hosted.responses import BinaryResponse
from judgeval.data.example import Example

class {scorer_name}(Judge[BinaryResponse]):
    async def score(self, data: Example) -> BinaryResponse:
        return BinaryResponse(value=True, reason="Test")
"""


def get_categorical_scorer_template(scorer_name: str) -> str:
    return f"""
from judgeval.judges import Judge
from judgeval.hosted.responses import CategoricalResponse
from judgeval.data.example import Example
from judgeval.hosted.responses import Category

class {scorer_name}Response(CategoricalResponse):
    categories = [
        Category(value="Category 1", description="Description for Category 1"),
        Category(value="Category 2", description="Description for Category 2"),
    ]

class {scorer_name}(Judge[{scorer_name}Response]):
    async def score(self, data: Example) -> {scorer_name}Response:
        return {scorer_name}Response(value="Category 1", reason="Test")
"""


def get_numeric_scorer_template(scorer_name: str) -> str:
    return f"""
from judgeval.judges import Judge
from judgeval.hosted.responses import NumericResponse
from judgeval.data.example import Example

class {scorer_name}(Judge[NumericResponse]):
    async def score(self, data: Example) -> NumericResponse:
        return NumericResponse(value=1.0, reason="Test")
"""
