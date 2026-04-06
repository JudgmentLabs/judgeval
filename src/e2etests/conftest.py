import os
import pytest
import random
import string
from typing import Callable
from dotenv import load_dotenv

from judgeval import Judgeval
from judgeval.judges import Judge, BinaryResponse
from e2etests.utils import delete_project, create_project
from judgeval.data.example import Example

ScorerFactory = Callable[[str], Judge[BinaryResponse]]

load_dotenv()

API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("JUDGMENT_ORG_ID")

if not API_KEY:
    pytest.skip("JUDGMENT_API_KEY not set", allow_module_level=True)


@pytest.fixture(scope="session")
def project_name():
    return "e2e-tests-" + "".join(
        random.choices(string.ascii_letters + string.digits, k=12)
    )


@pytest.fixture(scope="session")
def client(project_name: str):
    if not API_KEY or not ORGANIZATION_ID:
        pytest.skip(
            "JUDGMENT_API_KEY or JUDGMENT_ORG_ID not set", allow_module_level=True
        )
    create_project(project_name=project_name)
    client = Judgeval(project_name=project_name)
    yield client
    delete_project(project_name=project_name)


@pytest.fixture
def random_name() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=12))


@pytest.fixture
def local_scorer() -> ScorerFactory:
    def _make(prompt: str) -> Judge[BinaryResponse]:
        class LLMScorer(Judge[BinaryResponse]):
            async def score(self, data: Example) -> BinaryResponse:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                response = await client.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Input: {data['input']}\n"
                                f"Output: {data['actual_output']}"
                            ),
                        },
                    ],
                    response_format=BinaryResponse,
                )
                result = response.choices[0].message.parsed
                return result if result else BinaryResponse(value=False, reason="Error")

        return LLMScorer()

    return _make
