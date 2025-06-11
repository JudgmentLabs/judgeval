import pytest
from judgeval.judges.utils import create_judge, InvalidJudgeModelError
from judgeval.judges import LiteLLMJudge, MixtureOfJudges, TogetherJudge
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_model_lists():
    with patch("judgeval.judges.utils.LITELLM_SUPPORTED_MODELS", ["gpt-4"]):
        with patch(
            "judgeval.judges.utils.TOGETHER_SUPPORTED_MODELS",
            [
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
            ],
        ):
            yield


def test_create_judge_invalid_type():
    # Test invalid type (int)
    with pytest.raises(
        InvalidJudgeModelError,
        match="Model must be a string, list of strings, or a judgeval judge object",
    ):
        create_judge(model=42)


def test_create_judge_invalid_model_name():
    # Test invalid model name string
    with pytest.raises(
        InvalidJudgeModelError, match="Invalid judge model chosen: invalid-model"
    ):
        create_judge(model="invalid-model")


def test_create_judge_invalid_model_in_list():
    # Test list containing an invalid model name
    with pytest.raises(
        InvalidJudgeModelError, match="Invalid judge model chosen: invalid-model"
    ):
        create_judge(model=["gpt-4", "invalid-model"])


def test_create_judge_default_none():
    # Test default case when model=None
    judge, is_native = create_judge(model=None)
    assert isinstance(judge, LiteLLMJudge)
    assert judge.model == "gpt-4.1"
    assert is_native is True


def test_create_judge_existing_judge():
    # Test passing an existing judge object
    existing_judge = LiteLLMJudge(model="gpt-4")
    judge, is_native = create_judge(model=existing_judge)
    assert judge is existing_judge  # Should return the same instance
    assert is_native is True


def test_create_judge_valid_model_list():
    # Test creating a MixtureOfJudges with valid models
    models = [
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ]  # Assuming these are in LITELLM_SUPPORTED_MODELS
    judge, is_native = create_judge(model=models)
    assert isinstance(judge, MixtureOfJudges)
    assert judge.models == models
    assert is_native is True


def test_create_judge_litellm_model():
    # Test creating a LiteLLMJudge with a valid LiteLLM model
    judge, is_native = create_judge(
        model="gpt-4"
    )  # Assuming gpt-4 is in LITELLM_SUPPORTED_MODELS
    assert isinstance(judge, LiteLLMJudge)
    assert judge.model == "gpt-4"
    assert is_native is True


def test_create_judge_together_model():
    # Test creating a TogetherJudge with a valid Together model
    judge, is_native = create_judge(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    )  # Assuming this is in TOGETHER_SUPPORTED_MODELS
    assert isinstance(judge, TogetherJudge)
    assert judge.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert is_native is True
