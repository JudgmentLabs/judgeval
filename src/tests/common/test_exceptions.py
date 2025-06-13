import pytest
from judgeval.common.exceptions import (
    MissingTestCaseParamsError,
    JudgmentAPIError,
    InvalidJudgeModelError,
)


def test_missing_test_case_params_error():
    """Test that MissingTestCaseParamsError can be raised"""
    with pytest.raises(MissingTestCaseParamsError):
        raise MissingTestCaseParamsError()


def test_judgment_api_error():
    """Test JudgmentAPIError message handling"""
    error_message = "API connection failed"
    try:
        raise JudgmentAPIError(error_message)
    except JudgmentAPIError as e:
        assert str(e) == error_message
        assert e.message == error_message


def test_invalid_judge_model_error():
    """Test InvalidJudgeModelError message handling"""
    error_message = "Invalid model: gpt-5"
    try:
        raise InvalidJudgeModelError(error_message)
    except InvalidJudgeModelError as e:
        assert str(e) == error_message
        assert e.message == error_message
