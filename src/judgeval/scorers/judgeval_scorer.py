"""
Judgeval Scorer class

Enables client to create custom scorers that do not fall under any of the ready-made Judgment scorers.
To create a custom scorer, extend this class and implement the `score_example`, `a_score_example`, and `success_check` methods.
"""

from typing import Optional, Dict, Union, List
from abc import abstractmethod

from judgeval.common.logger import debug, info, warning, error
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.constants import UNBOUNDED_SCORERS

class JudgevalScorer:
    """
    Base class for scorers in `judgeval`.

    In practice, you should not implement this class unless you are creating a custom scorer.
    Judgeval offers 10+ default scorers that you can use out of the box.
    
    If you want to create a scorer that does not fall under any of the ready-made Judgment scorers,
    you can create a custom scorer by extending this class.
    """
    score_type: str  # name of your new scorer
    threshold: float  # The threshold to pass a test while using this scorer as a scorer
    score: Optional[float] = None  # The float score of the scorer run on the test case
    score_breakdown: Dict = None
    reason: Optional[str] = None  # The reason for the score when evaluating the test case
    success: Optional[bool] = None  # Whether the test case passed or failed
    evaluation_model: Optional[str] = None  # The model used to evaluate the test case
    strict_mode: bool = False  # Whether to run the scorer in strict mode
    async_mode: bool = True  # Whether to run the scorer in async mode
    verbose_mode: bool = True  # Whether to run the scorer in verbose mode
    include_reason: bool = False  # Whether to include the reason in the output
    custom_example: bool = False  # Whether the scorer corresponds to CustomExamples
    error: Optional[str] = None  # The error message if the scorer failed
    evaluation_cost: Optional[float] = None  # The cost of running the scorer
    verbose_logs: Optional[str] = None  # The verbose logs of the scorer
    additional_metadata: Optional[Dict] = None  # Additional metadata for the scorer
    error: Optional[str] = None
    success: Optional[bool] = None

    def __init__(
        self, 
        score_type: str, 
        threshold: float, 
        score: Optional[float] = None, 
        score_breakdown: Optional[Dict] = None, 
        reason: Optional[str] = None, 
        success: Optional[bool] = None, 
        evaluation_model: Optional[str] = None, 
        strict_mode: bool = False, 
        async_mode: bool = True, 
        verbose_mode: bool = True, 
        include_reason: bool = False, 
        custom_example: bool = False,
        error: Optional[str] = None, 
        evaluation_cost: Optional[float] = None, 
        verbose_logs: Optional[str] = None, 
        additional_metadata: Optional[Dict] = None
        ):
            debug(f"Initializing JudgevalScorer with score_type={score_type}, threshold={threshold}")
            if score_type in UNBOUNDED_SCORERS:
                if threshold < 0:
                    raise ValueError(f"Threshold for {score_type} must be greater than 0, got: {threshold}")
            else:
                if not 0 <= threshold <= 1:
                    raise ValueError(f"Threshold for {score_type} must be between 0 and 1, got: {threshold}")
            if strict_mode:
                warning("Strict mode enabled - scoring will be more rigorous")
            info(f"JudgevalScorer initialized with evaluation_model: {evaluation_model}")
            self.score_type = score_type
            self.threshold = threshold
            self.score = score
            self.score_breakdown = score_breakdown
            self.reason = reason
            self.success = success
            self.evaluation_model = evaluation_model
            self.strict_mode = strict_mode
            self.async_mode = async_mode
            self.verbose_mode = verbose_mode
            self.include_reason = include_reason
            self.custom_example = custom_example
            self.error = error
            self.evaluation_cost = evaluation_cost
            self.verbose_logs = verbose_logs
            self.additional_metadata = additional_metadata

    def _add_model(self, model: Optional[Union[str, List[str], JudgevalJudge]] = None):
        """
        Adds the evaluation model to the JudgevalScorer instance 

        This method is used at eval time
        """
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

    @abstractmethod
    def score_example(self, example, *args, **kwargs) -> float:
        """
        Measures the score on a single example
        """
        warning("Attempting to call unimplemented score_example method")
        error("score_example method not implemented")
        raise NotImplementedError("You must implement the `score` method in your custom scorer")

    @abstractmethod
    async def a_score_example(self, example, *args, **kwargs) -> float:
        """
        Asynchronously measures the score on a single example
        """
        warning("Attempting to call unimplemented a_score_example method")
        error("a_score_example method not implemented")
        raise NotImplementedError("You must implement the `a_score` method in your custom scorer") 
    
    @abstractmethod
    def _success_check(self) -> bool:
        """
        For unit testing, determines whether the test case passes or fails
        """
        warning("Attempting to call unimplemented success_check method")
        error("_success_check method not implemented")
        raise NotImplementedError("You must implement the `_success_check` method in your custom scorer")

    def __str__(self):
        debug("Converting JudgevalScorer instance to string representation")
        if self.error:
            warning(f"JudgevalScorer contains error: {self.error}")
        info(f"JudgevalScorer status - success: {self.success}, score: {self.score}")
        attributes = {
            "score_type": self.score_type,
            "threshold": self.threshold,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "reason": self.reason,
            "success": self.success,
            "evaluation_model": self.evaluation_model,
            "strict_mode": self.strict_mode,
            "async_mode": self.async_mode,
            "verbose_mode": self.verbose_mode,
            "include_reason": self.include_reason,
            "error": self.error,
            "evaluation_cost": self.evaluation_cost,
            "verbose_logs": self.verbose_logs,
            "additional_metadata": self.additional_metadata,
        }
        return f"JudgevalScorer({attributes})"
    
    def to_dict(self):
        return {
            "score_type": str(self.score_type),  # Convert enum to string for serialization
            "threshold": self.threshold
        }
