from typing import Optional, Union, List
from pydantic import BaseModel

from judgeval.constants import APIScorer
from judgeval.scorers import JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    scorer_progress_meter,
    create_verbose_logs,
    parse_response_json,
    check_example_params
)
from .prompts import ComparisonTemplate

required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.EXPECTED_OUTPUT,
]

class ComparisonDifference(BaseModel):
    actual_output_sentence: str
    expected_output_sentence: str
    reason: str

class ComparisonDifferences(BaseModel):
    differences: List[ComparisonDifference]

class ComparisonScorer(JudgevalScorer):
    def __init__(
        self,
        criteria: str,
        description: str,
        threshold: float = 1,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        super().__init__(
            score_type=APIScorer.COMPARISON,
            threshold=threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            verbose_mode=verbose_mode
        )
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()
        self.criteria = criteria
        self.description = description

    def score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)
        
        with scorer_progress_meter(self, display_meter=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(
                        example, 
                        _show_indicator=False
                    )
                )
            else:
                self.differences = self._find_differences(example)
                self.score = len(self.differences)
                self.reason = str(self.differences)
                self.success = self.score <= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                    
                return len(self.differences)

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(
            self, async_mode=True, display_meter=_show_indicator
        ):
            self.differences = self.a_find_differences(example)
            self.score = len(self.differences)
            self.reason = str(self.differences)
            self.success = self.score <= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    def _find_differences(self, example: Example) -> float:
        prompt = ComparisonTemplate.find_differences(
            criteria=self.criteria,
            description=self.description,
            actual_output=example.actual_output,
            expected_output=example.expected_output
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["differences"]
        else:
            try:
                res: ComparisonDifferences = self.model.generate(prompt, schema=ComparisonDifferences)
                return res.differences
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["differences"]

    async def a_find_differences(self, example: Example) -> float:
        prompt = ComparisonTemplate.find_differences(
            criteria=self.criteria,
            description=self.description,
            actual_output=example.actual_output,
            expected_output=example.expected_output
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["differences"]
        else:
            try:
                res: ComparisonDifferences = await self.model.a_generate(prompt, schema=ComparisonDifferences)
                return res.differences
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["differences"]

    def _success_check(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score <= self.threshold
            except:
                self.success = False
        return self.success
    
    @property
    def __name__(self):
        return f"Comparison - {self.criteria}"