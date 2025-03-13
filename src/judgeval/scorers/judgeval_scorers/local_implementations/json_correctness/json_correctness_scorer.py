from typing import Optional, Union, Any
from pydantic import BaseModel, ValidationError, create_model

from judgeval.constants import APIScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    scorer_progress_meter,
    create_verbose_logs,
    check_example_params
)
from judgeval.scorers import JudgevalScorer
from judgeval.data import Example, ExampleParams


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT
]


class JsonCorrectnessScorer(JudgevalScorer):

    def __init__(
        self,
        json_schema: Union[BaseModel, dict],
        model: Optional[Union[str, JudgevalJudge]] = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        user: Optional[str] = None
    ):
        super().__init__(
            score_type=APIScorer.JSON_CORRECTNESS,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=False,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.user = user
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

        if isinstance(json_schema, dict):
            # Convert to BaseModel
            fields = {
                key: (str if prop["type"] == "string" else Any, ...)
                for key, prop in json_schema["properties"].items()
            }

            # Dynamically create the model
            DynamicModel = create_model(json_schema["title"], **fields)

            self.json_schema = DynamicModel
        else:
            self.json_schema = json_schema

    def score_example(self, example: Example, _show_indicator: bool = True) -> float:
        check_example_params(example, required_params, self)
        with scorer_progress_meter(
            self,
            async_mode=self.async_mode,
            display_meter=_show_indicator,
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(example, _show_indicator=False)
                )
            else:
                valid_json = True
                try:
                    self.json_schema.model_validate_json(
                        example.actual_output
                    )
                except ValidationError as e:
                    valid_json = False

                self.score = 1.0 if valid_json else 0
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"LLM outputed Json:\n{example.actual_output}",
                        f"Score: {self.score}",
                    ],
                )

                return self.score

    async def a_score_example(self, example: Example, _show_indicator: bool = True) -> float:
        check_example_params(example, required_params, self)
        with scorer_progress_meter(
            self,
            async_mode=self.async_mode,
            display_meter=_show_indicator,
        ):
            valid_json = True
            try:
                self.json_schema.model_validate_json(
                    example.actual_output
                )
            except ValidationError as e:
                valid_json = False

            self.score = 1.0 if valid_json else 0
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"LLM outputed Json:\n{example.actual_output}",
                    f"Score: {self.score}",
                ],
            )
            return self.score

    def _success_check(self):
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "JSON Correctness"
    