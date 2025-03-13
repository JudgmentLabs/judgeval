from typing import Optional, List, Union, Tuple

from judgeval.constants import APIScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers import JudgevalScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    parse_response_json,
    scorer_progress_meter,
    create_verbose_logs,
    check_example_params,
)
from .prompts import (
    ACVerdict,
    AnswerCorrectnessTemplate,
    Statements,
    Verdicts,
    Reason,
)


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.EXPECTED_OUTPUT,
]


class AnswerCorrectnessScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False
    ):
        super().__init__(
            score_type=APIScorer.ANSWER_CORRECTNESS,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

    async def _a_get_statements(self, expected_output: str) -> List[str]:
        prompt = AnswerCorrectnessTemplate.deduce_statements(
            expected_output=expected_output,
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["statements"]
        else:
            try:
                res: Statements = await self.model.a_generate(
                    prompt, schema=Statements
                )
                return res.statements
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["statements"]

    def _get_statements(self, expected_output: str) -> List[str]:
        prompt = AnswerCorrectnessTemplate.deduce_statements(
            expected_output=expected_output,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["statements"]
        else:
            try:
                res: Statements = self.model.generate(
                    prompt, schema=Statements
                )
                return res.statements
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["statements"]

    async def _a_get_verdicts(self, actual_output: str) -> List[ACVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerCorrectnessTemplate.generate_verdicts(
            actual_output=actual_output,
            statements=self.statements,
        )

        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return [ACVerdict(**item) for item in data["verdicts"]]
        else:
            try:
                res: Verdicts = await self.model.a_generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return [ACVerdict(**item) for item in data["verdicts"]]

    def _get_verdicts(self, actual_output: str) -> List[ACVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerCorrectnessTemplate.generate_verdicts(
            actual_output=actual_output,
            statements=self.statements,
        )

        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return [ACVerdict(**item) for item in data["verdicts"]]
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return [ACVerdict(**item) for item in data["verdicts"]]

    async def _a_get_reason(self) -> str:
        if self.include_reason is False:
            return None

        incorrect_statements: List[Tuple[str, str]] = []
        for idx, verdict in enumerate(self.verdicts):
            if verdict.verdict.strip().lower() == "no":
                incorrect_statements.append((self.statements[idx], verdict.reason))

        prompt = AnswerCorrectnessTemplate.generate_reason(
            incorrect_statements=incorrect_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(
                    prompt=prompt, schema=Reason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["reason"]

    def _get_reason(self) -> str:
        if self.include_reason is False:
            return None

        incorrect_statements: List[Tuple[str, str]] = []
        for idx, verdict in enumerate(self.verdicts):
            if verdict.verdict.strip().lower() == "no":
                incorrect_statements.append((self.statements[idx], verdict.reason))

        prompt = AnswerCorrectnessTemplate.generate_reason(
            incorrect_statements=incorrect_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(
                    prompt=prompt, schema=Reason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["reason"]

    def _compute_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        correct_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                correct_count += 1

        score = correct_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(self, display_meter=_show_indicator):
            try:
                if self.async_mode:
                    loop = get_or_create_event_loop()
                    loop.run_until_complete(
                        self.a_score_example(example, _show_indicator=False)
                    )
                else:
                    self.statements = self._get_statements(example.expected_output)
                    self.verdicts = self._get_verdicts(example.actual_output)
                    self.score = self._compute_score()
                    self.reason = self._get_reason()
                    self.success = self.score >= self.threshold
                    self.verbose_logs = create_verbose_logs(
                        self,
                        steps=[
                            f"Statements:\n{self.statements}",
                            f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                            f"Score: {self.score}\nReason: {self.reason}",
                        ],
                    )
                return self.score
            except Exception as e:
                print(f"Error in score_example for AnswerCorrectnessScorer: {e}")
                raise
    
    async def a_score_example(
        self, 
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(self, async_mode=True, display_meter=_show_indicator):
            try:
                self.statements: List[str] = await self._a_get_statements(example.expected_output)
                self.verdicts: List[ACVerdict] = await self._a_get_verdicts(example.actual_output)
                self.score = self._compute_score()
                self.reason = await self._a_get_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{self.statements}",
                        f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score
            except Exception as e:
                print(f"Error in a_score_example for AnswerCorrectnessScorer: {e}")
                raise

    def _success_check(self) -> bool:
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
        return "Answer Correctness"
