from typing import Optional, List, Union, Tuple

from judgeval.constants import APIScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    scorer_progress_meter,
    create_verbose_logs,
    parse_response_json,
    check_example_params
)
from judgeval.scorers import JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.judgeval_scorers.local_implementations.answer_relevancy.prompts import (
    Statements,
    ARVerdict,
    Verdicts,
    Reason,
    AnswerRelevancyTemplate,
)

required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
]


class AnswerRelevancyScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        super().__init__(
            score_type=APIScorer.ANSWER_RELEVANCY,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

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
                    self.statements: List[str] = self._get_statements(
                        example.actual_output
                    )
                    self.verdicts: List[ARVerdict] = (
                        self._get_verdicts(example.input)
                    )
                    self.score = self._compute_score()
                    self.reason = self._get_reason(example.input)
                    self.success = self.score >= self.threshold
                    self.verbose_logs = create_verbose_logs(
                        self,
                        steps=[
                            f"Statements:\n{self.statements}",
                            # Convert to dict for serialization purposes
                            f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                            f"Score: {self.score}\nReason: {self.reason}",
                        ],
                    )
                    return self.score
            except Exception as e:
                raise

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)
        try:
            with scorer_progress_meter(
                self, async_mode=True, display_meter=_show_indicator
            ):
                self.statements: List[str] = await self._a_get_statements(
                    example.actual_output
                )
                self.verdicts: List[ARVerdict] = (
                    await self._a_get_verdicts(example.input)
                )
                self.score = self._compute_score()
                self.reason = await self._a_get_reason(example.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{self.statements}",
                        # Convert to dict for serialization purposes
                        f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score
        except Exception as e:
            print(f"Error: {e}")
            raise

    async def _a_get_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements: List[Tuple[str, str]] = []
        for idx, verdict in enumerate(self.verdicts):
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append((self.statements[idx], verdict.reason))

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
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

    def _get_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["reason"]

    async def _a_get_verdicts(
        self, input: str
    ) -> List[ARVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return [
                ARVerdict(**item) for item in data["verdicts"]
            ]
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                return [item for item in res.verdicts]
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return [
                    ARVerdict(**item) for item in data["verdicts"]
                ]

    def _get_verdicts(self, input: str) -> List[ARVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return [ARVerdict(**item) for item in data["verdicts"]]
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return [
                    ARVerdict(**item) for item in data["verdicts"]
                ]

    async def _a_get_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.deduce_statements(
            actual_output=actual_output,
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

    def _get_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.deduce_statements(
            actual_output=actual_output,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["statements"]
        else:
            try:
                res: Statements = self.model.generate(prompt, schema=Statements)
                return res.statements
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["statements"]

    def _compute_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

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
        return "Answer Relevancy"
    