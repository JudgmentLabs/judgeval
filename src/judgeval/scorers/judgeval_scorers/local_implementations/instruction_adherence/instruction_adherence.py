from typing import Optional, List, Union, Tuple
from pydantic import BaseModel

from judgeval.constants import APIScorer
from judgeval.scorers.utils import (get_or_create_event_loop,
                                    scorer_progress_meter,
                                    create_verbose_logs,
                                    parse_response_json,
                                    check_example_params
                                    )
from judgeval.scorers import JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.judgeval_scorers.local_implementations.instruction_adherence.prompt import (
    InstructionAdherenceTemplate,
)
required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
]

class Instructions(BaseModel):
    instructions: List[str]

class Verdict(BaseModel):
    instruction: str
    score: float
    reason: str

class ListOfVerdicts(BaseModel):
    verdicts: List[Verdict]

class InstructionAdherenceScorer(JudgevalScorer):
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
            score_type=APIScorer.INSTRUCTION_ADHERENCE,
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
                    self.instructions: List[str] = self._get_instructions(example.input)
                    self.verdicts: List[Verdict] = (
                        self._get_verdicts(self.instructions, example.actual_output)
                    )
                    self.score = self._compute_score()
                    self.reason = str(self.verdicts)
                    self.success = self.score >= self.threshold
                    self.verbose_logs = create_verbose_logs(
                        self,
                        steps=[
                            f"Instructions:\n{self.instructions}",
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
                self.instructions: List[str] = await self._a_get_instructions(example.input)
                self.verdicts: List[Verdict] = (
                    await self._a_get_verdicts(self.instructions, example.actual_output)
                )
                self.score = self._compute_score()
                self.reason = str(self.verdicts)
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Instructions:\n{self.instructions}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score
        except Exception as e:
            raise e


    async def _a_get_verdicts(
        self, instructions: List[str], actual_output: str
    ) -> List[Verdict]:
        if len(instructions) == 0:
            return []

        prompt = InstructionAdherenceTemplate.generate_verdicts(
            instructions=instructions,
            actual_output=actual_output,
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return [
                Verdict(**item) for item in data["verdicts"]
            ]
        else:
            try:
                res: List[Verdict] = await self.model.a_generate(
                    prompt, schema=List[Verdict]
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return [
                    Verdict(**item) for item in data["verdicts"]
                ]

    def _get_verdicts(self, instructions: List[str], actual_output: str) -> List[Verdict]:
        if len(instructions) == 0:
            return []

        prompt = InstructionAdherenceTemplate.generate_verdicts(
            instructions=instructions,
            actual_output=actual_output,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return [Verdict(**item) for item in data["verdicts"]]
        else:
            try:
                res: List[Verdict] = self.model.generate(prompt, schema=List[Verdict])
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return [
                    Verdict(**item) for item in data["verdicts"]
                ]

    async def _a_get_instructions(
        self,
        input: str,
    ) -> List[str]:
        prompt = InstructionAdherenceTemplate.get_instructions(
            input=input,
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["instructions"]
        else:
            try:
                res: List[str] = await self.model.a_generate(
                    prompt, schema=List[str]
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["instructions"]

    def _get_instructions(
        self,
        input: str,
    ) -> List[str]:
        prompt = InstructionAdherenceTemplate.get_instructions(
            input=input,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["instructions"]
        else:
            try:
                res: List[str] = self.model.generate(prompt, schema=List[str])
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["instructions"]

    def _compute_score(self):
        if len(self.verdicts) == 0:
            return 1
        score = 0
        for verdict in self.verdicts:
            score += verdict.score
        return score / len(self.verdicts)

    def success_check(self) -> bool:
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
        return "Instruction Adherence"