"""
Metric that evaluates hallucinations in model outputs

The hallucination metric determines whether your LLM generates factually correct information by comparing 
the actual_output to the provided context.

If you're looking to evaluate hallucination for a RAG system, refer to the faithfulness metric instead.

The HallucinationMetric uses an LLM to determine, for each context in contexts, whether there are any 
contradictions to the actual_output.

Although extremely similar to the FaithfulnessMetric, the HallucinationMetric is calculated differently 
since it uses contexts as the source of truth instead. Since contexts is the ideal segment of your 
knowledge base relevant to a specific input, the degree of hallucination can be measured by the degree 
of which the contexts is disagreed upon.

Faithfulness is measuring the number of statements in output that agree with contexts.
Hallucination is measuring the fraction of contexts that agree with output (do not contradict == agree)
"""

from typing import Optional, Union, List

from judgeval.constants import APIScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    scorer_progress_meter,
    create_verbose_logs,
    parse_response_json,
    check_example_params,
)
from judgeval.scorers import JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.judgeval_scorers.local_implementations.hallucination.prompts import *


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.CONTEXT,
]


class HallucinationScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = False,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        super().__init__(
            score_type=APIScorer.HALLUCINATION,
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
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(example, _show_indicator=False)
                )
            else:
                self.verdicts: List[HallucinationVerdict] = (
                    self._generate_verdicts(
                        example.actual_output, example.context
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score <= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(
            self, async_mode=True, display_meter=_show_indicator
        ):
            self.verdicts: List[HallucinationVerdict] = (
                await self._a_generate_verdicts(
                    example.actual_output, example.context
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score <= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self):
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = HallucinationTemplate.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["reason"]

    def _generate_reason(self):
        if self.include_reason is False:
            return None

        factual_alignments = []
        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = HallucinationTemplate.generate_reason(
            factual_alignments=factual_alignments,
            contradictions=contradictions,
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

    async def _a_generate_verdicts(
        self, actual_output: str, contexts: List[str]
    ) -> List[HallucinationVerdict]:
        verdicts: List[HallucinationVerdict] = []
        prompt = HallucinationTemplate.generate_verdicts(
            actual_output=actual_output, contexts=contexts
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                HallucinationVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    HallucinationVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(
        self, actual_output: str, contexts: List[str]
    ) -> List[HallucinationVerdict]:
        verdicts: List[HallucinationVerdict] = []
        prompt = HallucinationTemplate.generate_verdicts(
            actual_output=actual_output, contexts=contexts
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                HallucinationVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    HallucinationVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        hallucination_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                hallucination_count += 1

        score = hallucination_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

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
        return "Hallucination"