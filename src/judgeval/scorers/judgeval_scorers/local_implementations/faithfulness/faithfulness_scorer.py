"""
Code for the local implementation of the Faithfulness metric.
"""
from typing import List, Optional, Union
from judgeval.constants import APIScorer
from judgeval.data import (
    Example, 
    ExampleParams
)
from judgeval.scorers import JudgevalScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop, 
    check_example_params
)
from judgeval.judges.utils import create_judge
from judgeval.judges import JudgevalJudge
from judgeval.scorers.utils import (
    scorer_progress_meter, 
    create_verbose_logs, 
    parse_response_json
)
from judgeval.scorers.judgeval_scorers.local_implementations.faithfulness.prompts import *


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.RETRIEVAL_CONTEXT,
]


class FaithfulnessScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        user: Optional[str] = None
    ):
        super().__init__(
            score_type=APIScorer.FAITHFULNESS,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.user = user
        self.model, self.using_native_model = create_judge(model)
        self.using_native_model = True  # NOTE: SETTING THIS FOR LITELLM and TOGETHER usage
        self.evaluation_model = self.model.get_model_name()

    def score_example(
        self,
        example: Example,
        all_claims: bool = False,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)
        
        with scorer_progress_meter(self, display_meter=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(
                        example, 
                        all_claims=all_claims, 
                        _show_indicator=False
                    )
                )
            else:
                self.claims = self._generate_claims(example.actual_output, all_claims=all_claims)
                if self.additional_metadata is None:
                    self.additional_metadata = {}
                self.additional_metadata["claims"] = self.claims  # Add claims generated to metadata

                self.verdicts = self._generate_verdicts(example.retrieval_context)
                self.additional_metadata["verdicts"] = [v.model_dump() for v in self.verdicts]  # Add verdicts generated to metadata
                
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Claims:\n{self.claims}",
                        f"Verdicts:\n{self.verdicts}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(
            self, async_mode=True, display_meter=_show_indicator
        ):
            self.claims = await self._a_generate_claims(example.actual_output)


            if self.additional_metadata is None:
                self.additional_metadata = {}
            self.additional_metadata["claims"] = self.claims

            self.verdicts = await self._a_generate_verdicts(example.retrieval_context)  

            self.additional_metadata["verdicts"] = [v.model_dump() for v in self.verdicts]  # Add verdicts generated to metadata

            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Claims:\n{self.claims}",
                    f"Verdicts:\n{self.verdicts}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.model_dump())

        prompt: dict = FaithfulnessTemplate.justify_reason(
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

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = FaithfulnessTemplate.justify_reason(
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

    async def _a_generate_verdicts(self, retrieval_context: str) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []

        claims = [
            claim["claim"] for claim in self.claims
        ]  # We only need the claims, not the quotes involved

        prompt = FaithfulnessTemplate.create_verdicts(
            claims=claims,
            retrieval_context=retrieval_context,
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                FaithfulnessVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(self, retrieval_context: str) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []

        claims = [
            claim["claim"] for claim in self.claims
        ]  # We only need the claims, not the quotes involved

        prompt = FaithfulnessTemplate.create_verdicts(
            claims=claims,
            retrieval_context=retrieval_context,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                FaithfulnessVerdict(**item) for item in data["verdicts"]
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
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_claims(self, actual_output: str) -> List[str]:
        prompt = FaithfulnessTemplate.find_claims(text=actual_output)
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["claims"]

    def _generate_claims(self, actual_output: str, all_claims: bool = False) -> List[str]:
        prompt = FaithfulnessTemplate.find_claims(text=actual_output)
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["claims"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

        score = faithfulness_count / number_of_verdicts
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

    def get_claims(self):
        return self.claims
    
    def get_verdicts(self):
        return self.verdicts

    @property
    def __name__(self):
        return "Faithfulness"
    