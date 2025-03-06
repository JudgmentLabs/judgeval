from typing import List, Optional, Union
import asyncio

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
from judgeval.scorers.judgeval_scorers.local_implementations.faithfulness.prompts import (
    FaithfulnessTemplate, 
    Claims
)
from judgeval.scorers.judgeval_scorers.local_implementations.summarization.prompts import *


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
]


class SummarizationScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        n: int = 5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        assessment_questions: Optional[List[str]] = None,
        include_reason: bool = True,
        async_mode=True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        super().__init__(
            score_type=APIScorer.SUMMARIZATION,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

        if assessment_questions is not None and len(assessment_questions) == 0:
            self.assessment_questions = None
        else:
            self.assessment_questions = assessment_questions

        self.include_reason = include_reason
        self.n = n
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)
        try:
            with scorer_progress_meter(self, display_meter=_show_indicator):
                if self.async_mode:
                    loop = get_or_create_event_loop()
                    loop.run_until_complete(
                        self.a_score_example(example, _show_indicator=False)
                    )
                else:
                    self.claims: List[str] = self._generate_claims(
                        example.actual_output
                    )
                    
                    self.info_coverage_verdicts: List[InfoCoverageVerdict] = (
                        self._generate_info_coverage_verdicts(example)
                    )
                    
                    self.contradiction_verdicts: List[ContradictionVerdict] = (
                        self._generate_contradiction_verdicts(example)
                    )
                    
                    contradiction_score = self._calculate_score(ScoreType.CONTRADICTION)
                    info_coverage_score = self._calculate_score(ScoreType.INFO_COVERAGE)
                    self.score_breakdown = {
                        ScoreType.CONTRADICTION.value: contradiction_score,
                        ScoreType.INFO_COVERAGE.value: info_coverage_score,
                    }
                    self.score = min(contradiction_score, info_coverage_score)
                    self.reason = self._generate_reason()
                    self.success = self.score >= self.threshold
                    self.verbose_logs = create_verbose_logs(
                        self,
                        steps=[
                            f"Claims:\n{self.claims}",
                            f"Assessment Questions:\n{self.assessment_questions}",
                            f"Info Coverage Verdicts:\n{[v.model_dump() for v in self.info_coverage_verdicts]}",
                            f"Contradiction Verdicts:\n{[v.model_dump() for v in self.contradiction_verdicts]}",
                            f"Score: {self.score}\nReason: {self.reason}",
                        ],
                    )

                    return self.score
        except Exception as e:
            print(f"Error in SummarizationScorer score_example: {e}")
            raise

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        """
        To score, we take the following steps:
        1. Generate claims from the actual output
            - Extract key factual claims from the summary text

        2. Generate info coverage verdicts:
            a. Generate assessment questions if not provided
            b. Generate answers to the assessment questions for both summary and original text
            c. Compare answers to determine if summary adequately covers key information
            d. Calculate info coverage score based on matching answers

        3. Generate contradiction verdicts:
            a. Generate claims from the actual output
            b. Verify each claim against the original text for factual accuracy
            c. Calculate contradiction score based on verified claims

        4. Calculate final score:
            - Take minimum of info coverage and contradiction scores
            - Generate reason explaining the scoring
            - Check if score meets threshold for success
        """
        check_example_params(example, required_params, self)
        try:
            with scorer_progress_meter(
                self,
                async_mode=True,
                display_meter=_show_indicator,
            ):
                self.claims = await self._a_generate_claims(example.actual_output),
                
                self.info_coverage_verdicts, self.contradiction_verdicts = await asyncio.gather(
                    self._a_generate_info_coverage_verdicts(example),
                    self._a_generate_contradiction_verdicts(example),
                )
                
                contradiction_score = self._calculate_score(ScoreType.CONTRADICTION)
                info_coverage_score = self._calculate_score(ScoreType.INFO_COVERAGE)
                self.score_breakdown = {
                    ScoreType.CONTRADICTION.value: contradiction_score,
                    ScoreType.INFO_COVERAGE.value: info_coverage_score,
                }
                self.score = min(contradiction_score, info_coverage_score)
                self.reason = await self._a_generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Claims:\n{self.claims}",
                        f"Assessment Questions:\n{self.assessment_questions}",
                        f"Info Coverage Verdicts:\n{[v.model_dump() for v in self.info_coverage_verdicts]}",
                        f"Contradiction Verdicts:\n{[v.model_dump() for v in self.contradiction_verdicts]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score
        except Exception as e:
            print(f"Error in SummarizationScorer a_score_example: {e}")
            raise

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        redundancies = []
        for verdict in self.contradiction_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.info_coverage_verdicts:
            for verdict in self.info_coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """JSON:
"""

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
        redundancies = []
        for verdict in self.contradiction_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.info_coverage_verdicts:
            for verdict in self.info_coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """JSON:
"""

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

    def _calculate_score(self, score_type: ScoreType) -> float:
        if score_type == ScoreType.CONTRADICTION:
            total = len(self.contradiction_verdicts)
            if total == 0:
                return 0
            faithfulness_count = 0
            for verdict in self.contradiction_verdicts:
                # Different from the faithfulness score, this
                # penalizes 'idk' (full of fluff) summaries
                if verdict.verdict.strip().lower() == "yes":
                    faithfulness_count += 1

            score = faithfulness_count / total

        else:
            if self.assessment_questions is None:
                return 1
            total = 0
            coverage_count = 0
            for verdict in self.info_coverage_verdicts:
                if verdict.original_verdict.strip().lower() == "yes":
                    total += 1
                    if verdict.summary_verdict.strip().lower() == "yes":
                        coverage_count += 1

            if total == 0:
                return 0

            score = coverage_count / total

        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_answers(self, text: str) -> List[str]:
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = await self.model.a_generate(
                    prompt, schema=Answers
                )
                return res.answers
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["answers"]

    def _generate_answers(self, text: str) -> List[str]:
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = self.model.generate(prompt, schema=Answers)
                return res.answers
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["answers"]

    async def _a_generate_assessment_questions(self, text: str):
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["questions"]
        else:
            try:
                res: Questions = await self.model.a_generate(
                    prompt, schema=Questions
                )
                return res.questions
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["questions"]

    def _generate_assessment_questions(self, text: str):
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["questions"]
        else:
            try:
                res: Questions = self.model.generate(prompt, schema=Questions)
                return res.questions
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["questions"]

    async def _a_generate_info_coverage_verdicts(
        self, example: Example
    ) -> List[InfoCoverageVerdict]:
        if self.assessment_questions is None:
            self.assessment_questions = (
                await self._a_generate_assessment_questions(example.input)
            )

        tasks = [
            self._a_generate_answers(example.input),
            self._a_generate_answers(example.actual_output),
        ]
        results = await asyncio.gather(*tasks)
        original_answers = results[0]
        summary_answers = results[1]

        if len(original_answers) != len(summary_answers):
            raise ValueError("Number of verdicts generated does not equal.")

        coverage_veridcts: List[InfoCoverageVerdict] = []
        for i in range(len(original_answers)):
            coverage_veridcts.append(
                InfoCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )
        return coverage_veridcts

    def _generate_info_coverage_verdicts(
        self, example: Example
    ) -> List[InfoCoverageVerdict]:
        if self.assessment_questions is None:
            self.assessment_questions = self._generate_assessment_questions(
                example.input
            )

        original_answers = self._generate_answers(example.input)
        summary_answers = self._generate_answers(example.actual_output)

        if len(original_answers) != len(summary_answers):
            raise ValueError("Number of verdicts generated does not equal.")

        coverage_veridcts: List[InfoCoverageVerdict] = []
        for i in range(len(original_answers)):
            coverage_veridcts.append(
                InfoCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )

        return coverage_veridcts

    async def _a_generate_contradiction_verdicts(
        self,
        example: Example,
    ) -> List[ContradictionVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[ContradictionVerdict] = []

        prompt = SummarizationTemplate.generate_contradiction_verdicts(
            original_text=example.input,
            summary_claims=self.claims
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                ContradictionVerdict(**item)
                for item in data["verdicts"]
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
                    ContradictionVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_contradiction_verdicts(
        self,
        example: Example,
    ) -> List[ContradictionVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[ContradictionVerdict] = []

        prompt = SummarizationTemplate.generate_contradiction_verdicts(
            original_text=example.input,
            summary_claims=self.claims
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                ContradictionVerdict(**item)
                for item in data["verdicts"]
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
                    ContradictionVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template since it already works
        prompt = FaithfulnessTemplate.find_claims(text=text)
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

    def _generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template
        prompt = FaithfulnessTemplate.find_claims(text=text)
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
        return "Summarization"
    