#!/usr/bin/env python3
"""
Demo for using rules-based alerts with different types of scorers in Judgeval.

This script demonstrates:
1. How to create API scorers and custom scorers
2. How to define rules with conditions for these scorers
3. How to run an evaluation with rules
4. How to check the alert results
"""

import sys
import os
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from typing import List, Dict, Any, Optional
from datetime import datetime

from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import JudgevalScorer
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import FaithfulnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import AnswerRelevancyScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_correctness import AnswerCorrectnessScorer
from judgeval.data import Example
from judgeval.rules import Rule, Condition
from judgeval.run_evaluation import run_default_eval
from judgeval.evaluation_run import EvaluationRun
from judgeval.tracer import Tracer, TraceEvent, ObservationType, Span, Event
from judgeval.scorers.judgeval_scorer import JudgevalScorer
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer
from judgeval.rules import AlertResult, Condition, Rule, RulesEngine, AlertStatus


# Try to import utilities from judgeval package first, fall back to local helper if needed
try:
    from judgeval.scorers.utils import scorer_progress_meter, create_verbose_logs
    print("Using scoring utilities from judgeval package")
except ImportError:
    try:
        # Try to import from local helper module
        from utils_helper import scorer_progress_meter, create_verbose_logs
        print("Using local helper utilities")
    except ImportError:
        print("WARNING: Could not import scoring utilities. Please ensure either:")
        print("1. The judgeval package is properly installed with scorer utilities, or")
        print("2. The utils_helper.py file exists in the same directory as this script")
        raise


# Create sample examples for evaluation
def create_examples() -> List[Example]:
    """
    Create sample examples for evaluation.
    
    Returns:
        A list of Example objects
    """
    return [
        Example(
            example_id="123",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            input="What is the capital of France?",
            actual_output="The capital of France is Paris, which is located on the Seine River.",
            expected_output="Paris is the capital of France.",
            context=["France is a country in Western Europe.", 
                     "Paris is the capital and most populous city of France."]
        ),
        Example(
            example_id="12",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            input="What is the capital of Germany?",
            actual_output="Berlin is located in northeastern Germany and is the capital of Germany.",
            expected_output="Berlin is the capital of Germany.",
            context=["Germany is a country in Central Europe.", 
                     "Berlin is the capital and largest city of Germany."]
        ),
        Example(
            example_id="10",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            input="What is the capital of Italy?",
            actual_output="The capital of Italy is Milano, which is a major financial center.",  # This is incorrect
            expected_output="Rome is the capital of Italy.",
            context=["Italy is a country in Southern Europe.", 
                     "Rome is the capital city of Italy."]
        ),
    ]


# Custom Scorer Implementation
class SimpleKeywordScorer(JudgevalScorer):
    """
    A simple scorer that checks for keyword presence in the response.
    """
    
    def __init__(self, 
                 keywords: list[str], 
                 threshold: float = 0.5,
                 include_reason: bool = True,
                 async_mode: bool = True,
                 strict_mode: bool = False,
                 verbose_mode: bool = False):
        """
        Initialize the SimpleKeywordScorer.
        
        Args:
            keywords: List of keywords to look for
            threshold: Threshold for success (default 0.5)
            include_reason: Whether to include reason in output
            async_mode: Whether to use async scoring
            strict_mode: Whether to use strict scoring
            verbose_mode: Whether to include verbose logs
        """
        super().__init__(
            score_type="Simple Keyword",
            threshold=1 if strict_mode else threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.keywords = [k.lower() for k in keywords]
        self.error = None
        self.success = False
        self.reason = None
        self.score = 0.0
        self.verbose_logs = None
    
    def score_example(self, example: Example, _show_indicator: bool = True) -> float:
        """
        Score an example based on keyword presence.
        
        Args:
            example: The example to evaluate
            _show_indicator: Whether to show a progress indicator
            
        Returns:
            The calculated score
        """
        with scorer_progress_meter(self, display_meter=_show_indicator):
            try:
                if not example.actual_output:
                    self.score = 0.0
                    self.reason = "Empty response"
                    self.success = False
                    return self.score
                
                response_lower = example.actual_output.lower()
                
                # Count keyword matches
                matches = [keyword for keyword in self.keywords if keyword in response_lower]
                match_count = len(matches)
                
                # Calculate score
                if len(self.keywords) == 0:
                    self.score = 1.0  # Perfect score if no keywords specified
                else:
                    self.score = match_count / len(self.keywords)
                
                self.reason = f"Found {match_count} out of {len(self.keywords)} keywords: {', '.join(matches) if matches else 'none'}"
                self.success = self._success_check()
                
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Looking for keywords: {self.keywords}",
                        f"Found keywords: {matches}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                
                return self.score
            except Exception as e:
                self.error = str(e)
                self.success = False
                print(f"Error in score_example for SimpleKeywordScorer: {e}")
                raise
    
    async def a_score_example(self, example: Example, _show_indicator: bool = True) -> float:
        """
        Async version of score_example.
        
        Args:
            example: The example to evaluate
            _show_indicator: Whether to show a progress indicator
            
        Returns:
            The calculated score
        """
        with scorer_progress_meter(self, async_mode=True, display_meter=_show_indicator):
            try:
                # For this simple implementation, we can just call the sync version
                # In more complex scorers, you might want true async implementation
                if not example.actual_output:
                    self.score = 0.0
                    self.reason = "Empty response"
                    self.success = False
                    return self.score
                
                response_lower = example.actual_output.lower()
                
                # Count keyword matches
                matches = [keyword for keyword in self.keywords if keyword in response_lower]
                match_count = len(matches)
                
                # Calculate score
                if len(self.keywords) == 0:
                    self.score = 1.0  # Perfect score if no keywords specified
                else:
                    self.score = match_count / len(self.keywords)
                
                self.reason = f"Found {match_count} out of {len(self.keywords)} keywords: {', '.join(matches) if matches else 'none'}"
                self.success = self._success_check()
                
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Looking for keywords: {self.keywords}",
                        f"Found keywords: {matches}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                
                return self.score
            except Exception as e:
                self.error = str(e)
                self.success = False
                print(f"Error in a_score_example for SimpleKeywordScorer: {e}")
                raise
    
    def _success_check(self) -> bool:
        """
        Check if the score meets the threshold
        """
        if self.error is not None:
            return False
        return self.score >= self.threshold
    
    @property
    def __name__(self):
        """Return the name of this scorer"""
        return "Simple Keyword"

def main():
    """Run the demo for rules-based alerts."""
    # Create client
    client = JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))
    
    # Create examples
    examples = create_examples()
    
    # Create scorers
    # API scorers (built-in)
    faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
    relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
    correctness_scorer = AnswerCorrectnessScorer(threshold=0.9)
    
    # Custom scorer - check for capital city keywords
    keyword_scorer = SimpleKeywordScorer(
        keywords=["capital", "city", "located"],
        threshold=0.6
    )
    
    # Create rules
    rules = [
        Rule(
            name="All Conditions Check",
            description="Check if all conditions are met",
            conditions=[
                # Use scorer objects instead of strings
                Condition(metric=FaithfulnessScorer(threshold=0.7)),
                Condition(metric=AnswerRelevancyScorer(threshold=0.8)),
                Condition(metric=AnswerCorrectnessScorer(threshold=0.9))
            ],
            combine_type="all"  # Require all conditions to trigger
        ),
        Rule(
            name="Any Condition Check",
            description="Check if any condition is met",
            conditions=[
                Condition(metric=FaithfulnessScorer(threshold=0.7)),
                Condition(metric=AnswerRelevancyScorer(threshold=0.8)),
                Condition(metric=AnswerCorrectnessScorer(threshold=0.9))
            ],
            combine_type="any"  # Require any condition to trigger
        ),
        # New rule for custom scorer
        Rule(
            name="Keyword Check",
            description="Check if response contains enough keywords",
            conditions=[
                Condition(metric=SimpleKeywordScorer(keywords=["restaurant", "food", "cuisine"], threshold=0.6))
            ],
            combine_type="all"
        ),
        # Combined rule with custom and API scorers
        Rule(
            name="Comprehensive Quality Check",
            description="Check for both keyword presence and correctness",
            conditions=[
                Condition(metric=SimpleKeywordScorer(keywords=["restaurant", "food", "cuisine"], threshold=0.6)),
                Condition(metric=AnswerCorrectnessScorer(threshold=0.8))
            ],
            combine_type="all"
        )
    ]
    
    # Run evaluation with rules
    print("Running evaluation with rules...")
    results = client.run_evaluation(
        project_name="rules_demo",
        eval_run_name="rules_test",
        examples=examples,
        scorers=[faithfulness_scorer, relevancy_scorer, correctness_scorer, keyword_scorer],  # Added custom scorer
        model="gpt-4",  # Or any other supported model
        rules=rules, override=True
    )
    
    # Process and display results
    print("\nEvaluation Results:")
    for idx, result in enumerate(results):
        print(f"\nExample {idx+1} (ID: {result.example_id}):")
        print(f"  Input: {result.input}")
        print(f"  Actual Output: {result.actual_output}")
        print(f"  Expected Output: {result.expected_output}")
        
        # Display scorer results
        print("  Scorer Results:")
        for scorer_data in result.scorers_data:
            print(f"    {scorer_data.name}: {scorer_data.score:.2f} (Threshold: {scorer_data.threshold:.2f})")
        
        # Display alert results
        if result.additional_metadata and "alerts" in result.additional_metadata:
            print("  Alert Results:")
            alerts = result.additional_metadata["alerts"]
            for rule_id, alert in alerts.items():
                status = alert["status"]
                status_str = "TRIGGERED" if status == "triggered" else "NOT TRIGGERED"
                print(f"    Rule '{rule_id}': {status_str}")
                
                # Show condition results
                print("    Condition Results:")
                # Handle both field name variants - server uses conditions_result, client might use conditions_results
                conditions = alert.get("conditions_result", alert.get("conditions_results", []))
                for cond in conditions:
                    passed_str = "PASSED" if cond["passed"] else "FAILED"
                    print(f"      {cond['metric']}: {cond['value']:.2f} vs threshold {cond['threshold']:.2f} - {passed_str}")
                
        print("-" * 80)


if __name__ == "__main__":

    main() 