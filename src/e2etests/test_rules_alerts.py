"""
End-to-end tests for rules and alerts functionality in JudgeVal with tracing.
"""

import os
import asyncio
import pytest
from typing import Dict, List
from openai import OpenAI
from uuid import uuid4
from dotenv import load_dotenv

from judgeval.tracer import Tracer, wrap
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    AnswerCorrectnessScorer,
    AnswerRelevancyScorer,
    FaithfulnessScorer,
)
from judgeval.rules import Rule, Condition


# Load environment variables
load_dotenv()

# Create scorer instances to use in rules
faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
answer_relevancy_scorer = AnswerRelevancyScorer(threshold=0.7)
answer_correctness_scorer = AnswerCorrectnessScorer(threshold=0.7)

# Define rules for the tracer
rules = [
    Rule(
        name="All Metrics Quality Check",
        description="Check if all quality metrics meet thresholds",
        conditions=[
            Condition(metric=faithfulness_scorer),
            Condition(metric=answer_relevancy_scorer),
            Condition(metric=answer_correctness_scorer)
        ],
        combine_type="all"  # Require all conditions to trigger
    ),
    Rule(
        name="Any Metric Quality Check",
        description="Check if any quality metric meets threshold",
        conditions=[
            Condition(metric=faithfulness_scorer),
            Condition(metric=answer_relevancy_scorer),
            Condition(metric=answer_correctness_scorer)
        ],
        combine_type="any"  # Require any condition to trigger
    )
]

# Initialize tracer with rules
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="rules_test", rules=rules)
client = wrap(OpenAI())


@judgment.observe(span_type="Evaluation")
async def evaluate_example(example: Example) -> Dict:
    """Evaluate an example using the judgment API."""
    # Create scorers
    correctness_scorer = AnswerCorrectnessScorer(threshold=0.7)
    relevancy_scorer = AnswerRelevancyScorer(threshold=0.7)
    faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
    
    # Initialize client
    judgment_client = JudgmentClient()
    
    # Run evaluation
    results = judgment_client.run_evaluation(
        examples=[example],
        scorers=[correctness_scorer, relevancy_scorer, faithfulness_scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name="rules-test-project",
        eval_run_name=f"rules-test-{uuid4().hex[:8]}",
        override=True
    )
    
    # Get the result
    result = results[0]
    
    # Extract scores
    scores = {
        scorer_data.name: scorer_data.score 
        for scorer_data in result.scorers_data
    }
    
    # Extract alerts if available
    alerts = {}
    if hasattr(result, 'additional_metadata') and result.additional_metadata and 'alerts' in result.additional_metadata:
        alerts = result.additional_metadata['alerts']
    
    # Return formatted result
    return {
        "success": result.success,
        "scores": scores,
        "alerts": alerts
    }


@judgment.observe(span_type="Research")
async def get_llm_response(question: str) -> str:
    """Get a response from the LLM."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    
    # Evaluate the response using the current trace
    judgment.async_evaluate(
        scorers=[
            FaithfulnessScorer(threshold=0.7),
            AnswerRelevancyScorer(threshold=0.7)
        ],
        input=question,
        actual_output=response.choices[0].message.content,
        expected_output=None,  # No expected output for this test
        model="gpt-4",
        log_results=True
    )
    
    return response.choices[0].message.content


@pytest.fixture
def good_example():
    """Fixture for a good example."""
    return Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France.",
        retrieval_context=["Paris is the capital city of France."]
    )


@pytest.fixture
def bad_example():
    """Fixture for a bad example."""
    return Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Berlin.",
        expected_output="Paris is the capital of France.",
        retrieval_context=["Paris is the capital city of France."]
    )


@pytest.mark.asyncio
async def test_basic_rules(good_example, bad_example):
    """Test basic rule evaluation with different condition combinations."""
    
    # Start a trace for the test
    with judgment.trace(name="basic_rules_test", project_name="rules_test", overwrite=True) as trace:
        # Test metadata will be added to the test file output for now
        print("\nRunning basic_rules_test with rules functionality and tracing")
        
        # Evaluate examples
        with trace.span("evaluate_examples") as span:
            # Evaluate good example
            good_result = await evaluate_example(good_example)
            
            # Evaluate bad example
            bad_result = await evaluate_example(bad_example)
            
            # Print results for debugging
            print("\nGood Example Results:")
            print(f"Success: {good_result['success']}")
            print("Scores:")
            for name, score in good_result["scores"].items():
                print(f"  {name}: {score:.2f}")
            
            if good_result["alerts"]:
                print("Alerts:")
                for rule_id, alert in good_result["alerts"].items():
                    status = alert.get('status', 'unknown')
                    rule_name = alert.get('rule_name', rule_id)
                    print(f"  Rule '{rule_name}': {status}")
        
        # Test assertions
        # The good example should have high scores
        assert good_result["scores"]["Answer Correctness"] > 0.7, "Good example should have high correctness score"
        assert good_result["scores"]["Answer Relevancy"] > 0.7, "Good example should have high relevancy score"
        
        # The bad example should have lower scores
        assert bad_result["scores"]["Answer Correctness"] < 0.7, "Bad example should have low correctness score"
        
        # Check for alerts in good example
        if good_result["alerts"]:
            any_rule_id = next((k for k, v in good_result["alerts"].items() 
                                if v.get('rule_name') == "Any Metric Quality Check"), None)
            if any_rule_id:
                assert good_result["alerts"][any_rule_id]['status'] == 'triggered', \
                    "Any Metric Quality Check should be triggered for good example"


@pytest.mark.asyncio
async def test_complex_rules():
    """Test more complex rule combinations."""
    
    # Create scorers for this test
    correctness_scorer = AnswerCorrectnessScorer(threshold=0.7)
    relevancy_scorer = AnswerRelevancyScorer(threshold=0.7)
    
    # Create a custom rule for this test
    complex_rules = [
        Rule(
            name="Mixed Operators Rule",
            description="Check with mixed operators",
            conditions=[
                Condition(metric=correctness_scorer),
                Condition(metric=relevancy_scorer)
            ],
            combine_type="all"
        )
    ]
    
    # Create a new tracer with these rules
    # Note: Using the same tracer to avoid singleton warning
    judgment.rules.extend(complex_rules)
    
    # Create example
    example = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France.",
        retrieval_context=["Paris is the capital city of France."]
    )
    
    # Start a trace for the test
    with judgment.trace(name="complex_rules_test", project_name="rules_test", overwrite=True) as trace:
        # Print test info
        print("\nRunning complex_rules_test with mixed operators")
        
        # Evaluate example
        with trace.span("evaluate_example") as span:
            # Create scorers
            correctness_scorer = AnswerCorrectnessScorer(threshold=0.7)
            relevancy_scorer = AnswerRelevancyScorer(threshold=0.7)
            faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
            
            # Initialize client
            judgment_client = JudgmentClient()
            
            # Run evaluation
            results = judgment_client.run_evaluation(
                examples=[example],
                scorers=[correctness_scorer, relevancy_scorer, faithfulness_scorer],
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                log_results=True,
                project_name="rules-test-project",
                eval_run_name=f"complex-rules-test-{uuid4().hex[:8]}",
                override=True,
                rules=complex_rules  # Pass rules explicitly
            )
            
            # Get the result
            result = results[0]
            
            # Check for alerts
            alerts = {}
            if hasattr(result, 'additional_metadata') and result.additional_metadata and 'alerts' in result.additional_metadata:
                alerts = result.additional_metadata['alerts']
            
            # Test assertions
            assert result.success, "Example evaluation should succeed"
            
            # The Mixed Operators Rule should be triggered
            if alerts:
                mixed_rule_id = next((k for k, v in alerts.items() 
                                    if v.get('rule_name') == "Mixed Operators Rule"), None)
                if mixed_rule_id:
                    assert alerts[mixed_rule_id]['status'] == 'triggered', \
                        "Mixed Operators Rule should be triggered"


@pytest.mark.asyncio
async def test_llm_response_with_evaluation():
    """Test LLM response with evaluation in a trace."""
    
    # Start a trace for the test
    with judgment.trace(name="llm_response_test", project_name="rules_test", overwrite=True) as trace:
        # Print test info
        print("\nRunning llm_response_test to evaluate LLM output")
        
        # Test LLM response with evaluation
        with trace.span("test_llm_response") as span:
            question = "What is the capital of France?"
            response = await get_llm_response(question)
            
            # Test assertions
            assert "Paris" in response, "Response should mention Paris as the capital of France" 