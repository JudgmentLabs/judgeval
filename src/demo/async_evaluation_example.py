#!/usr/bin/env python3
"""
Examples demonstrating how to use async evaluation in multiple ways.
"""

import asyncio
import os
import time
from typing import List

from judgeval.data import Example, ScoringResult
from judgeval.judgment_client import JudgmentClient

# Get Judgment API key from environment (replace with your actual API key)
JUDGMENT_API_KEY = os.environ.get("JUDGMENT_API_KEY", "your_api_key_here")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID", "your_organization_id_here")

# Initialize the JudgmentClient
judgment_client = JudgmentClient(judgment_api_key=JUDGMENT_API_KEY, organization_id=ORGANIZATION_ID)


async def example_direct_await():
    """
    Example of directly awaiting the Task returned by run_evaluation with async_execution=True.
    This is the simplest approach and blocks until evaluation is complete.
    """
    print("\n=== Example: Direct Await ===")
    
    # Create example list
    examples = [
        Example(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris"
        ),
        Example(
            input="What is the capital of Italy?",
            actual_output="Rome is the capital of Italy.",
            expected_output="Rome"
        )
    ]
    
    # Set up scorers
    from judgeval.scorers import AnswerCorrectnessScorer
    scorers = [AnswerCorrectnessScorer(threshold=0.9)]
    
    # Start evaluation asynchronously and get a Task object
    print("Starting evaluation...")
    task = judgment_client.run_evaluation(
        examples=examples,
        scorers=scorers,
        model="gpt-4o-mini",
        project_name="async-examples",
        eval_run_name="async-example-direct",
        override=True,
        async_execution=True
    )
    
    # Directly await the task - this will block until the evaluation is done
    print("Awaiting results...")
    results = await task
    
    print(f"Evaluation completed! Received {len(results)} results")
    
    # Process the results
    print(results)


async def example_with_other_work():
    """
    Example of running other work while evaluation is in progress.
    Shows how to check task status and get results when ready.
    """
    print("\n=== Example: Do Other Work While Evaluating ===")
    
    # Create example list
    examples = [
        Example(
            input="What is the tallest mountain in the world?",
            actual_output="Mount Everest is the tallest mountain in the world.",
            expected_output="Mount Everest"
        ),
        Example(
            input="What is the largest ocean?",
            actual_output="The Pacific Ocean is the largest ocean on Earth.",
            expected_output="Pacific Ocean"
        )
    ]
    
    # Set up scorers
    from judgeval.scorers import AnswerCorrectnessScorer
    scorers = [AnswerCorrectnessScorer(threshold=0.9)]
    
    # Start evaluation asynchronously and get a Task object
    print("Starting evaluation...")
    task = judgment_client.run_evaluation(
        examples=examples,
        scorers=scorers,
        model="gpt-4o-mini",
        project_name="async-examples",
        eval_run_name="async-example-other-work",
        override=True,
        async_execution=True
    )
    
    # Do other work while evaluation is running
    print("Doing other work while evaluation runs in the background...")
    
    # Simulate other work with a few iterations
    for i in range(1, 4):
        print(f"  Doing work iteration {i}...")
        await asyncio.sleep(2)  # Simulate work with a delay
        
        # Check if the evaluation is done
        if task.done():
            print("  Evaluation completed during other work!")
            break
        else:
            print("  Evaluation still running...")
    
    # Get the results when ready
    try:
        if not task.done():
            print("Waiting for evaluation to complete...")
            
        results = await task  # Will return immediately if already done
        
        print(results)
                
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        if task.exception():
            print(f"Task exception: {task.exception()}")


async def main():
    """Run the examples."""
    # Run the first example: direct await
    await example_direct_await()
    
    # Run the second example: do other work while evaluating
    await example_with_other_work()


if __name__ == "__main__":
    asyncio.run(main()) 