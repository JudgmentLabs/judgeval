import os
import importlib.util
import json
import asyncio

from judgeval.data import Example
from judgeval.scorers.answer_relevancy_scorer import AnswerRelevancyScorer

# === Load prompts ===
PROMPT_FILE = "my_agents/examples/prompts.json"
with open(PROMPT_FILE, "r") as f:
    raw_prompts = json.load(f)

examples = [
    Example(
        input=item.get("input") or item.get("prompt"),
        expected_output=item.get("expected_output") or item.get("output"),
        example_id=str(i),
    )
    for i, item in enumerate(raw_prompts)
]

# === Load agent ===
AGENT_FILE = "my_agents/agents/echo_agent.py"
AGENT_FUNCTION_NAME = "run_agent"

spec = importlib.util.spec_from_file_location("agent_module", AGENT_FILE)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
agent_fn = getattr(agent_module, AGENT_FUNCTION_NAME)

# === Run agent and score ===
async def main():
    threshold = 0.5
    scorer = AnswerRelevancyScorer(threshold=threshold)
    passed = 0

    for example in examples:
        generated = agent_fn(example.input)
# example.actual_output = generated  # dynamic attribute used by scorer
        example.additional_metadata['actual_output'] = generated # Use additional_metadata instead
        score = scorer.score_example(example)
        success = score >= threshold
        if success:
            passed += 1

        print(f"Prompt: {example.input}")
        print(f"Generated: {generated}")
        print(f"Expected: {example.expected_output}")
        print(f"Score: {score:.2f} | Passed: {success}")
        print("---")

    print(f"\n Agent Accuracy: {passed}/{len(examples)} passed ({(passed / len(examples)) * 100:.2f}%)")

if __name__ == "__main__":
    asyncio.run(main())
