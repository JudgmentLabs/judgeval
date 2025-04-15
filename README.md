# Judgeval SDK

Judgeval is an open-source framework for building evaluation pipelines for multi-step agent workflows, supporting both real-time and experimental evaluation setups. To learn more about Judgment or sign up for free, visit our [website](https://www.judgmentlabs.ai/) or check out our [developer docs](https://judgment.mintlify.app/getting_started).

## Features

- **Development and Production Evaluation Layer**: Offers a robust evaluation layer for multi-step agent applications, including unit-testing and performance monitoring.
- **Plug-and-Evaluate**: Integrate LLM systems with 10+ research-backed metrics, including:
  - Hallucination detection
  - RAG retriever quality
  - And more
- **Custom Evaluation Pipelines**: Construct powerful custom evaluation pipelines tailored for your LLM systems.
- **Monitoring in Production**: Utilize state-of-the-art real-time evaluation foundation models to monitor LLM systems effectively.

## Installation

   ```bash
   pip install judgeval
   ```

## Quickstart: Evaluations

You can evaluate your workflow execution data to measure quality metrics such as hallucination.

Create a file named `evaluate.py` with the following code:

```python
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

client = JudgmentClient()

example = Example(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

scorer = FaithfulnessScorer(threshold=0.5)
results = client.run_evaluation(
    examples=[example],
    scorers=[scorer],
    model="gpt-4o",
)
print(results)
```
Click [here](https://judgment.mintlify.app/getting_started#create-your-first-experiment) for a more detailed explanation

## Quickstart: Traces

Track your workflow execution for full observability with just a few lines of code.

Create a file named `traces.py` with the following code:

```python
from judgeval.common.tracer import Tracer, wrap
from openai import OpenAI

client = wrap(OpenAI())
judgment = Tracer(project_name="my_project")

@judgment.observe(span_type="tool")
def my_tool():
    return "Hello world!"

@judgment.observe(span_type="function")
def main():
    task_input = my_tool()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{task_input}"}]
    )
    return res.choices[0].message.content
```
Click [here](https://judgment.mintlify.app/getting_started#create-your-first-trace) for a more detailed explanation

## Quickstart: Online Evaluations

Apply performance monitoring to measure the quality of your systems in production, not just on historical data.

Using the same traces.py file we created earlier:

```python
from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer
from openai import OpenAI

client = wrap(OpenAI())
judgment = Tracer(project_name="my_project")

@judgment.observe(span_type="tool")
def my_tool():
    return "Hello world!"

@judgment.observe(span_type="function")
def main():
    task_input = my_tool()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{task_input}"}]
    ).choices[0].message.content

    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=task_input,
        actual_output=res,
        model="gpt-4o"
    )

    return res
```
Click [here](https://judgment.mintlify.app/getting_started#create-your-first-online-evaluation) for a more detailed explanation

## Working with Datasets

In most scenarios, you'll have multiple examples that you want to evaluate together. Judgeval makes it easy to work with evaluation datasets through the `EvalDataset` class, which is a collection of examples you can scale evaluations across.

For complete documentation, visit our [Datasets Guide](https://judgment.mintlify.app/evaluation/data_datasets#overview).

#### Creating a Dataset

Creating an `EvalDataset` is straightforward - simply supply a list of `Example` objects:

```python
from judgeval.data import Example
from judgeval.data.datasets import EvalDataset

examples = [
    Example(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France."
    ),
    Example(
        input="Calculate 15% of 200",
        actual_output="30"
    ),
    Example(
        input="Write a haiku about programming",
        actual_output="Code flows like water\nBugs emerge from hidden depths\nDebugger saves all"
    )
]

dataset = EvalDataset(examples=examples)

# You can also add examples one at a time
dataset.add_example(Example(
    input="What's the square root of 16?",
    actual_output="4"
))
```

#### Saving and Loading Datasets

JudgeVal supports multiple formats for saving and loading datasets. The simplest way is using Judgment Cloud:

```python
from judgeval import JudgmentClient

# Saving a dataset
client = JudgmentClient()
client.push_dataset(alias="qa_examples", dataset=dataset)

# Loading a dataset
loaded_dataset = client.pull_dataset(alias="qa_examples")
```

#### Evaluating Your Dataset

You can evaluate all examples in your dataset using the `JudgmentClient`:

```python
res = client.evaluate_dataset(
    dataset=dataset,
    scorers=[FaithfulnessScorer(threshold=0.9)],
    model="gpt-4"
)
```

For more advanced usage, including additional storage formats, check out our [detailed documentation](https://judgment.mintlify.app/evaluation/data_datasets#overview).

## Integrations

### Integrating with LangGraph

We make it easy to integrate Judgeval with LangGraph workflows. By adding the `JudgevalCallbackHandler` to your LangGraph workflow, you can automatically trace and monitor your entire workflow execution.

```python
from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler, set_global_handler
from langgraph.graph import StateGraph

judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"), 
    project_name=PROJECT_NAME
)

graph_builder = StateGraph(State)

# YOUR LANGGRAPH WORKFLOW DEFINITION HERE

# Set up the Judgeval handler
handler = JudgevalCallbackHandler(judgment)
set_global_handler(handler)  # This will automatically trace your entire workflow

# Execute your workflow
result = graph.invoke({
    "messages": [HumanMessage(content=prompt)]
})

# Access execution information
print(handler.executed_nodes)      # List of node names that were executed
print(handler.executed_tools)      # List of tool names that were executed
print(handler.executed_node_tools) # Combined execution list (e.g. ['chatbot', 'tools', 'tools:search_tool'])
```

The `JudgevalCallbackHandler` automatically traces:

- All node visits
- Tool calls
- Execution flow
- Additional metadata about your workflow execution

When using the `JudgevalCallbackHandler`, you don't need to manually add `@observe` decorators to your nodes/tools - everything is automatically traced for you.

For more details about LangGraph integration, check out our [Integration Guide](https://judgment.mintlify.app/integrations/langgraph).

## Documentation and Demos

For more detailed documentation, please check out our [docs](https://judgment.mintlify.app/getting_started) and some of our [demo videos](https://www.youtube.com/@AlexShan-j3o) for reference!
