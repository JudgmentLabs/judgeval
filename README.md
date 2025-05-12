<h1 align="center" style="border-bottom: none; line-height: 1.2;">
    <div style="margin-bottom: 0.2em;"> 
        <a href="https://www.judgmentlabs.ai/"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
            <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
            <img alt="Judgment Logo" src="assets/logo-light.svg" width="200" />
        </picture></a>
        <br>
    </div>
    <br>
    <div style="font-size: 0.85em; display: block; margin-bottom: 0.5em;"> 
        Open-source framework for building evaluation pipelines for multi-step agent workflows
    </div>
</h1>
<p>
Judgeval supports both real-time and experimental evaluation setups, helping you build LLM systems that run better with comprehensive tracing, evaluations, and monitoring.
</p>

<div>
<a target="_blank" href="https://judgment.mintlify.app/getting_started">
</a>

</div>

<p align="center">
    <a href="https://www.judgmentlabs.ai/"><b>Website</b></a> •
    <a href="https://x.com/JudgmentLabs"><b>Twitter/X</b></a> •
    <a href="https://www.linkedin.com/company/judgmentlabs"><b>LinkedIn<b></a>•
    <a href="https://judgment.mintlify.app/getting_started"><b>Documentation</b></a> •
    <a href="https://www.youtube.com/@AlexShan-j3o"><b>Demos</b></a>
</p>


<h2 align="center">🚀 What is Judgeval?</h2>

Judgeval is an open-source framework for building evaluation pipelines for multi-step agent workflows, supporting both real-time and experimental evaluation setups.

<br>


**Development and Production Evaluation Layer**: Offers a robust evaluation layer for multi-step agent applications, including unit-testing and performance monitoring capabilities.

**Plug-and-Evaluate**: Easily integrate your LLM systems with 10+ research-backed metrics, including those for Hallucination detection, RAG retriever quality, and more.

**Custom Evaluation Pipelines**: Construct powerful and flexible custom evaluation pipelines tailored specifically for your LLM systems.

**Monitoring in Production**: Utilize state-of-the-art real-time evaluation foundation models to monitor your LLM systems effectively in production environments.


<br>

<h2 align="center">🛠️ Installation</h2>

Get started with Judgeval by installing the SDK using pip:

```bash
pip install judgeval
```

Ensure you have your `JUDGMENT_API_KEY` environment variable set to connect to the Judgeval backend.

<h2 align="center">🏁 Get Started</h2>

Here's how you can quickly start using Judgeval:

<h3 align="center">📝 Offline Evaluations</h3>
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
    model="gpt-4.1",
)
print(results)
```
Click [here](https://judgment.mintlify.app/getting_started#create-your-first-experiment) for a more detailed explanation.


<h3 align="center">📡 Online Evaluations</h3>
Apply performance monitoring to measure the quality of your systems in production, not just on historical data.

Using the same `traces.py` file we created earlier, modify `main` function

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
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"{task_input}"}]
    ).choices[0].message.content

    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=task_input,
        actual_output=res,
        model="gpt-4.1"
    )
    print("Online evaluation submitted.")
    return res

if __name__ == "__main__":
    main_result = main()
    print(f"Main function output: {main_result}")
```
Click [here](https://judgment.mintlify.app/getting_started#create-your-first-online-evaluation) for a more detailed explanation.

<h3 align="center">🛰️ Traces</h3>
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
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"{task_input}"}]
    )
    return res.choices[0].message.content

if __name__ == "__main__":
    main_result = main()
    print(f"Main function output: {main_result}")
    print("Trace sent to Judgeval!")
```
Click [here](https://judgment.mintlify.app/getting_started#create-your-first-trace) for a more detailed explanation.

<h2 align="center">⭐ Star Us on GitHub</h2>

If you find Judgeval useful, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the product.

<h2 align="center">🤝 Contributing</h2>

There are many ways to contribute to Judgeval:

* Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
* Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it
* Speaking or writing about Judgment and letting us know

<h2 align="center">Documentation and Demos</h2>

For more detailed documentation, please check out our [docs](https://judgment.mintlify.app/getting_started) and some of our [demo videos](https://www.youtube.com/@AlexShan-j3o) for reference!

##
