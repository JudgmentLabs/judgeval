<div align="center">

<img src="assets/logo-light.svg#gh-light-mode-only" alt="Judgment Logo" width="400" />
<img src="assets/logo-dark.svg#gh-dark-mode-only" alt="Judgment Logo" width="400" />

**Build monitoring & evaluation pipelines for complex agents**

[Website](https://www.judgmentlabs.ai/) • [Twitter/X](https://x.com/JudgmentLabs) • [LinkedIn](https://www.linkedin.com/company/judgmentlabs) • [Documentation](https://judgment.mintlify.app/getting_started) • [Demos](https://www.youtube.com/@AlexShan-j3o)

</div>

## 🚀 What is Judgeval?

Judgeval is an open-source tool for testing, monitoring, and optimizing AI agents. Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).


**🔍 Tracing**
* Automatic agent tracing for common agent frameworks and SDKs (LangGraph, OpenAI, Anthropic, etc.)
* Track input/output, latency, cost, token usage at every step
    * Granular cost tracking per customer/per task
* Function tracing with `@judgment.observe` decorator

**🧪 Evals**
* Plug-and-measure 15+ metrics, including:
  * Tool call accuracy
  * Hallucinations
  * Instruction adherence
  * Retrieval context recall

    Our metric implementations are research-backed by Stanford and Berkeley AI labs. Check out our [research](https://judgmentlabs.ai/research)!
* Build custom evaluators that seamlessly connect with our infrastructure!
* Use our evals for:
    * ⚠️ Unit-testing your agent
    * 🔬 Experimentally testing new prompts and models
    * 🛡️ Online evaluations to guardrail your agent's actions and responses

**📊 Datasets**
* Export trace data to datasets hosted on Judgment's Platform and export to JSON, Parquet, S3, etc.
* Run evals on datasets as unit-tests or to A/B test agent configs

**💡 Insights**
* Error clustering groups agent failures to uncover failure patterns and speed up root cause analysis
* Trace agent failures to their exact source. Judgment's Osiris agent localizes errors to specific agent components, enabling precise, targeted fixes.


## 🛠️ Installation

Get started with Judgeval by installing our SDK using pip:

```bash
pip install judgeval
```

Ensure you have your `JUDGMENT_API_KEY` environment variable set to connect to the [Judgment platform](https://app.judgmentlabs.ai/). If you don't have a key, create an account on the platform!

## 🏁 Get Started

Here's how you can quickly start using Judgeval:

### 🛰️ Tracing

Track your agent execution with full observability with just a few lines of code.
Create a file named `traces.py` with the following code:

```python
from judgeval.common.tracer import Tracer, wrap
from openai import OpenAI

client = wrap(OpenAI())
judgment = Tracer(project_name="my_project")

@judgment.observe(span_type="tool")
def my_tool():
    return "What's the capital of the U.S.?"

@judgment.observe(span_type="function")
def main():
    task_input = my_tool()
    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"{task_input}"}]
    )
    return res.choices[0].message.content

main()
```

[Click here](https://judgment.mintlify.app/getting_started#create-your-first-trace) for a more detailed explanation.

### 📝 Offline Evaluations

You can evaluate your agent's execution to measure quality metrics such as hallucination.
Create a file named `evaluate.py` with the following code:

```python evaluate.py
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

[Click here](https://judgment.mintlify.app/getting_started#create-your-first-experiment) for a more detailed explanation.

### 📡 Online Evaluations

Apply performance monitoring to measure the quality of your systems in production, not just on traces.

Using the same `traces.py` file we created earlier, modify `main` function:

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

main()
```

[Click here](https://judgment.mintlify.app/getting_started#create-your-first-online-evaluation) for a more detailed explanation.

## 🏢 Self-Hosting

Run Judgment on your own infrastructure: we provide comprehensive self-hosting capabilities that give you full control over the backend and data plane that Judgeval interfaces with.

### Key Features
* Deploy Judgment on your own AWS account
* Store data in your own Supabase instance
* Access Judgment through your own custom domain

### Getting Started
1. Check out our [self-hosting documentation](https://judgment.mintlify.app/self_hosting/get_started) for detailed setup instructions, along with how your self-hosted instance can be accessed
2. Use the [Judgment CLI](https://github.com/JudgmentLabs/judgment-cli) to deploy your self-hosted environment
3. After your self-hosted instance is setup, make sure the `JUDGMENT_API_URL` environmental variable is set to your self-hosted backend endpoint

## ⭐ Star Us on GitHub

If you find Judgeval useful, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the product.

## 🤝 Contributing

There are many ways to contribute to Judgeval:

- Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
- Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it
- Speaking or writing about Judgment and letting us know!

## Documentation and Demos

For more detailed documentation, please check out our [developer docs](https://judgment.mintlify.app/getting_started) and some of our [demo videos](https://www.youtube.com/@AlexShan-j3o) for reference!
