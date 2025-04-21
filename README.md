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

    # Basic initialization
    client = wrap(OpenAI())
    judgment = Tracer(project_name="my_project")

    # Or with S3 storage enabled
    # NOTE: Make sure AWS creds correspond to an account with write access to the specified S3 bucket
    judgment = Tracer(
        project_name="my_project",
        use_s3=True,
        s3_bucket_name="my-traces-bucket", # Bucket created automatically if it doesn't exist
        s3_aws_access_key_id="your-access-key",  # Optional: defaults to AWS_ACCESS_KEY_ID env var
        s3_aws_secret_access_key="your-secret-key",  # Optional: defaults to AWS_SECRET_ACCESS_KEY env var
        s3_region_name="us-west-1"  # Optional: defaults to AWS_REGION env var or "us-west-1"
    )

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

## Documentation and Demos

For more detailed documentation, please check out our [docs](https://judgment.mintlify.app/getting_started) and some of our [demo videos](https://www.youtube.com/@AlexShan-j3o) for reference!

## 
