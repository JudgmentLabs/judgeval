<div align="center">

<a href="https://judgmentlabs.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_darkmode.svg">
    <img src="assets/logo_lightmode.svg" alt="Judgment Logo" width="400" />
  </picture>
</a>

<br>

## Agent Behavior Monitoring (ABM)

Track and judge any agent behavior in online and offline setups. Set up Sentry-style alerts and analyze agent behaviors / topic patterns at scale! 

[![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.judgmentlabs.ai/documentation)
[![Judgment Cloud](https://img.shields.io/badge/Judgment%20Cloud-brightgreen)](https://app.judgmentlabs.ai/register)
[![Self-Host](https://img.shields.io/badge/Self--Host-orange)](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)


[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)

</div>


</table>

## [NEW] 🎆 Agent Reinforcement Learning

Train your agents with multi-turn reinforcement learning using judgeval and [Fireworks AI](https://fireworks.ai/)! Judgeval's ABM now integrates with Fireworks' Reinforcement Fine-Tuning (RFT) endpoint, supporting gpt-oss, qwen3, Kimi2, DeepSeek, and more.

Judgeval's agent monitoring infra provides a simple harness for integrating GRPO into any Python agent, giving builders a quick method to **try RL with minimal code changes** to their existing agents!

```python
await trainer.train(
    agent_function=your_agent_function,  # entry point to your agent
    scorers=[RewardScorer()],  # Custom scorer you define based on task criteria, acts as reward
    prompts=training_prompts  # Tasks
)
```

**That's it!** Judgeval automatically manages trajectory collection and reward tagging - your agent can learn from production data with minimal code changes. 

👉 Check out the [Wikipedia Racer notebook](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/rl/WikiRacingAgent_RL.ipynb), where an agent learns to navigate Wikipedia using RL, to see Judgeval in action.


You can view and monitor training progress for free via the [Judgment Dashboard](https://app.judgmentlabs.ai/).


## Judgeval Overview

Judgeval is an open-source framework for agent behavior monitoring. Judgeval offers a toolkit to track and judge agent behavior in online and offline setups, enabling you to convert interaction data from production/test environments into improved agents. To get started, try running one of the notebooks below or dive deeper in our [docs](https://docs.judgmentlabs.ai/documentation).

Our mission is to unlock the power of production data for agent development, enabling teams to improve their apps by catching real-time failures and optimizing over their users' preferences.

## 📚 Cookbooks

| Try Out | Notebook | Description |
|:---------|:-----|:------------|
| RL | [Wikipedia Racer](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/rl/WikiRacingAgent_RL.ipynb) | Train agents with reinforcement learning |
| Online ABM | [Research Agent](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/monitoring/Research_Agent_Online_Monitoring.ipynb) | Monitor agent behavior in production |
| Custom Scorers | [HumanEval](https://colab.research.google.com/github/JudgmentLabs/judgment-cookbook/blob/main/custom_scorers/HumanEval_Custom_Scorer.ipynb) | Build custom evaluators for your agents |
| Offline Testing | [Get Started For Free] | Compare how different prompts, models, or agent configs affect performance across ANY metric |

You can access our [repo of cookbooks](https://github.com/JudgmentLabs/judgment-cookbook).

You can find a list of [video tutorials for Judgeval use cases](https://www.youtube.com/@Alexshander-JL).

## Why Judgeval?

🤖 **Simple to run multi-turn RL**: Optimize your agents with multi-turn RL without managing compute infrastructure or data pipelines. Just add a few lines of code to your existing agent code and train!

⚙️ **Custom Evaluators**: No restriction to only monitoring with prefab scorers. Judgeval provides simple abstractions for custom Python scorers, supporting any LLM-as-a-judge rubrics/models and code-based scorers that integrate to our live agent-tracking infrastructure. [Learn more](https://docs.judgmentlabs.ai/documentation/evaluation/custom-scorers)

🚨 **Production Monitoring**: Run any custom scorer in a hosted, virtualized secure container to flag agent behaviors online in production. Get Slack alerts for failures and add custom hooks to address regressions before they impact users. [Learn more](https://docs.judgmentlabs.ai/documentation/performance/online-evals)

📊 **Behavior/Topic Grouping**: Group agent runs by behavior type or topic for deeper analysis. Drill down into subsets of users, agents, or use cases to reveal patterns of agent behavior.
<!-- Add link to Bucketing docs once we have it -->
<!-- 
TODO: Once we have trainer code docs, plug in here
-->

🧪 **Run experiments on your agents**: Compare test different prompts, models, or agent configs across customer segments. Measure which changes improve agent performance and decrease bad agent behaviors.

<!-- 
Use this once we have AI PM features:

**Run experiments on your agents**: A/B test different prompts, models, or agent configs across customer segments. Measure which changes improve agent performance and decrease bad agent behaviors. [Learn more]

-->

## 🛠️ Quickstart

Get started with Judgeval by installing our SDK using pip:

```bash
pip install judgeval
```

Ensure you have your `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` environment variables set to connect to the [Judgment Platform](https://app.judgmentlabs.ai/).

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

**If you don't have keys, [create an account for free](https://app.judgmentlabs.ai/register) on the platform!**

### Start monitoring with Judgeval

```python
from judgeval.tracer import Tracer, wrap
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
from openai import OpenAI


judgment = Tracer(project_name="default_project")
client = wrap(OpenAI())  # tracks all LLM calls

@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"

@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    task = format_question(prompt)
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": task}]
    )

    judgment.async_evaluate(  # trigger online monitoring
        scorer=AnswerRelevancyScorer(threshold=0.5),  # swap with any scorer
        example=Example(input=task, actual_output=response),  # customize to your data
        model="gpt-5",
    )
    return response.choices[0].message.content

run_agent("What is the capital of the United States?")
```

Running this code will deliver monitoring results to your [free platform account](https://app.judgmentlabs.ai/register) and should look like this:

![Judgment Platform Trajectory View](assets/quickstart_trajectory_ss.png)


### Customizable Scorers Over Agent Behavior

Judgeval's strongest suit is the full customization over the types of scorers you can run online monitoring with. No restrictions to only single-prompt LLM judges or prefab scorers - if you can express your scorer
in python code, judgeval can monitor it! Under the hood, judgeval hosts your scorer in a virtualized secure container, enabling online monitoring for any scorer.


First, create a behavior scorer in a file called `helpfulness_scorer.py`:

```python
from judgeval.data import Example
from judgeval.scorers.example_scorer import ExampleScorer

# Define custom example class
class QuestionAnswer(Example):
    question: str
    answer: str

# Define a server-hosted custom scorer
class HelpfulnessScorer(ExampleScorer):
    name: str = "Helpfulness Scorer"
    server_hosted: bool = True  # Enable server hosting
    async def a_score_example(self, example: QuestionAnswer):
        # Custom scoring logic for agent behavior
        # Can be an arbitrary combination of code and LLM calls
        if len(example.answer) > 10 and "?" not in example.answer:
            self.reason = "Answer is detailed and provides helpful information"
            return 1.0
        else:
            self.reason = "Answer is too brief or unclear"
            return 0.0
```

Then deploy your scorer to Judgment's infrastructure:

```bash
echo "pydantic" > requirements.txt
uv run judgeval upload_scorer helpfulness_scorer.py requirements.txt
```

Now you can instrument your agent with monitoring and online evaluation:

```python
from judgeval.tracer import Tracer, wrap
from helpfulness_scorer import HelpfulnessScorer, QuestionAnswer
from openai import OpenAI

judgment = Tracer(project_name="default_project")
client = wrap(OpenAI())  # tracks all LLM calls

@judgment.observe(span_type="tool")
def format_task(question: str) -> str:  # replace with your prompt engineering
    return f"Please answer the following question: {question}"

@judgment.observe(span_type="tool")
def answer_question(prompt: str) -> str:  # replace with your LLM system calls
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@judgment.observe(span_type="function")
def run_agent(question: str) -> str:
    task = format_task(question)
    answer = answer_question(task)

    # Add online evaluation with server-hosted scorer
    judgment.async_evaluate(
        scorer=HelpfulnessScorer(),
        example=QuestionAnswer(question=question, answer=answer),
        sampling_rate=0.9  # Evaluate 90% of agent runs
    )

    return answer

if __name__ == "__main__":
    result = run_agent("What is the capital of the United States?")
    print(result)
```

Congratulations! Your online eval result should look like this:

![Custom Scorer Online ABM](assets/custom_scorer_online_abm.png)

You can now run any online scorer in a secure Firecracker microVMs with no latency impact on your applications.

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
