<div align="center">

<a href="https://judgmentlabs.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_darkmode.svg">
    <img src="assets/logo_lightmode.svg" alt="Judgment Logo" width="400" />
  </picture>
</a>

<br>

## Agent Behavior Monitoring (ABM)

Track and judge any agent behavior in online and offline setups. Set up Sentry-style alerts and run RL easily! 

[![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.judgmentlabs.ai/documentation)
[![Judgment Cloud](https://img.shields.io/badge/Judgment%20Cloud-brightgreen)](https://app.judgmentlabs.ai/register)
[![Self-Host](https://img.shields.io/badge/Self--Host-orange)](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)


[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

</div>


</table>

## [NEW] 🎆 Agent Reinforcement Learning

Train your agents with reinforcement learning using [Fireworks AI](https://fireworks.ai/)! Judgeval now integrates with Fireworks' Reinforcement Fine-Tuning (RFT) endpoint. 
Judgeval provides a simple harness for integrating GRPO into any Python agent, giving builders a quick method to **try RL with minimal code changes** to their existing agents!

```python
await trainer.train(
    agent_function=your_agent_function,
    scorers=[RewardScorer()],  # Custom scorer you define based on task criteria
    prompts=training_prompts,  # Tasks
    rft_provider="fireworks"
)
```

**That's it!** Judgeval automatically manages trajectory collection and reward tagging - your agent can learn from production data with minimal code changes. You can view and monitor training progress for free via the [Judgment Dashboard](https://app.judgmentlabs.ai/).

## Judgeval Overview

Judgeval is an open-source framework for agent behavior monitoring. Judgeval offers a toolkit to track and judge agent behavior in online and offline setups, enabling you to convert interaction data from production/test environments into improved agents. To get started, try running one of the notebooks below depending on your use case or diving deeper in our [docs](https://docs.judgmentlabs.ai/documentation).

Our mission is to unlock the power of production data for agent development, enabling teams to improve their apps by catching real-time failures and optimizing over their users' preferences.

## 📚 Cookbooks

| Use Case | Link |
|:---------|:-----|
| Custom Scorers | [Link to custom scorers cookbook] |
| Online Monitoring | [Link to monitoring cookbook] |
| RL | [Link to RL cookbook] |

You can access our repo of cookbooks [here](https://github.com/JudgmentLabs/judgeval-cookbook).


## Why Judgeval?

• **Custom Evaluators**: Judgeval provides simple abstractions for custom evaluators and their applications to your agents, supporting LLM-as-a-judge and code-based evaluators that connect to datasets our and metric-tracking infrastructure. [Learn more](https://docs.judgmentlabs.ai/documentation/evaluation/scorers/custom-scorers)

• **Production Monitoring**: Run any custom scorer online in production. Get Slack alerts for failures and add custom hooks to address regressions before they impact users. [Learn more](https://docs.judgmentlabs.ai/documentation/performance/online-evals)

• **Simple to run RL**: Go from agent code to optimization via RL easily without managing compute infrastructure or data pipelines. Simply plug onto your agents in production and train!

• **Experiment with evals**: Test out different prompts, models, or changes to your agent by running tests using any evaluator. Visualize and compare results over time! [Learn more](https://docs.judgmentlabs.ai/documentation/evaluation/introduction)

<!--
<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />


|  |  |
|:---|:---:|
| <h3>🧪 Evals</h3>Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure. <br><br>**Useful for:**<br>• ⚠️ Unit-testing <br>• 🔬 A/B testing <br>• 🛡️ Online guardrails | <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p> |
| <h3>📡 Monitoring</h3>Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.<br><br> **Useful for:** <br>• 📉 Identifying degradation early <br>• 📈 Visualizing performance trends across agent versions and time | <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p> |
| <h3>📊 Datasets</h3>Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc. <br><br>Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions. <br><br> **Useful for:**<br>• 🗃️ Agent environment interaction data for optimization<br>• 🔄 Scaled analysis for A/B tests | <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p> |

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

**If you don't have keys, [create an account](https://app.judgmentlabs.ai/register) on the platform!**

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
        model="gpt-4.1",
        messages=[{"role": "user", "content": task}]
    )

    judgment.async_evaluate(  # trigger online monitoring
        scorer=AnswerRelevancyScorer(threshold=0.5),  # swap with your scorers
        example=Example(input=task, actual_output=response),  # customize to your data
        model="gpt-4.1",
    )
    return response.choices[0].message.content

run_agent("What is the capital of the United States?")
```


<!--
## 🏢 Self-Hosting

Run Judgment on your own infrastructure: we provide comprehensive self-hosting capabilities that give you full control over the backend and data plane that Judgeval interfaces with.

### Key Features
* Deploy Judgment on your own AWS account
* Store data in your own Supabase instance
* Access Judgment through your own custom domain

### Getting Started
1. Check out our [self-hosting documentation](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started) for detailed setup instructions, along with how your self-hosted instance can be accessed
2. Use the [Judgment CLI](https://docs.judgmentlabs.ai/documentation/developer-tools/judgment-cli/installation) to deploy your self-hosted environment
3. After your self-hosted instance is setup, make sure the `JUDGMENT_API_URL` environmental variable is set to your self-hosted backend endpoint

## 💻 Development with Cursor
Building agents and LLM workflows in Cursor works best when your coding assistant has the proper context about Judgment integration. The Cursor rules file contains the key information needed for your assistant to implement Judgment features effectively.

Refer to the official [documentation](https://docs.judgmentlabs.ai/documentation/developer-tools/cursor/cursor-rules) for access to the rules file and more information on integrating this rules file with your codebase.
-->


## ⭐ Star Us

If you find Judgeval useful, please consider giving us a star! Your support helps us grow our community and continue improving the repository.

## ❤️ Contributors

There are many ways to contribute to Judgeval:

- Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
- Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it
- Speaking or writing about Judgment and letting us know!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
