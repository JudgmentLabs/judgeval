<div align="center">

<a href="https://judgmentlabs.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/new_darkmode.svg">
    <img src="assets/new_lightmode.svg" alt="Judgment Logo" width="400" />
  </picture>
</a>

<br>

## Agent Behavior Monitoring (ABM)

Run online monitoring on agent behavior using any scorer. Set up sentry-style alerts and run RL jobs easily!


[![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.judgmentlabs.ai/documentation)
[![Judgment Cloud](https://img.shields.io/badge/Judgment%20Cloud-brightgreen)](https://app.judgmentlabs.ai/register)
[![Self-Host](https://img.shields.io/badge/Self--Host-orange)](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)


[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/tGVFf8UBUY)

</div>


</table>

## Agent Reinforcement Learning

Train your agents with reinforcement learning using [Fireworks AI](https://fireworks.ai/)! Judgeval integrates seamlessly with Fireworks' Reinforcement Fine-Tuning (RFT) to help your agents learn from reward signals and improve their behavior over time.

```python
# Start the training process
await trainer.train(
    agent_function=your_agent_function,
    scorers=[RewardScorer()],
    prompts=training_prompts,
    rft_provider="fireworks"
)
```

**That's it!** Your agent will now learn from the reward signals and improve over time. Check your [Judgment Dashboard](https://app.judgmentlabs.ai/) to monitor training progress.

## Judgeval Overview

Judgeval is an open-source framework for custom scoring and online monitoring of agent behavior. We believe the true signals lie in agent behavior and production data - which is why Judgeval enables you to run RL jobs directly on high-quality signals from your production environment. This creates a data flywheel from monitoring to improvement, helping your agents continuously learn and adapt.

## 📚 Cookbooks

| Use Case | Link |
|:---------|:-----|
| Custom Scorers | [Link to custom scorers cookbook] |
| Online Monitoring | [Link to monitoring cookbook] |
| RL | [Link to RL cookbook] |

You can access our repo of cookbooks [here](https://github.com/JudgmentLabs/judgeval-cookbook).


## Why Judgeval?

• **Custom Evaluators**: Build custom evaluators on top of your agents with LLM-as-a-judge, manual labeling, and code-based evaluators that connect to our metric-tracking infrastructure. [Learn more](https://docs.judgmentlabs.ai/documentation/evaluation/scorers/custom-scorers)

• **Production Monitoring**: Get Slack alerts for agent failures in production with online monitoring. Add custom hooks to address production regressions before they impact users. [Learn more](https://docs.judgmentlabs.ai/documentation/performance/online-evals)

• **Data-Driven Optimization**: Analyze production data and run further optimizations from that data, including reinforcement learning integrations that improve agent performance over time.

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

## ⭐ Star Us on GitHub

If you find Judgeval useful, please consider giving us a star on GitHub! Your support helps us grow our community and continue improving the repository.

## ❤️ Contributors

There are many ways to contribute to Judgeval:

- Submit [bug reports](https://github.com/JudgmentLabs/judgeval/issues) and [feature requests](https://github.com/JudgmentLabs/judgeval/issues)
- Review the documentation and submit [Pull Requests](https://github.com/JudgmentLabs/judgeval/pulls) to improve it
- Speaking or writing about Judgment and letting us know!

<!-- Contributors collage -->
[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval)](https://github.com/JudgmentLabs/judgeval/graphs/contributors)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
