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

```bash
export FIREWORKS_API_KEY=...
```

### 1. Set Up Training Configuration

First, configure your training environment and model:

```python
from judgeval.trainer import JudgmentTrainer, TrainerConfig
from judgeval.trainer.trainable_model import TrainableModel

# Set up your judgment tracer first
judgment = Tracer(
    project_name="your-rl-project",
    api_key=os.getenv("JUDGMENT_API_KEY")
)

# Set up your model configuration
model = wrap(TrainableModel(
    TrainerConfig(
        base_model_name="your-base-model",
        model_id="",  # Your fine-tuned model ID
        user_id="",   # Your Fireworks user ID
        deployment_id=""  # Your Fireworks deployment ID
    )
))

# Create training prompts
training_prompts = [
    {"input": "your training data here"},
    # Add more training examples...
]

# Initialize trainer
trainer = JudgmentTrainer(
    tracer=judgment,
    project_name="your-rl-project",
    trainable_model=model,
    config=TrainerConfig(
        base_model_name="your-base-model",
        model_id="",
        user_id="",
        deployment_id="",
        num_steps=32,
        num_prompts_per_step=16,
        num_generations_per_prompt=16
    )
)
```

### 2. Define Your Reward Scorer

Create a custom scorer that defines what "good" behavior looks like for your agent:

```python
from judgeval.scorers import ExampleScorer
from judgeval.data import Example

class RewardScorer(ExampleScorer):
    """
    Custom reward function for your agent's performance.
    """
    score_type: str = "CustomReward"

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        # Implement your reward logic here
        # Return a score between 0 and 1 (or higher for bonuses)
        
        # Example: Simple reward based on task completion
        if self.task_completed_successfully(example):
            return 1.0
        elif self.partial_success(example):
            return 0.5
        else:
            return 0.0
    
    def task_completed_successfully(self, example):
        # Your implementation here
        pass
    
    def partial_success(self, example):
        # Your implementation here
        pass
```

### 3. Create Your Agent Function

Define your agent with the `@judgment.observe` decorator:

```python
@judgment.observe(span_type="function")
async def your_agent_function(input_data):
    """
    Your agent implementation.
    
    Args:
        input_data: Input data for your agent
        
    Returns:
        Agent's response/output
    """
    # Your agent logic here
    # Use LLM calls, tool usage, etc.
    
    # Example structure:
    # 1. Process input
    # 2. Make decisions using LLM
    # 3. Execute actions
    # 4. Return results
    
    return agent_output
```

### 4. Start Training

Run your RL training job:

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

## ✨ Features

<img src="assets/product_shot.png" alt="Judgment Platform" width="800" />


|  |  |
|:---|:---:|
| <h3>🧪 Evals</h3>Build custom evaluators on top of your agents. Judgeval supports LLM-as-a-judge, manual labeling, and code-based evaluators that connect with our metric-tracking infrastructure. <br><br>**Useful for:**<br>• ⚠️ Unit-testing <br>• 🔬 A/B testing <br>• 🛡️ Online guardrails | <p align="center"><img src="assets/test.png" alt="Evaluation metrics" width="800"/></p> |
| <h3>📡 Monitoring</h3>Get Slack alerts for agent failures in production. Add custom hooks to address production regressions.<br><br> **Useful for:** <br>• 📉 Identifying degradation early <br>• 📈 Visualizing performance trends across agent versions and time | <p align="center"><img src="assets/errors.png" alt="Monitoring Dashboard" width="1200"/></p> |
| <h3>📊 Datasets</h3>Export environment interactions and test cases to datasets for scaled analysis and optimization. Move datasets to/from Parquet, S3, etc. <br><br>Run evals on datasets as unit tests or to A/B test different agent configurations, enabling continuous learning from production interactions. <br><br> **Useful for:**<br>• 🗃️ Agent environment interaction data for optimization<br>• 🔄 Scaled analysis for A/B tests | <p align="center"><img src="assets/datasets_preview_screenshot.png" alt="Dataset management" width="1200"/></p> |

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

## 📚 Cookbooks

Have your own? We're happy to feature it if you create a PR or message us on [Discord](https://discord.gg/tGVFf8UBUY).

You can access our repo of cookbooks [here](https://github.com/JudgmentLabs/judgment-cookbook).

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
