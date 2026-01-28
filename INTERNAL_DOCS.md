# Judgeval SDK - Internal Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation & Configuration](#installation--configuration)
3. [Core Concepts](#core-concepts)
4. [Tracing & Monitoring](#tracing--monitoring)
5. [LLM Provider Wrappers](#llm-provider-wrappers)
6. [Scorers & Evaluation](#scorers--evaluation)
7. [Datasets](#datasets)
8. [Prompt Management](#prompt-management)
9. [Reinforcement Learning Training](#reinforcement-learning-training)
10. [CLI Commands](#cli-commands)
11. [Integrations](#integrations)
12. [V1 API (New Architecture)](#v1-api-new-architecture)
13. [Environment Variables](#environment-variables)
14. [Architecture Overview](#architecture-overview)

---

## Overview

Judgeval is an open-source Python SDK for **Agent Behavior Monitoring (ABM)**. It provides tools to:

- **Track and trace** agent behavior in production and test environments
- **Evaluate** agent outputs using built-in and custom scorers
- **Monitor** LLM calls across multiple providers (OpenAI, Anthropic, Google, Together)
- **Train** agents with multi-turn reinforcement learning (GRPO)
- **Manage** datasets and prompts with version control
- **Run experiments** comparing different models, prompts, or configurations

The SDK integrates with the [Judgment Platform](https://app.judgmentlabs.ai/) for visualization, alerting, and analysis.

---

## Installation & Configuration

### Installation

```bash
pip install judgeval

# Optional dependencies
pip install judgeval[s3]      # S3 export support
pip install judgeval[trainer] # Reinforcement learning with Fireworks AI
```

### Required Environment Variables

```bash
export JUDGMENT_API_KEY=your_api_key
export JUDGMENT_ORG_ID=your_organization_id
```

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGMENT_API_URL` | `https://api.judgmentlabs.ai` | API endpoint URL |
| `JUDGMENT_ENABLE_MONITORING` | `true` | Enable/disable tracing |
| `JUDGMENT_ENABLE_EVALUATIONS` | `true` | Enable/disable evaluations |
| `JUDGMENT_DEFAULT_GPT_MODEL` | `gpt-5-mini` | Default model for evaluations |
| `JUDGMENT_MAX_CONCURRENT_EVALUATIONS` | `10` | Max concurrent eval threads |
| `JUDGMENT_LOG_LEVEL` | `WARNING` | Logging level |

---

## Core Concepts

### Example

The fundamental data structure representing a single evaluation case.

```python
from judgeval.data import Example

example = Example(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris",  # Optional
    context=["France is a country in Europe"],  # Optional
    retrieval_context=["Paris is the capital city"],  # Optional
    tools_called=["search_tool"],  # Optional
    expected_tools=["search_tool"],  # Optional
    additional_metadata={"user_id": "123"}  # Optional
)
```

**Available Fields (ExampleParams enum):**
- `INPUT` - The input/query
- `ACTUAL_OUTPUT` - The agent's response
- `EXPECTED_OUTPUT` - Ground truth output
- `CONTEXT` - General context
- `RETRIEVAL_CONTEXT` - Retrieved documents for RAG
- `TOOLS_CALLED` - Tools the agent actually called
- `EXPECTED_TOOLS` - Tools the agent should have called
- `ADDITIONAL_METADATA` - Custom metadata

### ScoringResult

Contains evaluation results for one or more scorers applied to a single example.

```python
from judgeval.data import ScoringResult

# Result structure
result.success          # bool: True if all scorers passed
result.scorers_data     # List[ScorerData]: Individual scorer results
result.data_object      # Example: Original evaluated example
result.run_duration     # float: Evaluation duration in seconds
```

### ScorerData

Individual scorer result within a ScoringResult.

```python
scorer_data.name              # Scorer name
scorer_data.score             # float: Score value
scorer_data.threshold         # float: Pass/fail threshold
scorer_data.success           # bool: score >= threshold
scorer_data.reason            # str: Explanation of score
scorer_data.evaluation_model  # str: Model used for evaluation
scorer_data.error             # str: Error message if failed
```

---

## Tracing & Monitoring

### Tracer (Singleton)

The main entry point for monitoring. Creates spans around functions and tracks LLM calls.

```python
from judgeval.tracer import Tracer, wrap

# Initialize tracer (singleton pattern)
judgment = Tracer(project_name="my_project")

# Wrap LLM clients for automatic tracing
from openai import OpenAI
client = wrap(OpenAI())
```

### @observe Decorator

Instruments functions to create spans with inputs/outputs captured.

```python
@judgment.observe(span_type="function")
def my_function(input: str) -> str:
    return process(input)

@judgment.observe(span_type="tool", span_name="custom_name")
async def async_tool(query: str) -> str:
    return await search(query)
```

**Span Types:**
- `function` - General function
- `tool` - Tool/capability call
- `span` - Generic span (default)
- `llm` - LLM call (auto-detected when wrapping clients)
- `generator` - Generator function
- `generator_item` - Individual yield from generator

**Additional Parameters:**
- `span_name` - Custom span name (defaults to function name)
- `attributes` - Custom span attributes dict
- `scorer_config` - TraceScorerConfig for auto-evaluation
- `disable_generator_yield_span` - Skip spans for individual yields

### @agent Decorator

Creates agent context that propagates to all child spans. Enables tracking agent hierarchy.

```python
class MyAgent:
    def __init__(self, name: str):
        self.name = name
        self.state = {}
    
    @judgment.agent(
        identifier="name",           # Use self.name as instance identifier
        track_state=True,            # Capture self.state before/after
        track_attributes=["state"],  # Or specify specific attributes
        field_mappings={"state": "agent_state"}  # Rename in output
    )
    @judgment.observe(span_type="function")
    async def run(self, prompt: str):
        # All spans inside have agent_id, class_name, instance_name
        pass
```

### Customer ID Tracking

Associate traces with customer/user IDs for filtering.

```python
@judgment.observe(span_type="function")
def handle_request(user_id: str, query: str):
    judgment.set_customer_id(user_id)  # Propagates to all child spans
    return process(query)
```

### async_evaluate

Trigger online evaluation within a traced span.

```python
from judgeval.scorers import AnswerRelevancyScorer

@judgment.observe(span_type="function")
def run_agent(prompt: str) -> str:
    response = get_response(prompt)
    
    # Async evaluation (non-blocking)
    judgment.async_evaluate(
        scorer=AnswerRelevancyScorer(threshold=0.5),
        example=Example(input=prompt, actual_output=response),
        sampling_rate=0.9  # Evaluate 90% of requests
    )
    
    return response
```

### Force Flush

Ensure all spans are exported before shutdown.

```python
# Manual flush
judgment.force_flush(timeout_millis=30000)

# Auto-registered at program exit
```

---

## LLM Provider Wrappers

The SDK automatically instruments LLM calls when you wrap the client.

### Supported Providers

| Provider | Client Class | Import |
|----------|-------------|--------|
| OpenAI | `OpenAI`, `AsyncOpenAI` | `from openai import OpenAI` |
| Anthropic | `Anthropic`, `AsyncAnthropic` | `from anthropic import Anthropic` |
| Together AI | `Together`, `AsyncTogether` | `from together import Together` |
| Google GenAI | `Client` | `from google.genai import Client` |

### Usage

```python
from judgeval.tracer import Tracer, wrap

judgment = Tracer(project_name="my_project")

# OpenAI
from openai import OpenAI
client = wrap(OpenAI())
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Anthropic
from anthropic import Anthropic
client = wrap(Anthropic())
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Together AI
from together import Together
client = wrap(Together())

# Google GenAI
from google.genai import Client
client = wrap(Client())
```

### Traced Data

For each LLM call, the following is captured:
- Input messages
- Model name
- Response content
- Token usage (prompt, completion, total)
- Cost estimates (USD)
- Latency
- Streaming chunks (if streaming)

---

## Scorers & Evaluation

### Built-in API Scorers

Pre-built scorers that run on Judgment's infrastructure.

#### AnswerRelevancyScorer

Evaluates if the output is relevant to the input.

```python
from judgeval.scorers import AnswerRelevancyScorer

scorer = AnswerRelevancyScorer(threshold=0.5)
# Required: input, actual_output
```

#### FaithfulnessScorer

Evaluates if the output is faithful to the retrieval context (no hallucinations).

```python
from judgeval.scorers import FaithfulnessScorer

scorer = FaithfulnessScorer(threshold=0.5)
# Required: input, actual_output, retrieval_context
```

#### AnswerCorrectnessScorer

Evaluates if the output matches the expected output.

```python
from judgeval.scorers import AnswerCorrectnessScorer

scorer = AnswerCorrectnessScorer(threshold=0.5)
# Required: input, actual_output, expected_output
```

#### InstructionAdherenceScorer

Evaluates if the output follows the instructions in the input.

```python
from judgeval.scorers import InstructionAdherenceScorer

scorer = InstructionAdherenceScorer(threshold=0.5)
# Required: input, actual_output
```

### PromptScorer (Custom LLM-as-Judge)

Create custom evaluation criteria using natural language prompts.

```python
from judgeval.scorers import PromptScorer

# Create new scorer
scorer = PromptScorer.create(
    name="tone_checker",
    prompt="Is the response professional and polite? Rate 0-1.",
    threshold=0.7,
    options={"professional": 1.0, "casual": 0.5, "rude": 0.0},  # Optional
    model="gpt-4",
    description="Checks for professional tone"
)

# Retrieve existing scorer
scorer = PromptScorer.get("tone_checker")

# Update scorer
scorer.set_prompt("New prompt text")
scorer.set_threshold(0.8)
scorer.set_model("gpt-5")
```

### TracePromptScorer

PromptScorer variant that works with full traces instead of examples.

```python
from judgeval.scorers import TracePromptScorer

scorer = TracePromptScorer.create(
    name="trace_evaluator",
    prompt="Did the agent complete the task correctly?",
    threshold=0.5
)
```

### Custom ExampleScorer

Create fully custom Python scorers with arbitrary logic.

```python
from judgeval.data import Example
from judgeval.scorers.example_scorer import ExampleScorer

class MyCustomScorer(ExampleScorer):
    name: str = "My Custom Scorer"
    server_hosted: bool = True  # Run on Judgment infrastructure
    
    async def a_score_example(self, example: Example) -> float:
        # Custom scoring logic
        if len(example.actual_output) > 100:
            self.reason = "Output is detailed"
            return 1.0
        else:
            self.reason = "Output is too brief"
            return 0.5
```

### Uploading Custom Scorers

```python
from judgeval import JudgmentClient

client = JudgmentClient()
client.upload_custom_scorer(
    scorer_file_path="my_scorer.py",
    requirements_file_path="requirements.txt",
    unique_name="my_scorer",  # Auto-detected if not provided
    overwrite=False,
    scorer_type="example"  # or "trace"
)
```

### Running Evaluations

```python
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer

client = JudgmentClient()

examples = [
    Example(
        input="What is Python?",
        actual_output="Python is a programming language.",
        retrieval_context=["Python is a high-level programming language."]
    )
]

results = client.run_evaluation(
    examples=examples,
    scorers=[
        AnswerRelevancyScorer(threshold=0.5),
        FaithfulnessScorer(threshold=0.5)
    ],
    project_name="my_project",
    eval_run_name="experiment_1",
    assert_test=False  # Set True to raise on failures
)

# Inspect results
for result in results:
    print(f"Success: {result.success}")
    for scorer_data in result.scorers_data:
        print(f"  {scorer_data.name}: {scorer_data.score} ({scorer_data.reason})")
```

---

## Datasets

Manage collections of examples for evaluation and training.

### Creating Datasets

```python
from judgeval.dataset import Dataset
from judgeval.data import Example

# Create with examples
dataset = Dataset.create(
    name="my_dataset",
    project_name="my_project",
    examples=[
        Example(input="Q1", actual_output="A1"),
        Example(input="Q2", actual_output="A2")
    ],
    overwrite=False,
    batch_size=100  # For large datasets
)
```

### Retrieving Datasets

```python
# Get existing dataset
dataset = Dataset.get(
    name="my_dataset",
    project_name="my_project"
)

# List all datasets in project
datasets = Dataset.list(project_name="my_project")
for ds_info in datasets:
    print(f"{ds_info.name}: {ds_info.entries} entries")
```

### Adding Examples

```python
# Add from iterable
dataset.add_examples(new_examples, batch_size=100)

# Add from JSON file
dataset.add_from_json("examples.json")

# Add from YAML file
dataset.add_from_yaml("examples.yaml")
```

### Exporting Datasets

```python
# Save to file
dataset.save_as("json", "./exports", save_name="my_export")
dataset.save_as("yaml", "./exports")
```

### Iterating

```python
# Datasets are iterable
for example in dataset:
    print(example.input)

print(f"Total: {len(dataset)} examples")
```

### Dataset Kinds

- `example` - Collection of Example objects
- `trace` - Collection of Trace objects (from monitoring)

---

## Prompt Management

Version-controlled prompt storage with tagging.

### Creating Prompts

```python
from judgeval.prompt import Prompt

prompt = Prompt.create(
    project_name="my_project",
    name="system_prompt",
    prompt="You are a helpful assistant. Answer the user's question about {{topic}}.",
    tags=["production", "v1"]
)
```

### Retrieving Prompts

```python
# Get latest version
prompt = Prompt.get(
    project_name="my_project",
    name="system_prompt"
)

# Get specific version
prompt = Prompt.get(
    project_name="my_project",
    name="system_prompt",
    commit_id="abc123"
)

# Get by tag
prompt = Prompt.get(
    project_name="my_project",
    name="system_prompt",
    tag="production"
)
```

### Managing Tags

```python
# Add tags to a commit
Prompt.tag(
    project_name="my_project",
    name="system_prompt",
    commit_id="abc123",
    tags=["staging"]
)

# Remove tags
Prompt.untag(
    project_name="my_project",
    name="system_prompt",
    tags=["old_tag"]
)
```

### Listing Versions

```python
versions = Prompt.list(
    project_name="my_project",
    name="system_prompt"
)

for version in versions:
    print(f"{version.commit_id}: {version.tags} ({version.created_at})")
```

### Compiling Prompts

Use `{{variable}}` syntax for templating.

```python
prompt = Prompt.get(project_name="my_project", name="system_prompt")

# Compile with variables
compiled = prompt.compile(topic="Python programming")
# Result: "You are a helpful assistant. Answer the user's question about Python programming."
```

---

## Reinforcement Learning Training

Train agents using multi-turn reinforcement learning with Fireworks AI.

### Configuration

```python
from judgeval.trainer import TrainerConfig, TrainableModel, JudgmentTrainer
from judgeval.tracer import Tracer

config = TrainerConfig(
    deployment_id="my-deployment",
    user_id="my-user",
    model_id="my-model",
    base_model_name="qwen2p5-7b-instruct",
    rft_provider="fireworks",  # Currently only "fireworks" supported
    num_steps=5,
    num_generations_per_prompt=4,
    num_prompts_per_step=4,
    concurrency=100,
    epochs=1,
    learning_rate=1e-5,
    temperature=1.5,
    max_tokens=50,
    enable_addons=True
)

trainable_model = TrainableModel(config)
tracer = Tracer(project_name="my_project")

trainer = JudgmentTrainer(
    config=config,
    trainable_model=trainable_model,
    tracer=tracer,
    project_name="my_project"
)
```

### Training

```python
from judgeval.scorers import ExampleScorer

class RewardScorer(ExampleScorer):
    name: str = "Reward Scorer"
    
    async def a_score_example(self, example) -> float:
        # Custom reward logic
        return 1.0 if task_completed(example) else 0.0

# Define training prompts
prompts = [
    {"task": "Navigate to Wikipedia article about AI"},
    {"task": "Find information about machine learning"}
]

# Run training
model_config = await trainer.train(
    agent_function=your_agent_function,
    scorers=[RewardScorer()],
    prompts=prompts
)

# Save trained model config
model_config.save_to_file("trained_model.json")
```

### Loading Trained Models

```python
from judgeval.trainer import ModelConfig, TrainableModel

# Load config
loaded_config = ModelConfig.load_from_file("trained_model.json")

# Create model from config
trained_model = TrainableModel.from_model_config(loaded_config)

# Use for inference
response = trained_model.chat.completions.create(
    model="current",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## CLI Commands

### Upload Custom Scorer

```bash
# Upload example scorer
judgeval upload_scorer scorer.py requirements.txt --unique-name my_scorer

# Upload trace scorer
judgeval upload_scorer scorer.py requirements.txt --trace

# Overwrite existing
judgeval upload_scorer scorer.py requirements.txt --overwrite
```

### Load OpenTelemetry Environment

Configure OTEL environment variables for external instrumentation tools.

```bash
judgeval load_otel_env my_project -- python my_app.py
```

### Version

```bash
judgeval version
```

---

## Integrations

### LangGraph

```python
from judgeval.integrations.langgraph import Langgraph

# Initialize LangGraph with OTEL tracing
Langgraph.initialize(otel_only=True)
```

### OpenLIT

```python
from judgeval.integrations.openlit import Openlit

# Initialize OpenLIT integration
Openlit.initialize()
```

### Claude Agent SDK (V1)

```python
from judgeval.v1 import Judgeval

client = Judgeval()

# Wrap Claude agent SDK
from claude_agent_sdk import Agent
wrapped_agent = client.tracer.wrap(Agent())
```

---

## V1 API (New Architecture)

The V1 API provides a cleaner, factory-based interface.

### Initialization

```python
from judgeval.v1 import Judgeval

client = Judgeval(
    api_key="your_key",  # Or use env vars
    organization_id="your_org_id"
)
```

### Tracer Factory

```python
# Access tracer factory
tracer_factory = client.tracer

# Create isolated tracer for testing
isolated_tracer = tracer_factory.create_isolated(project_name="test_project")
```

### Scorers Factory

```python
# Access scorers
scorers = client.scorers

# Built-in scorers
answer_relevancy = scorers.built_in.answer_relevancy(threshold=0.5)
faithfulness = scorers.built_in.faithfulness(threshold=0.5)
answer_correctness = scorers.built_in.answer_correctness(threshold=0.5)
instruction_adherence = scorers.built_in.instruction_adherence(threshold=0.5)

# Prompt scorers
prompt_scorer = scorers.prompt_scorer.create(
    name="my_scorer",
    prompt="Is this good?",
    threshold=0.5
)

# Custom scorers
custom = scorers.custom.get("my_custom_scorer")
```

### Evaluation Factory

```python
evaluation = client.evaluation

results = evaluation.run(
    examples=[...],
    scorers=[...],
    project_name="my_project",
    eval_name="experiment_1"
)
```

### Datasets Factory

```python
datasets = client.datasets

# Create
ds = datasets.create(name="test", project_name="proj", examples=[...])

# Get
ds = datasets.get(name="test", project_name="proj")

# List
all_ds = datasets.list(project_name="proj")
```

### Prompts Factory

```python
prompts = client.prompts

# Create
prompt = prompts.create(
    project_name="proj",
    name="my_prompt",
    prompt="Hello {{name}}",
    tags=["v1"]
)

# Get
prompt = prompts.get(project_name="proj", name="my_prompt")
```

### Trainers Factory

```python
trainers = client.trainers

trainer = trainers.create_fireworks(
    config=config,
    trainable_model=model,
    tracer=tracer,
    project_name="my_project"
)
```

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `JUDGMENT_API_KEY` | API key for Judgment Platform |
| `JUDGMENT_ORG_ID` | Organization ID |

### Optional - API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGMENT_API_URL` | `https://api.judgmentlabs.ai` | API endpoint |
| `JUDGMENT_ENABLE_MONITORING` | `true` | Enable tracing |
| `JUDGMENT_ENABLE_EVALUATIONS` | `true` | Enable evaluations |

### Optional - Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGMENT_DEFAULT_GPT_MODEL` | `gpt-5-mini` | Default evaluation model |
| `JUDGMENT_DEFAULT_TOGETHER_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct-Lite` | Default Together model |
| `JUDGMENT_MAX_CONCURRENT_EVALUATIONS` | `10` | Max concurrent evals |

### Optional - S3 Export

| Variable | Description |
|----------|-------------|
| `JUDGMENT_S3_ACCESS_KEY_ID` | AWS access key |
| `JUDGMENT_S3_SECRET_ACCESS_KEY` | AWS secret key |
| `JUDGMENT_S3_REGION_NAME` | AWS region |
| `JUDGMENT_S3_BUCKET_NAME` | S3 bucket name |
| `JUDGMENT_S3_PREFIX` | Key prefix (default: `spans/`) |
| `JUDGMENT_S3_ENDPOINT_URL` | Custom endpoint (for S3-compatible) |

### Optional - Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGMENT_LOG_LEVEL` | `WARNING` | Logging level |
| `JUDGMENT_NO_COLOR` | unset | Disable colored output |

### Third-Party

| Variable | Description |
|----------|-------------|
| `TOGETHERAI_API_KEY` | Together AI API key |
| `TOGETHER_API_KEY` | Together AI API key (alternate) |

---

## Architecture Overview

### Package Structure

```
judgeval/
├── __init__.py          # JudgmentClient, Judgeval exports
├── api/                 # HTTP client for Judgment API
├── cli.py               # CLI commands (typer)
├── constants.py         # APIScorerType enum, model lists
├── data/                # Core data types
│   ├── example.py       # Example class
│   ├── result.py        # ScoringResult class
│   ├── trace.py         # Trace, TraceSpan classes
│   └── scorer_data.py   # ScorerData class
├── dataset/             # Dataset management
├── evaluation/          # Evaluation runner
├── integrations/        # Third-party integrations
│   ├── langgraph/
│   └── openlit/
├── judges/              # LLM judge utilities
├── prompt/              # Prompt management
├── scorers/             # Scoring system
│   ├── base_scorer.py   # BaseScorer class
│   ├── example_scorer.py # ExampleScorer for custom scorers
│   ├── api_scorer.py    # API scorer configs
│   └── judgeval_scorers/# Built-in scorer implementations
├── tracer/              # Tracing system
│   ├── __init__.py      # Tracer class, wrap function
│   ├── exporters/       # Span exporters (Judgment, S3, InMemory)
│   ├── processors/      # Span processors
│   └── llm/             # LLM provider wrappers
│       ├── llm_openai/
│       ├── llm_anthropic/
│       ├── llm_together/
│       └── llm_google/
├── trainer/             # RL training system
│   ├── trainer.py       # JudgmentTrainer factory
│   ├── fireworks_trainer.py
│   ├── trainable_model.py
│   └── config.py        # TrainerConfig, ModelConfig
├── utils/               # Utilities
│   ├── decorators/
│   ├── wrappers/
│   └── file_utils.py
└── v1/                  # New V1 API architecture
    ├── __init__.py      # Judgeval class
    ├── tracer/
    ├── scorers/
    ├── evaluation/
    ├── datasets/
    ├── prompts/
    └── trainers/
```

### Key Design Patterns

1. **Singleton Pattern** - `Tracer` and `JudgmentClient` use singleton to ensure single instance
2. **Factory Pattern** - `JudgmentTrainer` creates provider-specific trainers
3. **Decorator Pattern** - `@observe` and `@agent` wrap functions with tracing
4. **Context Variables** - Agent context and customer ID propagate through async operations
5. **OpenTelemetry** - Built on OTEL for standard trace/span format

### Data Flow

```
User Code
    │
    ▼
@observe decorator → Creates Span
    │
    ▼
wrap(client) → Instruments LLM calls
    │
    ▼
JudgmentSpanProcessor → Batches spans
    │
    ▼
JudgmentSpanExporter → Sends to API
    │
    ▼
Judgment Platform → Visualization & Alerts
```

### Evaluation Flow

```
Examples + Scorers
    │
    ▼
JudgmentClient.run_evaluation()
    │
    ├── API Scorers → Judgment API
    │
    ├── Custom Scorers (server_hosted=True) → E2B containers
    │
    └── Custom Scorers (local) → Local execution
    │
    ▼
ScoringResults → Logged to platform
```

---

## Best Practices

### 1. Always Initialize Tracer First

```python
from judgeval.tracer import Tracer, wrap

# Initialize before wrapping clients
judgment = Tracer(project_name="my_project")
client = wrap(OpenAI())
```

### 2. Use Appropriate Span Types

```python
@judgment.observe(span_type="function")  # Main logic
def process():
    pass

@judgment.observe(span_type="tool")  # External calls, tools
def call_api():
    pass
```

### 3. Set Customer ID Early

```python
@judgment.observe(span_type="function")
def handle_request(user_id: str):
    judgment.set_customer_id(user_id)  # Set once, propagates everywhere
```

### 4. Use Sampling for High-Volume Evaluation

```python
judgment.async_evaluate(
    scorer=scorer,
    example=example,
    sampling_rate=0.1  # Only evaluate 10%
)
```

### 5. Batch Dataset Operations

```python
# Good - batched
dataset.add_examples(large_list, batch_size=100)

# Avoid - one at a time
for example in large_list:
    dataset.add_examples([example])
```

### 6. Use Tags for Prompt Versioning

```python
# Always tag production prompts
Prompt.tag(project_name="proj", name="prompt", commit_id="abc", tags=["production"])
```

---

## Troubleshooting

### Common Issues

**1. "API Key not set"**
- Ensure `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` are set

**2. "No span context found for async_evaluate"**
- Call `async_evaluate` inside an `@observe` decorated function

**3. "Client will not be wrapped"**
- Initialize `Tracer` before calling `wrap()`

**4. Spans not appearing**
- Check `JUDGMENT_ENABLE_MONITORING=true`
- Call `judgment.force_flush()` before exit

**5. Custom scorer upload fails**
- Ensure file has exactly one `ExampleScorer` subclass
- Check Python syntax in scorer file

---

*Last updated: January 2026*
