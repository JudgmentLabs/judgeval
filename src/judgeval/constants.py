"""
Constant variables used throughout source code
"""

from enum import Enum
import litellm
import os


class APIScorerType(str, Enum):
    """
    Collection of proprietary scorers implemented by Judgment.

    These are ready-made evaluation scorers that can be used to evaluate
    Examples via the Judgment API.
    """

    PROMPT_SCORER = "Prompt Scorer"
    FAITHFULNESS = "Faithfulness"
    ANSWER_RELEVANCY = "Answer Relevancy"
    ANSWER_CORRECTNESS = "Answer Correctness"
    INSTRUCTION_ADHERENCE = "Instruction Adherence"
    EXECUTION_ORDER = "Execution Order"
    DERAILMENT = "Derailment"
    TOOL_ORDER = "Tool Order"
    CLASSIFIER = "Classifier"
    TOOL_DEPENDENCY = "Tool Dependency"
    CUSTOM = "Custom"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive lookup
        for member in cls:
            if member.value == value.lower():
                return member


UNBOUNDED_SCORERS: set[APIScorerType] = (
    set()
)  # scorers whose scores are not bounded between 0-1

ROOT_API = os.getenv("JUDGMENT_API_URL", "https://api.judgmentlabs.ai")
# API URLs
JUDGMENT_EVAL_API_URL = f"{ROOT_API}/evaluate/"
JUDGMENT_TRACE_EVAL_API_URL = f"{ROOT_API}/evaluate_trace/"
JUDGMENT_DATASETS_PUSH_API_URL = f"{ROOT_API}/datasets/push/"
JUDGMENT_DATASETS_APPEND_EXAMPLES_API_URL = f"{ROOT_API}/datasets/insert_examples/"
JUDGMENT_DATASETS_PULL_API_URL = f"{ROOT_API}/datasets/pull_for_judgeval/"
JUDGMENT_DATASETS_DELETE_API_URL = f"{ROOT_API}/datasets/delete/"
JUDGMENT_DATASETS_EXPORT_JSONL_API_URL = f"{ROOT_API}/datasets/export_jsonl/"
JUDGMENT_DATASETS_PROJECT_STATS_API_URL = f"{ROOT_API}/datasets/fetch_stats_by_project/"
JUDGMENT_DATASETS_INSERT_API_URL = f"{ROOT_API}/datasets/insert_examples/"
JUDGMENT_EVAL_LOG_API_URL = f"{ROOT_API}/log_eval_results/"
JUDGMENT_EVAL_FETCH_API_URL = f"{ROOT_API}/fetch_experiment_run/"
JUDGMENT_EVAL_DELETE_API_URL = (
    f"{ROOT_API}/delete_eval_results_by_project_and_run_names/"
)
JUDGMENT_EVAL_DELETE_PROJECT_API_URL = f"{ROOT_API}/delete_eval_results_by_project/"
JUDGMENT_PROJECT_DELETE_API_URL = f"{ROOT_API}/projects/delete/"
JUDGMENT_PROJECT_CREATE_API_URL = f"{ROOT_API}/projects/add/"
JUDGMENT_TRACES_FETCH_API_URL = f"{ROOT_API}/traces/fetch/"
JUDGMENT_TRACES_SAVE_API_URL = f"{ROOT_API}/traces/save/"
JUDGMENT_TRACES_UPSERT_API_URL = f"{ROOT_API}/traces/upsert/"
JUDGMENT_TRACES_DELETE_API_URL = f"{ROOT_API}/traces/delete/"
JUDGMENT_TRACES_ADD_ANNOTATION_API_URL = f"{ROOT_API}/traces/add_annotation/"
JUDGMENT_TRACES_SPANS_BATCH_API_URL = f"{ROOT_API}/traces/spans/batch/"
JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL = (
    f"{ROOT_API}/traces/evaluation_runs/batch/"
)
JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL = f"{ROOT_API}/add_to_run_eval_queue/"
JUDGMENT_GET_EVAL_STATUS_API_URL = f"{ROOT_API}/get_evaluation_status/"
# RabbitMQ
RABBITMQ_HOST = os.getenv(
    "RABBITMQ_HOST", "rabbitmq-networklb-faa155df16ec9085.elb.us-west-1.amazonaws.com"
)
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 5672)
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "task_queue")
# Models
LITELLM_SUPPORTED_MODELS = set(litellm.model_list)

TOGETHER_SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "Qwen/Qwen2-VL-72B-Instruct",
    "meta-llama/Llama-Vision-Free",
    "Gryphe/MythoMax-L2-13b",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "deepseek-ai/DeepSeek-R1",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "google/gemma-2-27b-it",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2-72B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "togethercomputer/MoA-1",
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "databricks/dbrx-instruct",
    "meta-llama/Llama-3-8b-chat-hf",
    "google/gemma-2b-it",
    "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-p",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Gryphe/MythoMax-L2-13b-Lite",
    "meta-llama/Llama-2-7b-chat-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-2-13b-chat-hf",
    "scb10x/scb10x-llama3-typhoon-v1-5-8b-instruct",
    "scb10x/scb10x-llama3-typhoon-v1-5x-4f316",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "microsoft/WizardLM-2-8x22B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "scb10x/scb10x-llama3-1-typhoon2-60256",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "scb10x/scb10x-llama3-1-typhoon-18370",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "togethercomputer/MoA-1-Turbo",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

JUDGMENT_SUPPORTED_MODELS = {"osiris-large", "osiris-mini", "osiris"}

ACCEPTABLE_MODELS = (
    set(litellm.model_list) | set(TOGETHER_SUPPORTED_MODELS) | JUDGMENT_SUPPORTED_MODELS
)

## System settings
MAX_WORKER_THREADS = 10

# Maximum number of concurrent operations for evaluation runs
MAX_CONCURRENT_EVALUATIONS = 50  # Adjust based on system capabilities
