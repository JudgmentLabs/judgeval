from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import os
from typing import overload


@overload
def optional_env_var(var_name: str) -> str | None: ...


@overload
def optional_env_var(var_name: str, default: str) -> str: ...


def optional_env_var(var_name: str, default: str | None = None) -> str | None:
    return os.getenv(var_name, default)


JUDGMENT_API_KEY = optional_env_var("JUDGMENT_API_KEY")
JUDGMENT_ORG_ID = optional_env_var("JUDGMENT_ORG_ID")
JUDGMENT_API_URL = optional_env_var("JUDGMENT_API_URL", "https://api.judgmentlabs.ai")

JUDGMENT_DEFAULT_GPT_MODEL = optional_env_var(
    "JUDGMENT_DEFAULT_GPT_MODEL", "gpt-5-mini"
)
JUDGMENT_DEFAULT_TOGETHER_MODEL = optional_env_var(
    "JUDGMENT_DEFAULT_TOGETHER_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
)
JUDGMENT_MAX_CONCURRENT_EVALUATIONS = int(
    optional_env_var("JUDGMENT_MAX_CONCURRENT_EVALUATIONS", "10")
)


JUDGMENT_ENABLE_MONITORING = optional_env_var("JUDGMENT_ENABLE_MONITORING", "true")
JUDGMENT_ENABLE_EVALUATIONS = optional_env_var("JUDGMENT_ENABLE_EVALUATIONS", "true")

JUDGMENT_S3_ACCESS_KEY_ID = optional_env_var("JUDGMENT_S3_ACCESS_KEY_ID")
JUDGMENT_S3_SECRET_ACCESS_KEY = optional_env_var("JUDGMENT_S3_SECRET_ACCESS_KEY")
JUDGMENT_S3_REGION_NAME = optional_env_var("JUDGMENT_S3_REGION_NAME")
JUDGMENT_S3_BUCKET_NAME = optional_env_var("JUDGMENT_S3_BUCKET_NAME")
JUDGMENT_S3_PREFIX = optional_env_var("JUDGMENT_S3_PREFIX", "spans/")
JUDGMENT_S3_ENDPOINT_URL = optional_env_var("JUDGMENT_S3_ENDPOINT_URL")
JUDGMENT_S3_SIGNATURE_VERSION = optional_env_var("JUDGMENT_S3_SIGNATURE_VERSION", "s3")
JUDGMENT_S3_ADDRESSING_STYLE = optional_env_var("JUDGMENT_S3_ADDRESSING_STYLE", "auto")


JUDGMENT_NO_COLOR = optional_env_var("JUDGMENT_NO_COLOR")
JUDGMENT_LOG_LEVEL = optional_env_var("JUDGMENT_LOG_LEVEL", "WARNING")


TOGETHERAI_API_KEY = optional_env_var("TOGETHERAI_API_KEY")
TOGETHER_API_KEY = optional_env_var("TOGETHER_API_KEY")
