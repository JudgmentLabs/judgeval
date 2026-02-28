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


@overload
def optional_int_env_var(var_name: str) -> int | None: ...


@overload
def optional_int_env_var(var_name: str, default: int) -> int: ...


def optional_int_env_var(var_name: str, default: int | None = None) -> int | None:
    result = optional_env_var(var_name)
    if result is None:
        return default
    try:
        return int(result)
    except ValueError:
        return default


JUDGMENT_API_KEY = optional_env_var("JUDGMENT_API_KEY")
JUDGMENT_ORG_ID = optional_env_var("JUDGMENT_ORG_ID")
JUDGMENT_API_URL = optional_env_var("JUDGMENT_API_URL", "https://api.judgmentlabs.ai")

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


JUDGMENT_BG_WORKERS = optional_int_env_var("JUDGMENT_BG_WORKERS", 4)
JUDGMENT_BG_MAX_QUEUE = optional_int_env_var("JUDGMENT_BG_MAX_QUEUE", 1024)

JUDGMENT_NO_COLOR = optional_env_var("JUDGMENT_NO_COLOR")
JUDGMENT_LOG_LEVEL = optional_env_var("JUDGMENT_LOG_LEVEL", "WARNING")
