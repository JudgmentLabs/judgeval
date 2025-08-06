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

JUDGMENT_DEFAULT_GPT_MODEL = optional_env_var("JUDGMENT_DEFAULT_GPT_MODEL", "gpt-4.1")
JUDGMENT_DEFAULT_TOGETHER_MODEL = optional_env_var(
    "JUDGMENT_DEFAULT_TOGETHER_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
)


__all__ = (
    "JUDGMENT_API_KEY",
    "JUDGMENT_ORG_ID",
    "JUDGMENT_API_URL",
    "JUDGMENT_DEFAULT_GPT_MODEL",
    "JUDGMENT_DEFAULT_TOGETHER_MODEL",
)
