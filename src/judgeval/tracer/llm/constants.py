from enum import Enum


class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    GOOGLE = "google"
    GROQ = "groq"
    DEFAULT = "default"
