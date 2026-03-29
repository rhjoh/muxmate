from dataclasses import dataclass
from typing import Literal
import os

ALIBABA_BASE_URL_ANTHROPIC = "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic"
ALIBABA_BASE_URL_OPENAI = "https://coding-intl.dashscope.aliyuncs.com/v1"
ZAI_CODING_PLAN_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


@dataclass()
class AppConfig:
    api_key: str
    provider_type: Literal["anthropic", "openai"]
    base_url: str
    model: str
    max_tokens: int
    command_timeout_seconds: int


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env variable: {name}")
    return value


def load_config() -> AppConfig:
    return AppConfig(
        api_key=require_env("ZAI_API_KEY"),
        provider_type="openai",
        base_url=ZAI_CODING_PLAN_BASE_URL,
        model="glm-5-turbo",
        max_tokens=2048,
        command_timeout_seconds=120,
    )


#
# def load_config() -> AppConfig:
#     return AppConfig(
#         api_key=require_env("ALIBABA_CLOUD_KEY"),
#         provider_type="anthropic",
#         base_url=ALIBABA_BASE_URL_ANTHROPIC,
#         model="glm-5",
#         max_tokens=2048,
#         command_timeout_seconds=120,
#     )
