"""Model factory. Per-role so a strong model can drive the meta agent while
cheaper roles run elsewhere. Editable."""
from __future__ import annotations

import json
import os

from agents import ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

DEFAULT_MODEL = os.environ.get("AR_MODEL", "deepseek-v4-flash-vision-exp")
CONTEXT_TOKENS = int(os.environ.get("AR_CONTEXT_TOKENS", 131072))

# Chat completions, not the Responses API. DeepSeek serves both, but its Responses
# endpoint accepts `store`, `previous_response_id` and `truncation` and then ignores
# them -- history silently does not carry -- and it ignores `thinking: disabled` too,
# so reasoning cannot be turned off there. Chat completions honours all of it.
BASE_URL = os.environ.get("OPENAI_BASE_URL")

# The planner and the meta agent are the two roles that have to reason before they
# can act, and a reasoning model with its reasoning channel closed writes that
# reasoning into the answer instead: the first live run lost a whole node to a
# 40k-character planner message with none of the fields the graph parses.
THINKING_ROLES = {"planner", "meta"}
# Accepted by the endpoint but, measured against it, not actually load-bearing:
# reasoning token counts do not track the setting. Kept because it is the documented
# knob and costs nothing if it starts working.
REASONING_EFFORT = os.environ.get("AR_REASONING_EFFORT", "low")
EXTRA_BODY = json.loads(os.environ.get("AR_LLM_EXTRA_BODY", "{}"))

ROLE_MODELS = {
    "meta": os.environ.get("AR_MODEL_META", DEFAULT_MODEL),
    "scout": os.environ.get("AR_MODEL_SCOUT", DEFAULT_MODEL),
    "engineer": os.environ.get("AR_MODEL_ENGINEER", DEFAULT_MODEL),
    "analyst": os.environ.get("AR_MODEL_ANALYST", DEFAULT_MODEL),
    "planner": os.environ.get("AR_MODEL_PLANNER", DEFAULT_MODEL),
}


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "none"),
                       timeout=float(os.environ.get("AR_LLM_TIMEOUT", 900)), max_retries=3)


def model_for(role: str = "meta") -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(model=ROLE_MODELS.get(role, DEFAULT_MODEL),
                                      openai_client=_client())


def settings_for(role: str = "meta", temperature: float = 0.3) -> ModelSettings:
    """Per-role request settings. No token cap: truncating a role mid-answer loses
    the whole turn's work, which costs more than the tokens it saves."""
    if role in THINKING_ROLES:
        body = {"thinking": {"type": "enabled"}, "reasoning_effort": REASONING_EFFORT}
    else:
        body = {"thinking": {"type": "disabled"}}
    body.update(EXTRA_BODY)
    return ModelSettings(temperature=temperature, extra_body=body)
