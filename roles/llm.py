"""Model factory. Per-role so a strong model can drive the meta agent while
cheaper roles run elsewhere. Editable."""
from __future__ import annotations

import json
import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = os.environ.get("AR_MODEL", "deepseek-v4-flash-vision-exp")
CONTEXT_TOKENS = int(os.environ.get("AR_CONTEXT_TOKENS", 131072))

# Vendor-specific request fields. DeepSeek bills reasoning tokens, and the ReAct
# loops here do not need them, so thinking is off unless overridden.
EXTRA_BODY = json.loads(os.environ.get("AR_LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}'))

ROLE_MODELS = {
    "meta": os.environ.get("AR_MODEL_META", DEFAULT_MODEL),
    "scout": os.environ.get("AR_MODEL_SCOUT", DEFAULT_MODEL),
    "engineer": os.environ.get("AR_MODEL_ENGINEER", DEFAULT_MODEL),
    "analyst": os.environ.get("AR_MODEL_ANALYST", DEFAULT_MODEL),
}


def chat(role: str = "meta", temperature: float = 0.3, **kw) -> ChatOpenAI:
    return ChatOpenAI(
        model=ROLE_MODELS.get(role, DEFAULT_MODEL),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "none"),
        temperature=temperature,
        timeout=int(os.environ.get("AR_LLM_TIMEOUT", 900)),
        max_retries=3,
        extra_body=dict(EXTRA_BODY),
        **kw,
    )
