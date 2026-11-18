"""LLM provider configuration.

Every provider below exposes an OpenAI-compatible Chat Completions API,
so one client library works for all of them. Switch with LLM_PROVIDER in .env.
With DEMO_MODE=true, or when no key is set, a scripted DemoClient is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .fake import DemoClient


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None
    key_env: str
    default_model: str


PROVIDERS: dict[str, Provider] = {
    # Free tier: https://aistudio.google.com/apikey
    "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
    # Free tier: https://console.groq.com/keys
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openai": Provider("openai", None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": Provider("anthropic", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
}


@dataclass(frozen=True)
class ClientInfo:
    client: Any
    model: str
    demo: bool


def make_client() -> ClientInfo:
    """Build a client for LLM_PROVIDER (default gemini), or the demo client."""
    name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if name not in PROVIDERS:
        raise SystemExit(f"Unknown LLM_PROVIDER '{name}'. Choose one of: {', '.join(PROVIDERS)}")
    provider = PROVIDERS[name]
    api_key = os.getenv(provider.key_env, "").strip()

    if os.getenv("DEMO_MODE", "").lower() in {"1", "true", "yes"} or not api_key:
        return ClientInfo(DemoClient(), "demo-script", demo=True)

    from openai import OpenAI  # imported lazily so demo mode needs no network setup

    return ClientInfo(OpenAI(api_key=api_key, base_url=provider.base_url), os.getenv("LLM_MODEL", provider.default_model), demo=False)
