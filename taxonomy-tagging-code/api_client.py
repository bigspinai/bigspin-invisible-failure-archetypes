"""
Thread-safe API clients for Anthropic and OpenAI.

Reusable across all phases. Lazy-initialized, singleton pattern.
"""

import json
import re
import threading

from dotenv import load_dotenv
load_dotenv(override=True)


# ── Thread-safe client singletons ─────────────────────────────────────────

_anthropic_client = None
_openai_client = None
_client_lock = threading.Lock()


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        with _client_lock:
            if _anthropic_client is None:
                from anthropic import Anthropic
                _anthropic_client = Anthropic()
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                from openai import OpenAI
                _openai_client = OpenAI()
    return _openai_client


# ── API call functions ────────────────────────────────────────────────────

def call_anthropic(model: str, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """Call an Anthropic model. Returns the text response."""
    client = _get_anthropic_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def call_openai(model: str, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """Call an OpenAI model. Returns the text response."""
    client = _get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        temperature=0,
        messages=messages,
    )
    return resp.choices[0].message.content


def call_model(provider: str, model: str, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """Call a model by provider name."""
    if provider == "anthropic":
        return call_anthropic(model, prompt, system, max_tokens)
    elif provider == "openai":
        return call_openai(model, prompt, system, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_tagger(config: dict, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """Call a model using a tagger config dict with 'provider' and 'model' keys."""
    return call_model(config["provider"], config["model"], prompt, system, max_tokens)


# ── JSON response parsing ─────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict | None:
    """
    Parse a JSON response from an LLM, handling common formatting issues.
    Returns None if all parsing attempts fail.
    """
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first/last braces
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
