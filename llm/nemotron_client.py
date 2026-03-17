"""
LLM client for Symphonia Studio.
Supports OpenAI-compatible APIs and NVIDIA Nemotron (via base URL and API key).
Configure via environment: OPENAI_API_KEY / OPENAI_BASE_URL or NEMOTRON_API_KEY / NEMOTRON_BASE_URL.
"""

import json
import os
import re
from typing import Any, Optional

# Optional: use openai package if available for compatibility
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _get_config() -> tuple[str, str, str]:
    """Returns (api_key, base_url, model). Prefer Nemotron env, then OpenAI."""
    api_key = os.environ.get("NEMOTRON_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("NEMOTRON_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = os.environ.get("NEMOTRON_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    return api_key, base_url, model


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from model output (handles markdown code blocks)."""
    text = text.strip()
    # Remove optional markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    # Find first { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return None


def complete(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    Send one user message with a system prompt; return the assistant reply text.
    """
    api_key, base_url, model = _get_config()
    if not api_key and HAS_OPENAI:
        # Allow OpenAI lib to use default key from env
        api_key = "not-set"

    if HAS_OPENAI and api_key:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    # Fallback: raw HTTP for OpenAI-compatible endpoints
    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
            return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"LLM API error {e.code}: {body}") from e


def complete_json(system_prompt: str, user_prompt: str, temperature: float = 0.5) -> Optional[dict]:
    """
    Same as complete() but parses the reply as JSON and returns a dict.
    Returns None if parsing fails.
    """
    raw = complete(system_prompt, user_prompt, temperature=temperature)
    return _extract_json(raw)
