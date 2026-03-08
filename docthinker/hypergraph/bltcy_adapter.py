"""Adapter helpers for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional, Sequence
import asyncio
import aiohttp

DEFAULT_API_BASE = "https://api.openai.com/v1/chat/completions"


async def bltcy_gpt4o_mini_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[Sequence[dict]] = None,
    image_data: Optional[Iterable[str]] = None,
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs,
) -> str:
    """Call BLTCY GPT-4o-mini with optional image inputs."""

    # Ignore unsupported kwargs (e.g., hashing_kv from GraphCore)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    api_key = api_key or os.environ.get("BLTCY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("BLTCY_API_KEY (or OPENAI_API_KEY) must be provided.")

    api_base = (
        api_base
        or os.environ.get("BLTCY_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_API_BASE
    )
    api_base = api_base.rstrip("/")
    if not api_base.endswith("chat/completions"):
        api_base = api_base + "/chat/completions"
    model = model or os.environ.get("BLTCY_MODEL", "gpt-4o-mini")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for item in history_messages or []:
        content = item.get("content")
        if isinstance(content, list):
            messages.append(item)
        else:
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": str(content),
                }
            )

    if image_data:
        user_content = [{"type": "text", "text": prompt}]
        for encoded in image_data:
            user_content.append({"type": "image_base64", "image": encoded})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": str(prompt)})

    payload = {
        "model": model,
        "messages": messages,
    }
    extra_body = kwargs.pop("extra_body", None)
    if extra_body and isinstance(extra_body, dict):
        try:
            payload.update(extra_body)
        except Exception:
            pass
    stream = kwargs.get("stream", False)
    if not stream and "api.openai.com" not in api_base.lower():
        # Some OpenAI-compatible providers require this flag for non-streaming.
        payload["enable_thinking"] = False
    is_openai_gpt5 = "api.openai.com" in api_base.lower() and str(model or "").lower().startswith("gpt-5")
    if max_tokens is not None:
        try:
            token_value = int(max_tokens)
        except Exception:
            token_value = 1024
        if is_openai_gpt5:
            payload["max_completion_tokens"] = max(token_value, 600)
        else:
            payload["max_tokens"] = token_value
    if temperature is not None:
        try:
            payload["temperature"] = float(temperature)
        except Exception:
            payload["temperature"] = 0.2

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout_cfg = aiohttp.ClientTimeout(total=180)
    last_error = None
    alt_base = os.environ.get(
        "BLTCY_FALLBACK_API_BASE", "https://api.openai.com/v1/chat/completions"
    )
    used_alt = False
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                async with session.post(api_base, json=payload, headers=headers, timeout=timeout_cfg) as response:
                    if response.status >= 400:
                        try:
                            err_text = await response.text()
                        except Exception:
                            err_text = "<no body>"
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=f"Bad Request: {err_text}",
                            headers=response.headers,
                        )
                    data = await response.json()
                    choices = data.get("choices")
                    if not choices:
                        raise RuntimeError(json.dumps(data, ensure_ascii=False))
                    return choices[0]["message"]["content"]
        except aiohttp.ClientResponseError as e:
            last_error = e
            if getattr(e, "status", None) in (429, 500, 502, 503, 504) and attempt < 2:
                msg = str(e)
                # If quota exhausted, do NOT retry to avoid long stalls
                if "insufficient_quota" in msg or "token-limit" in msg:
                    raise
                await asyncio.sleep(1 * (2 ** attempt))
                continue
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
            # If DNS or connector errors occur, try fallback base once
            if not used_alt and ("getaddrinfo" in str(e).lower() or isinstance(e, aiohttp.ClientConnectorError)):
                api_base = alt_base
                used_alt = True
                await asyncio.sleep(0.5)
                continue
            if attempt < 2:
                await asyncio.sleep(1 * (2 ** attempt))
                continue
            raise
    raise last_error
