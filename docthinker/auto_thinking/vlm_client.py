"""Async client for an OpenAI-compatible multimodal chat API."""
#对接vlm客户端，负责发送图像和处理响应。
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import asyncio

import aiohttp


class VLMClient:
    """Lightweight wrapper around an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = "https://api.openai.com/v1",
        model: str = "qwen-vl-max",
        timeout: float = 240.0,
    ) -> None:
        self.api_key = api_key
        # Normalise API base: allow passing either root or full chat endpoint.
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"
        self.api_base = api_base
        self.model = model
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    @staticmethod
    def _is_openai_endpoint(url: str) -> bool:
        u = str(url or "").lower()
        return "api.openai.com" in u

    @staticmethod
    def _use_max_completion_tokens(model: str) -> bool:
        return str(model or "").lower().startswith("gpt-5")

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session and not self._session.closed:
                return self._session
            timeout_cfg = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=50, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout_cfg, connector=connector)
            return self._session

    async def close(self) -> None:
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def generate(
        self,
        prompt: str,
        *,
        images: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 5120,
        temperature: float = 0.2,
        extra_messages: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
    ) -> str:
        """Generate a response using the multimodal endpoint."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if extra_messages:
            messages: List[dict] = list(extra_messages)
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if images:
                content = [{"type": "text", "text": prompt}]
                for img in images or []:
                    content.append(self._encode_image(img))
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})

        completion_budget = int(max_tokens)
        if self._use_max_completion_tokens(self.model):
            completion_budget = max(completion_budget, 600)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self._use_max_completion_tokens(self.model):
            payload["max_completion_tokens"] = completion_budget
        else:
            payload["max_tokens"] = completion_budget
        if extra_body:
            try:
                payload.update(dict(extra_body))
            except Exception:
                payload.update({"enable_thinking": extra_body.get("enable_thinking", False)})
        # For some non-OpenAI providers, explicitly disable provider-specific thinking mode.
        if "enable_thinking" not in payload and not self._is_openai_endpoint(self.api_base):
            payload["enable_thinking"] = False
        # OpenAI chat endpoint does not accept provider-specific flags such as enable_thinking.
        if self._is_openai_endpoint(self.api_base):
            payload.pop("enable_thinking", None)

        last_error = None
        for attempt in range(3):
            try:
                timeout_cfg = aiohttp.ClientTimeout(total=self.timeout)
                session = await self._get_session()
                async with session.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout_cfg,
                ) as response:
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
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError(f"Unexpected response payload: {data}")
                    return choices[0]["message"]["content"]
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                last_error = e
                raise
        raise last_error

    _MAX_LONG_EDGE = 4096
    _MAX_PIXELS = 16_000_000

    @classmethod
    def _encode_image(cls, path_like: str) -> dict:
        """Encode image file as the API expects, resizing if too large."""
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path_like}")

        raw = path.read_bytes()
        mime, _ = mimetypes.guess_type(path_like)
        if not mime:
            mime = "image/png"

        raw, mime = cls._maybe_resize(raw, mime)

        encoded = base64.b64encode(raw).decode("utf-8")
        data_uri = f"data:{mime};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": data_uri}}

    @classmethod
    def _maybe_resize(cls, raw: bytes, mime: str) -> tuple:
        """Downscale image if it exceeds VLM API size limits."""
        try:
            from PIL import Image
            import io
        except ImportError:
            return raw, mime

        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        long_edge = max(w, h)
        total_px = w * h

        if long_edge <= cls._MAX_LONG_EDGE and total_px <= cls._MAX_PIXELS:
            return raw, mime

        scale = 1.0
        if long_edge > cls._MAX_LONG_EDGE:
            scale = min(scale, cls._MAX_LONG_EDGE / long_edge)
        if total_px > cls._MAX_PIXELS:
            scale = min(scale, (cls._MAX_PIXELS / total_px) ** 0.5)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        out_fmt = "JPEG" if mime.endswith("jpeg") else "PNG"
        if out_fmt == "JPEG":
            img = img.convert("RGB")
        img.save(buf, format=out_fmt)
        return buf.getvalue(), mime
