import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from docthinker.auto_thinking.vlm_client import VLMClient
from docthinker.server.app import AsyncModelRouter
from graphcore.coregraph.llm.openai import openai_complete_if_cache


class _FakeAioHttpResponse:
    def __init__(self, payload: dict):
        self.status = 200
        self._payload = payload
        self.request_info = None
        self.history = ()
        self.headers = {}

    async def text(self):
        return ""

    async def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class _FakePostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAioHttpSession:
    def __init__(self):
        self.last_payload = None
        self.closed = False

    def post(self, *_args, **kwargs):
        self.last_payload = json.loads(kwargs["data"])
        return _FakePostContext(_FakeAioHttpResponse(self.last_payload))

    async def close(self):
        self.closed = True


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())

    async def close(self):
        return None


class GPT5TokenCompatTests(unittest.IsolatedAsyncioTestCase):
    async def test_vlm_client_uses_max_completion_tokens_for_gpt5(self):
        client = VLMClient(api_key="k", api_base="https://api.openai.com/v1", model="gpt-5.2")
        fake_session = _FakeAioHttpSession()
        client._session = fake_session

        result = await client.generate("hello", max_tokens=123)
        self.assertEqual(result, "ok")
        self.assertIn("max_completion_tokens", fake_session.last_payload)
        self.assertNotIn("max_tokens", fake_session.last_payload)
        self.assertEqual(fake_session.last_payload["max_completion_tokens"], 600)

    async def test_vlm_client_keeps_max_tokens_for_non_gpt5(self):
        client = VLMClient(api_key="k", api_base="https://api.openai.com/v1", model="gpt-4.1")
        fake_session = _FakeAioHttpSession()
        client._session = fake_session

        result = await client.generate("hello", max_tokens=321)
        self.assertEqual(result, "ok")
        self.assertIn("max_tokens", fake_session.last_payload)
        self.assertNotIn("max_completion_tokens", fake_session.last_payload)
        self.assertEqual(fake_session.last_payload["max_tokens"], 321)

    async def test_async_model_router_uses_max_completion_tokens_for_gpt5(self):
        fake_client = _FakeOpenAIClient()
        router = AsyncModelRouter(client=fake_client, models=["gpt-5.2"])
        await router.chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=77)
        call = fake_client.chat.completions.calls[-1]
        self.assertIn("max_completion_tokens", call)
        self.assertNotIn("max_tokens", call)
        self.assertEqual(call["max_completion_tokens"], 600)

    async def test_graphcore_openai_maps_max_tokens_for_gpt5(self):
        fake_client = _FakeOpenAIClient()

        with patch("graphcore.coregraph.llm.openai.create_openai_async_client", return_value=fake_client):
            out = await openai_complete_if_cache(
                model="gpt-5.2",
                prompt="hi",
                api_key="k",
                base_url="https://api.openai.com/v1",
                max_tokens=55,
            )

        self.assertEqual(out, "ok")
        call = fake_client.chat.completions.calls[-1]
        self.assertIn("max_completion_tokens", call)
        self.assertNotIn("max_tokens", call)
        self.assertEqual(call["max_completion_tokens"], 600)


if __name__ == "__main__":
    unittest.main()
