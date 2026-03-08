import unittest

from docthinker.server.app import _make_vision_model_func


class _FakeVLMClient:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return "ok"


class VisionModelAdapterUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_messages_are_forwarded_to_vlm(self):
        client = _FakeVLMClient()
        func = _make_vision_model_func(client)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        out = await func("", messages=messages, system_prompt=None)

        self.assertEqual("ok", out)
        self.assertEqual(1, len(client.calls))
        call = client.calls[0]
        self.assertEqual("", call["prompt"])
        self.assertEqual(messages, call["extra_messages"])
        self.assertIsNone(call["images"])

    async def test_image_data_is_forwarded_as_images(self):
        client = _FakeVLMClient()
        func = _make_vision_model_func(client)

        out = await func("p", image_data="x.png", system_prompt="s")

        self.assertEqual("ok", out)
        self.assertEqual(1, len(client.calls))
        call = client.calls[0]
        self.assertEqual("p", call["prompt"])
        self.assertEqual(["x.png"], call["images"])
        self.assertEqual("s", call["system_prompt"])


if __name__ == "__main__":
    unittest.main()
