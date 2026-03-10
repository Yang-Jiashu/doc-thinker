"""
快速测试千问 API 是否可用。
用法: python scripts/test_qwen_quick.py
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ.get("LLM_BINDING_API_KEY", "")
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def main():
    import openai
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE)

    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print()

    # 测试 LLM
    models_to_test = [
        ("qwen-plus", "chat"),
        ("qwen-turbo", "chat"),
        ("qwen-vl-max", "chat"),
        ("qwen-vl-plus", "chat"),
        ("text-embedding-v3", "embed"),
    ]

    for model, mtype in models_to_test:
        try:
            if mtype == "chat":
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "说一个字"}],
                    max_tokens=10,
                )
                text = r.choices[0].message.content.strip()
                print(f"  [OK]   {model:30s} -> {text[:30]}")
            else:
                r = await client.embeddings.create(model=model, input=["测试"])
                dim = len(r.data[0].embedding)
                print(f"  [OK]   {model:30s} -> dim={dim}")
        except Exception as e:
            msg = str(e)
            if "Unpurchased" in msg or "AccessDenied" in msg:
                print(f"  [FAIL] {model:30s} -> not activated")
            elif "model_not" in msg:
                print(f"  [FAIL] {model:30s} -> model name not supported")
            else:
                print(f"  [FAIL] {model:30s} -> {msg[:60]}")

    print("\n如果看到 '未开通'，请访问 https://bailian.console.aliyun.com/?tab=model#/model-market 开通对应模型")


if __name__ == "__main__":
    asyncio.run(main())
