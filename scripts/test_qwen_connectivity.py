"""Quick connectivity test for all Qwen DashScope models."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ.get("LLM_BINDING_API_KEY", "")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RERANK_BASE = "https://dashscope.aliyuncs.com/compatible-api/v1"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    tag = PASS if ok else FAIL
    print(f"  {tag} {name}")
    if detail:
        print(f"        {detail[:200]}")


async def test_llm():
    print("\n--- Test 1: LLM (qwen-plus) ---")
    import openai
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=DASHSCOPE_BASE)
    try:
        t0 = time.time()
        resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "用一句话介绍你自己"}],
            max_tokens=100,
        )
        text = resp.choices[0].message.content
        record("LLM qwen-plus", True, f"{time.time()-t0:.1f}s — {text[:80]}")
    except Exception as e:
        record("LLM qwen-plus", False, str(e))


async def test_vlm():
    print("\n--- Test 2: VLM (qwen-vl-max) ---")
    import aiohttp
    url = f"{DASHSCOPE_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "qwen-vl-max",
        "messages": [{"role": "user", "content": "用一句话介绍你自己，你是什么模型"}],
        "max_tokens": 100,
    }
    try:
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    record("VLM qwen-vl-max", False, f"HTTP {resp.status}: {body[:150]}")
                    return
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]
                record("VLM qwen-vl-max", True, f"{time.time()-t0:.1f}s — {text[:80]}")
    except Exception as e:
        record("VLM qwen-vl-max", False, str(e))


async def test_embedding():
    print("\n--- Test 3: Embedding (text-embedding-v3) ---")
    import openai
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=DASHSCOPE_BASE)
    try:
        t0 = time.time()
        resp = await client.embeddings.create(
            model="text-embedding-v3",
            input=["轮履带变换式侦察搜救机器人", "知识图谱构建"],
        )
        dims = len(resp.data[0].embedding)
        count = len(resp.data)
        record("Embedding text-embedding-v3", True,
               f"{time.time()-t0:.1f}s — {count} vectors, dim={dims}")
    except Exception as e:
        record("Embedding text-embedding-v3", False, str(e))


async def test_rerank():
    print("\n--- Test 4: Rerank (gte-rerank) ---")
    import aiohttp
    url = f"{RERANK_BASE}/reranks"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gte-rerank",
        "query": "机器人的技术原理是什么",
        "documents": [
            "该机器人采用摩擦驱动模块和弹性轮履带一体化设计",
            "今天天气真好适合出去玩",
            "通过曲柄连杆机构实现轮履带的快速切换",
        ],
        "top_n": 2,
    }
    try:
        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    record("Rerank gte-rerank", False, f"HTTP {resp.status}: {body[:150]}")
                    return
                data = await resp.json()
                top_results = data.get("results", [])
                record("Rerank gte-rerank", True,
                       f"{time.time()-t0:.1f}s — {len(top_results)} results, "
                       f"top index={top_results[0]['index'] if top_results else '?'}")
    except Exception as e:
        record("Rerank gte-rerank", False, str(e))


async def main():
    print("=" * 60)
    print("Qwen DashScope Model Connectivity Test")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}" if len(API_KEY) > 12 else "NOT SET")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not API_KEY:
        print("\nERROR: No API key found. Set LLM_BINDING_API_KEY in .env")
        return 1

    await test_llm()
    await test_vlm()
    await test_embedding()
    await test_rerank()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)}")
    if failed:
        print("\nFailed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
