"""
NeuroAgent - 主入口

启动类人脑智能体

用法：
    python main.py              # 启动交互模式
    python main.py --server     # 启动 API 服务
    python main.py --chat       # 启动聊天模式
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8，避免 emoji 等字符报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 加载 .env（项目根目录）
_root = Path(__file__).resolve().parent
_env = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

# 确保能导入本地模块
sys.path.insert(0, str(_root))

from neuro_core import MemoryEngine
from cognition import CognitiveProcessor
from retrieval import HybridRetriever
from agent import NeuroAgent, SessionManager
from perception.chat import ChatPerceiver
from api.server import create_app


async def interactive_chat(agent: NeuroAgent):
    """交互式聊天"""
    print("=" * 50)
    print("🧠 NeuroAgent - 类人脑智能体")
    print("输入 'quit' 退出, 'save' 保存记忆, 'consolidate' 巩固记忆")
    print("=" * 50)
    
    session_id = None
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == "quit":
                print("💾 保存记忆...")
                agent.save()
                print("👋 再见!")
                break
            
            if query.lower() == "save":
                agent.save()
                print("✅ 记忆已保存")
                continue
            
            if query.lower() == "consolidate":
                print("🔄 执行记忆巩固...")
                result = await agent.consolidate()
                print(f"✅ 巩固完成: {result}")
                continue
            
            # 生成回答
            print("🤔 思考中...")
            result = await agent.respond(query, session_id=session_id)
            
            if not session_id:
                session_id = result.get("session_id")
            
            print(f"\n🤖 Agent: {result['answer']}")
            
            # 显示检索到的记忆
            if result['memories']:
                print(f"\n💭 相关记忆 ({len(result['memories'])}):")
                for i, mem in enumerate(result['memories'][:3], 1):
                    print(f"  {i}. [{mem['source']}] {mem['summary'][:80]}...")
        
        except KeyboardInterrupt:
            print("\n\n💾 保存记忆...")
            agent.save()
            print("👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def create_simple_llm():
    """创建简单的 LLM 函数（无 API Key 时使用）"""
    async def llm_func(prompt: str) -> str:
        return f"[LLM 模拟] 收到: {prompt[:80]}..."
    return llm_func


def create_simple_embedding():
    """创建简单的 embedding 函数（无 API Key 时使用）"""
    import random
    def embedding_func(texts):
        n = len(texts) if isinstance(texts, (list, tuple)) else 1
        return [[random.random() for _ in range(128)] for _ in range(n)]
    return embedding_func


def _create_llm_from_env():
    """从环境变量创建真实 LLM（OpenAI 兼容），无有效 Key 则返回 None。"""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or ""
    api_key = (api_key or "").strip()
    if not api_key or api_key.lower().startswith("your_") or api_key == "EMPTY":
        return None
    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST") or "").strip() or None
    model = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)
    except ImportError:
        return None

    async def llm_func(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM 调用失败: {e}]"
        return "[LLM 无有效回复]"
    return llm_func


def _create_embedding_from_env():
    """从环境变量创建真实 Embedding（OpenAI 兼容），无有效 Key 则返回 None。"""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or os.getenv("EMBEDDING_BINDING_API_KEY") or ""
    api_key = (api_key or "").strip()
    if not api_key or api_key.lower().startswith("your_") or api_key == "EMPTY":
        return None
    base_url = (os.getenv("EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST") or "").strip() or None
    model = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url or None)
    except ImportError:
        return None

    def embedding_func(texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        try:
            r = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in r.data]
        except Exception:
            return None
    return embedding_func


def main():
    parser = argparse.ArgumentParser(description="NeuroAgent - 类人脑智能体")
    parser.add_argument("--server", action="store_true", help="启动 API 服务")
    parser.add_argument("--chat", action="store_true", help="启动聊天模式")
    parser.add_argument("--host", default="0.0.0.0", help="API 服务主机")
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口")
    parser.add_argument("--working-dir", default="./neuro_agent_data", help="工作目录")
    
    args = parser.parse_args()
    
    # 创建 Agent
    print("🚀 初始化 NeuroAgent...")
    
    llm_func = _create_llm_from_env()
    if llm_func is None:
        llm_func = create_simple_llm()
        print("   (未配置 LLM API Key，使用模拟回复；可复制 env.example 为 .env 并填写 LLM_BINDING_API_KEY)")
    embedding_func = _create_embedding_from_env()
    if embedding_func is None:
        embedding_func = create_simple_embedding()
    
    agent = NeuroAgent(
        llm_func=llm_func,
        embedding_func=embedding_func,
        working_dir=args.working_dir,
    )
    
    # 注册感知器
    chat_perceiver = ChatPerceiver(cognitive_processor=agent.cognition)
    agent.register_perceiver("chat", chat_perceiver)
    
    print("✅ Agent 初始化完成")
    print(f"   工作目录: {args.working_dir}")
    print(f"   Episode 数量: {len(agent.memory.episode_store.all_episodes())}")
    
    if args.server:
        # 启动 API 服务
        try:
            import uvicorn
            app = create_app(agent)
            print(f"\n🌐 启动 API 服务: http://{args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            print("❌ 需要安装 uvicorn: pip install uvicorn")
    
    elif args.chat or not args.server:
        # 启动交互式聊天
        asyncio.run(interactive_chat(agent))


if __name__ == "__main__":
    main()
