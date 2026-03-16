import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import Any, List
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphcore.coregraph.utils import EmbeddingFunc

from docthinker import DocThinker, DocThinkerConfig
from docthinker.api_config import APIConfig
from docthinker.cognitive import CognitiveProcessor
from docthinker.providers import load_settings, get_embed_client, get_vlm_client
from docthinker.utils import create_bltcy_rerank_func
from docthinker.services import IngestionService
from docthinker.session_manager import SessionManager
from docthinker.auto_thinking.orchestrator import HybridRAGOrchestrator
from docthinker.auto_thinking.classifier import ComplexityClassifier
from docthinker.auto_thinking.decomposer import QuestionDecomposer
from docthinker.auto_thinking.vlm_client import VLMClient as AutoThinkingVLMClient
from docthinker.hypergraph import HyperGraphRAG

from .state import state
from .memory import save_all_memory_engines
from .routers import health_router, sessions_router, ingest_router, query_router, graph_router


class AsyncModelRouter:
    """Round-robin model router with bounded concurrency and fallback."""

    def __init__(self, client: Any, models: List[str], max_concurrency: int = 32):
        self.client = client
        self.models = [m for m in models if m]
        self.max_concurrency = max(1, int(max_concurrency))
        self._cursor = 0
        self._cursor_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def _next_start_index(self) -> int:
        async with self._cursor_lock:
            if not self.models:
                return 0
            idx = self._cursor % len(self.models)
            self._cursor += 1
            return idx

    async def chat_completion(self, *, messages: List[dict], max_tokens: int = 2048, stream: bool = False) -> Any:
        if not self.models:
            raise ValueError("No LLM models configured for routing")

        start = await self._next_start_index()
        last_err: Exception | None = None
        async with self._semaphore:
            for offset in range(len(self.models)):
                model = self.models[(start + offset) % len(self.models)]
                try:
                    is_gpt5 = str(model or "").lower().startswith("gpt-5")
                    completion_budget = max(int(max_tokens), 600) if is_gpt5 else int(max_tokens)
                    token_kwargs = (
                        {"max_completion_tokens": completion_budget}
                        if is_gpt5
                        else {"max_tokens": completion_budget}
                    )
                    return await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **token_kwargs,
                        stream=stream,
                    )
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        raise RuntimeError("Model router failed without explicit exception")


def _make_vision_model_func(vlm_client: Any):
    """Build a vision model adapter that supports both image_data and OpenAI-style messages."""

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        image_data: Any = None,
        messages: List[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        images = None
        if image_data:
            if isinstance(image_data, (list, tuple)):
                images = list(image_data)
            else:
                images = [image_data]

        return await vlm_client.generate(
            prompt or "",
            images=images,
            system_prompt=system_prompt,
            max_tokens=int(kwargs.get("max_tokens", 4096)),
            temperature=float(kwargs.get("temperature", 0.2)),
            extra_messages=messages,
        )

    return vision_model_func


def _cleanup_global_graphcore_artifacts(workdir: str) -> None:
    """Remove root-level GraphCore artifacts so knowledge remains session-scoped."""
    root = Path(workdir)
    if not root.exists():
        return

    removed: list[str] = []
    for pattern in ("graph_*.graphml", "vdb_*.json", "kv_store_*.json"):
        for path in root.glob(pattern):
            try:
                path.unlink()
                removed.append(path.name)
            except Exception:
                continue

    if removed:
        print(
            "INFO: Removed global GraphCore artifacts for session isolation: "
            + ", ".join(sorted(removed))
        )


def _create_rag_config() -> DocThinkerConfig:
    return DocThinkerConfig(
        working_dir=state.settings.workdir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )


async def _get_embedding_func() -> Any:
    embed_client = get_embed_client(state.settings)

    async def embedding_func_impl(texts: List[str]) -> Any:
        resp = await embed_client.embeddings.create(
            model=state.settings.embed_model,
            input=texts,
        )
        vectors: List[List[float]] = []
        for item in resp.data:
            emb = getattr(item, "embedding", None)
            if isinstance(emb, list):
                vectors.append(emb)
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=state.settings.embed_dim,
        max_token_size=8192,
        func=embedding_func_impl,
    )


async def _get_llm_model_func() -> Any:
    vlm_client = get_vlm_client(state.settings)
    models = state.settings.llm_models or [state.settings.llm_model]
    model_router = AsyncModelRouter(
        client=vlm_client,
        models=models,
        max_concurrency=state.settings.llm_router_max_concurrency,
    )

    async def chat_complete(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await model_router.chat_completion(messages=messages, max_tokens=2048, stream=False)
        if not hasattr(resp, "choices") or not resp.choices:
            return str(resp)
        return resp.choices[0].message.content

    return chat_complete


async def _initialize_rag() -> DocThinker:
    embedding_func = await _get_embedding_func()
    chat_complete = await _get_llm_model_func()
    rerank_func = create_bltcy_rerank_func(
        api_key=state.settings.rerank_api_key,
        base_url=state.settings.rerank_base_url,
        model_name=state.settings.rerank_model,
    )
    config = _create_rag_config()
    vlm_client = AutoThinkingVLMClient(
        api_key=state.settings.llm_api_key,
        api_base=state.settings.vlm_base_url,
        model=state.settings.vlm_model,
    )
    state.vision_vlm_client = vlm_client
    vision_model_func = _make_vision_model_func(vlm_client)

    graphcore_kwargs = {
        "llm_model_max_async": state.settings.graphcore_llm_max_async,
        "embedding_func_max_async": state.settings.graphcore_embedding_max_async,
        "max_parallel_insert": state.settings.graphcore_max_parallel_insert,
    }
    if rerank_func:
        graphcore_kwargs["rerank_model_func"] = rerank_func

    return DocThinker(
        config=config,
        llm_model_func=chat_complete,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        graphcore_kwargs=graphcore_kwargs,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.settings = load_settings()
    state.api_config = APIConfig()
    Path(state.settings.workdir).mkdir(parents=True, exist_ok=True)
    state.session_manager = SessionManager(base_storage_path=state.settings.workdir)
    _cleanup_global_graphcore_artifacts(state.settings.workdir)

    state.rag_instance = await _initialize_rag()
    # Session isolation: do not initialize global GraphCore.
    state.kg_entity_ids = set()

    state.cognitive_processor = CognitiveProcessor(
        llm_func=state.rag_instance.llm_model_func,
        embedding_func=state.rag_instance.embedding_func,
        knowledge_graph=None,
    )
    try:
        from neuro_memory import MemoryEngine

        _embed = getattr(state.rag_instance.embedding_func, "func", state.rag_instance.embedding_func)

        async def _neuro_embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            out = await _embed(texts)
            if hasattr(out, "tolist"):
                return out.tolist()
            return list(out) if out else []

        def _kg_entity_resolver(entity_id: str) -> bool:
            return entity_id in getattr(state, "kg_entity_ids", set())

        def _session_memory_factory(session_id: str):
            if not state.session_manager:
                raise ValueError("Session manager not initialized")
            talk_dir = state.session_manager.get_session_talk_dir(session_id)
            engine = MemoryEngine(
                embedding_func=_neuro_embed,
                llm_func=state.rag_instance.llm_model_func,
                working_dir=str(talk_dir),
                kg_entity_resolver=_kg_entity_resolver,
            )
            engine.load()
            return engine

        state.memory_engine_factory = _session_memory_factory
        state.memory_engines = {}
        state.memory_engine = None
        print("INFO: Neuro memory engine initialized in session-scoped mode.")
    except Exception as e:
        print(f"WARNING: Neuro memory engine not initialized: {e}")
        state.memory_engine = None
        state.memory_engine_factory = None
        state.memory_engines = {}

    state.ingestion_service = IngestionService(
        rag_global=state.rag_instance,
        session_manager=state.session_manager,
        create_rag_config=_create_rag_config,
        get_llm_model_func=_get_llm_model_func,
        get_embedding_func=_get_embedding_func,
    )

    # Initialize Auto-Thinking Orchestrator
    try:
        at_client = AutoThinkingVLMClient(
            api_key=state.settings.llm_api_key,
            api_base=state.settings.vlm_base_url,
            model=state.settings.vlm_model,
        )
        state.auto_thinking_vlm_client = at_client
        classifier = ComplexityClassifier(vlm_client=at_client)
        decomposer = QuestionDecomposer(vlm_client=at_client)

        # Initialize HyperGraphRAG
        hyper_system = HyperGraphRAG(
            working_dir=state.settings.workdir,
            llm_model_func=state.rag_instance.llm_model_func,
            embedding_func=state.rag_instance.embedding_func,
            vlm_client=at_client,
            graph_construction_mode=state.rag_instance.config.graph_construction_mode,
            spacy_model=state.rag_instance.config.spacy_model,
        )
        
        state.orchestrator = HybridRAGOrchestrator(
            rag_system=state.rag_instance,
            hyper_system=hyper_system,
            classifier=classifier,
            vlm_client=at_client,
            decomposer=decomposer,
            enable_multi_step=True,
            sync_mode="eager", # Sync immediately for demo purposes
        )
        print("INFO: Auto-Thinking Orchestrator (Hybrid) initialized.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Auto-Thinking Orchestrator: {e}")

    yield

    # Save tri-graph state on shutdown
    for sid, mgr in getattr(state, "tri_graph_managers", {}).items():
        try:
            mgr.save_all()
        except Exception:
            pass

    save_all_memory_engines()
    for client_attr in ("auto_thinking_vlm_client", "vision_vlm_client"):
        client = getattr(state, client_attr, None)
        if client is not None and hasattr(client, "close"):
            try:
                await client.close()
            except Exception:
                pass
    if state.rag_instance:
        await state.rag_instance.finalize_storages()


def create_app() -> FastAPI:
    api_config = APIConfig()
    app = FastAPI(
        title="Multi-Document Enhanced RAG API",
        description="API service for multi-document enhanced RAG system using knowledge graph",
        version="1.0.0",
        lifespan=lifespan,
    )

    if api_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for r in [health_router, sessions_router, ingest_router, query_router, graph_router]:
        app.include_router(r, prefix=api_config.api_prefix)

    return app


app = create_app()

