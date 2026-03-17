import os
from pydantic import BaseModel, Field

# 项目根目录（doc-thinker），确保 workdir 不随 CWD 变化
_DOC_THINKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_WORKDIR = os.path.join(_DOC_THINKER_ROOT, "data", "_system")


class AppSettings(BaseModel):
    llm_api_key: str
    vlm_base_url: str
    llm_model: str
    llm_models: list[str] = Field(default_factory=list)
    llm_router_max_concurrency: int = 32
    vlm_model: str

    keyword_llm_model: str = "qwen-turbo"
    """Lightweight model for keyword extraction (no extended thinking)."""

    entity_extraction_llm_model: str = "qwen-turbo"
    """Lightweight model for entity extraction during ingestion (no extended thinking)."""

    embed_api_key: str
    embed_base_url: str
    embed_model: str
    embed_dim: int = 1024
    rerank_api_key: str
    rerank_base_url: str
    rerank_model: str
    graphcore_llm_max_async: int = 16
    graphcore_embedding_max_async: int = 16
    graphcore_max_parallel_insert: int = 8
    graphcore_max_gleaning: int = 0
    extraction_llm_model: str = ""

    workdir: str = "./data/_system"
    timeout_seconds: int = 3600


def load_settings() -> AppSettings:
    _dashscope_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    llm_api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    embed_api_key = os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or llm_api_key or "EMPTY"
    llm_model = os.getenv("LLM_MODEL") or "qwen-plus"
    llm_models_raw = os.getenv("LLM_MODELS") or llm_model
    llm_models = [x.strip() for x in llm_models_raw.split(",") if x.strip()]
    openai_base = os.getenv("OPENAI_BASE_URL") or _dashscope_base

    return AppSettings(
        llm_api_key=llm_api_key,
        vlm_base_url=(
            os.getenv("LLM_VLM_HOST")
            or os.getenv("LLM_BINDING_HOST")
            or openai_base
        ),
        llm_model=llm_model,
        llm_models=llm_models,
        llm_router_max_concurrency=int(os.getenv("LLM_ROUTER_MAX_CONCURRENCY") or 32),
        vlm_model=os.getenv("VLM_MODEL") or "qwen-vl-max",
        keyword_llm_model=os.getenv("KEYWORD_LLM_MODEL") or "qwen-turbo",
        entity_extraction_llm_model=os.getenv("ENTITY_EXTRACTION_LLM_MODEL") or "qwen-turbo",
        embed_api_key=embed_api_key,
        embed_base_url=(
            os.getenv("LLM_EMBED_HOST")
            or os.getenv("EMBEDDING_BINDING_HOST")
            or openai_base
        ),
        embed_model=os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "text-embedding-v3",
        embed_dim=int(os.getenv("EMBEDDING_DIM") or os.getenv("EMBED_DIM") or 1024),
        rerank_api_key=os.getenv("RERANK_API_KEY") or "",
        rerank_base_url=(
            os.getenv("RERANK_HOST")
            or os.getenv("EMBEDDING_BINDING_HOST")
            or os.getenv("LLM_BINDING_HOST")
            or openai_base
        ),
        rerank_model=os.getenv("RERANK_MODEL") or "",
        graphcore_llm_max_async=int(os.getenv("GRAPHCORE_LLM_MAX_ASYNC") or os.getenv("MAX_ASYNC") or 16),
        graphcore_embedding_max_async=int(
            os.getenv("GRAPHCORE_EMBEDDING_MAX_ASYNC") or os.getenv("EMBEDDING_FUNC_MAX_ASYNC") or 16
        ),
        graphcore_max_parallel_insert=int(
            os.getenv("GRAPHCORE_MAX_PARALLEL_INSERT") or os.getenv("MAX_PARALLEL_INSERT") or 8
        ),
        graphcore_max_gleaning=int(os.getenv("MAX_GLEANING") or 0),
        extraction_llm_model=os.getenv("EXTRACTION_LLM_MODEL") or "",
        workdir=os.getenv("RAG_WORKDIR") or _DEFAULT_WORKDIR,
        timeout_seconds=int(os.getenv("TIMEOUT") or 3600),
    )

