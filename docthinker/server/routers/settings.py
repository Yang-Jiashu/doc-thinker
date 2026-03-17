"""Settings API — read/write runtime model & API configuration."""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..state import state

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    llm_api_key_masked: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_models: list[str] = []
    vlm_model: str = ""
    keyword_llm_model: str = ""
    entity_extraction_llm_model: str = ""

    embed_api_key_masked: str = ""
    embed_base_url: str = ""
    embed_model: str = ""
    embed_dim: int = 1024

    rerank_api_key_masked: str = ""
    rerank_base_url: str = ""
    rerank_model: str = ""

    llm_max_async: int = 16
    embedding_max_async: int = 16
    max_parallel_insert: int = 8
    llm_router_max_concurrency: int = 32

    workdir: str = ""


class SettingsUpdate(BaseModel):
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    vlm_model: Optional[str] = None
    keyword_llm_model: Optional[str] = None
    entity_extraction_llm_model: Optional[str] = None

    embed_api_key: Optional[str] = None
    embed_base_url: Optional[str] = None
    embed_model: Optional[str] = None
    embed_dim: Optional[int] = None

    rerank_api_key: Optional[str] = None
    rerank_base_url: Optional[str] = None
    rerank_model: Optional[str] = None

    llm_max_async: Optional[int] = None
    embedding_max_async: Optional[int] = None
    max_parallel_insert: Optional[int] = None
    llm_router_max_concurrency: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    if not key or key == "EMPTY":
        return ""
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:4] + "****" + key[-4:]


_ENV_MAP: dict[str, str] = {
    "llm_api_key": "LLM_BINDING_API_KEY",
    "llm_base_url": "LLM_BINDING_HOST",
    "llm_model": "LLM_MODEL",
    "vlm_model": "VLM_MODEL",
    "keyword_llm_model": "KEYWORD_LLM_MODEL",
    "entity_extraction_llm_model": "ENTITY_EXTRACTION_LLM_MODEL",
    "embed_api_key": "EMBEDDING_BINDING_API_KEY",
    "embed_base_url": "EMBEDDING_BINDING_HOST",
    "embed_model": "EMBEDDING_MODEL",
    "embed_dim": "EMBEDDING_DIM",
    "rerank_api_key": "RERANK_API_KEY",
    "rerank_base_url": "RERANK_HOST",
    "rerank_model": "RERANK_MODEL",
    "llm_max_async": "MAX_ASYNC",
    "embedding_max_async": "EMBEDDING_FUNC_MAX_ASYNC",
    "max_parallel_insert": "MAX_PARALLEL_INSERT",
    "llm_router_max_concurrency": "LLM_ROUTER_MAX_CONCURRENCY",
}


def _persist_to_dotenv(updates: dict[str, str]) -> None:
    """Append or update key=value pairs in the project .env file."""
    env_path = _PROJECT_ROOT / ".env"
    existing: dict[str, str] = {}
    lines: list[str] = []

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=", 1)[0].strip()
                existing[k] = line
            lines.append(line)

    for env_key, env_val in updates.items():
        new_line = f"{env_key}={env_val}"
        if env_key in existing:
            idx = lines.index(existing[env_key])
            lines[idx] = new_line
        else:
            lines.append(new_line)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/settings")
async def get_settings() -> SettingsResponse:
    s = state.settings
    if not s:
        return SettingsResponse()

    return SettingsResponse(
        llm_api_key_masked=_mask_key(s.llm_api_key),
        llm_base_url=s.vlm_base_url,
        llm_model=s.llm_model,
        llm_models=s.llm_models,
        vlm_model=s.vlm_model,
        keyword_llm_model=s.keyword_llm_model,
        entity_extraction_llm_model=s.entity_extraction_llm_model,

        embed_api_key_masked=_mask_key(s.embed_api_key),
        embed_base_url=s.embed_base_url,
        embed_model=s.embed_model,
        embed_dim=s.embed_dim,

        rerank_api_key_masked=_mask_key(s.rerank_api_key),
        rerank_base_url=s.rerank_base_url,
        rerank_model=s.rerank_model,

        llm_max_async=s.graphcore_llm_max_async,
        embedding_max_async=s.graphcore_embedding_max_async,
        max_parallel_insert=s.graphcore_max_parallel_insert,
        llm_router_max_concurrency=s.llm_router_max_concurrency,

        workdir=s.workdir,
    )


@router.post("/settings")
async def update_settings(body: SettingsUpdate) -> dict:
    s = state.settings
    if not s:
        return {"success": False, "message": "Settings not initialized"}

    env_updates: dict[str, str] = {}
    changed: list[str] = []

    for field_name, env_key in _ENV_MAP.items():
        new_val = getattr(body, field_name, None)
        if new_val is None:
            continue

        str_val = str(new_val)
        os.environ[env_key] = str_val
        env_updates[env_key] = str_val
        changed.append(field_name)

        if hasattr(s, field_name):
            if isinstance(getattr(s, field_name), int):
                setattr(s, field_name, int(new_val))
            else:
                setattr(s, field_name, str_val)

    if body.llm_api_key is not None:
        s.llm_api_key = body.llm_api_key
    if body.embed_api_key is not None:
        s.embed_api_key = body.embed_api_key
    if body.rerank_api_key is not None:
        s.rerank_api_key = body.rerank_api_key
    if body.llm_base_url is not None:
        s.vlm_base_url = body.llm_base_url
    if body.embed_base_url is not None:
        s.embed_base_url = body.embed_base_url
    if body.rerank_base_url is not None:
        s.rerank_base_url = body.rerank_base_url

    if body.llm_model is not None:
        s.llm_model = body.llm_model
        s.llm_models = [m.strip() for m in body.llm_model.split(",") if m.strip()]

    if env_updates:
        try:
            _persist_to_dotenv(env_updates)
        except Exception:
            pass

    return {
        "success": True,
        "message": f"Updated: {', '.join(changed)}" if changed else "No changes",
        "changed": changed,
        "note": "Changes applied to runtime. Restart server for full effect on model connections.",
    }
