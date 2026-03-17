from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, Any, List


@dataclass(frozen=True)
class IngestionService:
    rag_global: Any
    session_manager: Any
    create_rag_config: Callable[[], Any]
    get_llm_model_func: Callable[[], Awaitable[Any]]
    get_embedding_func: Callable[[], Awaitable[Any]]

    async def _ensure_ready(self, rag: Any) -> None:
        if not getattr(rag, "graphcore", None):
            await rag._ensure_graphcore_initialized()

    async def _insert_text(self, rag: Any, text: str, file_path: str | None = None) -> None:
        await self._ensure_ready(rag)
        graphcore = rag.graphcore
        if hasattr(graphcore, "ainsert"):
            await graphcore.ainsert(text, file_paths=file_path)
        else:
            await graphcore.insert(text)

    async def _get_session_rag(self, session_id: str) -> Any:
        config = self.create_rag_config()
        graphcore_kwargs = getattr(self.rag_global, "graphcore_kwargs", {})
        try:
            session_rag = self.session_manager.get_session_rag(
                session_id, config, graphcore_kwargs
            )
        except TypeError:
            session_rag = self.session_manager.get_session_rag(session_id, config)
        session_rag.llm_model_func = await self.get_llm_model_func()
        session_rag.keyword_llm_model_func = getattr(self.rag_global, "keyword_llm_model_func", None)
        session_rag.entity_extraction_llm_model_func = getattr(self.rag_global, "entity_extraction_llm_model_func", None)
        session_rag.embedding_func = await self.get_embedding_func()
        session_rag.vision_model_func = getattr(self.rag_global, "vision_model_func", None)
        await session_rag._ensure_graphcore_initialized()
        return session_rag

    async def _resolve_target_rag(self, session_id: Optional[str]) -> Any:
        if not session_id:
            raise ValueError("session_id is required")
        return await self._get_session_rag(session_id)

    async def ingest_text(
        self, text: str, session_id: Optional[str] = None, file_path: str | None = None,
    ) -> None:
        """Ingest text into a session graph."""
        target_rag = await self._resolve_target_rag(session_id)
        await self._insert_text(target_rag, text, file_path=file_path)

    async def ingest_folder(self, folder_path: str, session_id: Optional[str] = None) -> None:
        """Ingest folder into a session graph."""
        target_rag = await self._resolve_target_rag(session_id)
        await target_rag.process_folder_complete(folder_path)

    async def ingest_files(self, file_paths: List[str], session_id: Optional[str] = None) -> None:
        """Ingest files into a session graph."""
        target_rag = await self._resolve_target_rag(session_id)
        for file_path in file_paths:
            await target_rag.process_document_complete(file_path)
