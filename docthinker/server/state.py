import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Optional, Any, Set, Dict, Callable

from docthinker.session_manager import SessionManager
from docthinker.cognitive import CognitiveProcessor
from docthinker.services import IngestionService
from docthinker.providers import AppSettings

_log = logging.getLogger("docthinker.state")


@dataclass
class AppState:
    settings: Optional[AppSettings] = None
    api_config: Optional[Any] = None
    session_manager: Optional[SessionManager] = None

    rag_instance: Optional[Any] = None
    cognitive_processor: Optional[CognitiveProcessor] = None
    ingestion_service: Optional[IngestionService] = None
    orchestrator: Optional[Any] = None
    memory_engine: Optional[Any] = None  # deprecated: retained for compatibility
    memory_engine_factory: Optional[Callable[[str], Any]] = None
    memory_engines: Dict[str, Any] = field(default_factory=dict)
    memory_engine_lock: Any = field(default_factory=RLock)
    kg_entity_ids: Set[str] = field(default_factory=set)

    expanded_node_managers: Dict[str, Any] = field(default_factory=dict)
    expanded_node_lock: Any = field(default_factory=RLock)

    # C1: Tri-Graph Architecture — session-scoped TriGraphManager instances
    tri_graph_managers: Dict[str, Any] = field(default_factory=dict)
    tri_graph_lock: Any = field(default_factory=RLock)


state = AppState()

def get_tri_graph_manager(session_id: Optional[str]) -> Optional[Any]:
    """Resolve or create a session-scoped TriGraphManager.

    Shared helper used by both ingest and graph routers.
    Returns None (with INFO logging) if prerequisites are missing.
    """
    if not session_id or not state.session_manager:
        return None
    session = state.session_manager.get_session(session_id)
    if not session:
        _log.info("[tri_graph] session not found: %s", session_id)
        return None
    metadata = session.get("metadata") or {}
    knowledge_dir = metadata.get("knowledge_dir") or session.get("knowledge_dir")
    if not knowledge_dir:
        _log.info("[tri_graph] no knowledge_dir for session %s", session_id)
        return None
    with state.tri_graph_lock:
        mgr = state.tri_graph_managers.get(session_id)
        if mgr is not None:
            return mgr
        # Prefer the fast entity-extraction model (qwen-turbo) for causal
        # extraction; fall back to the slower query model if unavailable.
        llm_func = (
            getattr(state.rag_instance, "entity_extraction_llm_model_func", None)
            or (state.rag_instance.llm_model_func if state.rag_instance else None)
        )
        if not llm_func:
            _log.info("[tri_graph] no llm_func available for session %s", session_id)
            return None
        from docthinker.trigraph import TriGraphManager
        mgr = TriGraphManager(knowledge_dir=Path(knowledge_dir), llm_func=llm_func)
        state.tri_graph_managers[session_id] = mgr
        return mgr

