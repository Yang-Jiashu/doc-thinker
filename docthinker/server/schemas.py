from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = False
    session_id: Optional[str] = None
    memory_mode: str = "session"
    retrieval_instruction: Optional[str] = None
    enable_thinking: bool = False
    enable_expanded_matching: bool = True
    expanded_top_k: int = 2
    expanded_min_score: float = 0.2
    enable_image_asset_activation: bool = True
    image_activation_threshold: float = 0.62
    image_activation_top_k: int = 3


class MultiDocumentQueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = False
    session_id: Optional[str] = None


class EntityRelationshipRequest(BaseModel):
    entity_name: str
    entity_type: str
    document_id: str
    properties: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class RelationshipRequest(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    document_id: str
    properties: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    content: str
    source_type: str = "text"
    session_id: Optional[str] = None


class SignalIngestRequest(BaseModel):
    payload: Any
    modality: Optional[str] = None
    source_type: str = "signal"
    source_uri: Optional[str] = None
    timestamp: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

