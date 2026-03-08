import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _truncate(text: Any, max_len: int = 3000) -> str:
    value = str(text or "")
    if len(value) <= max_len:
        return value
    return value[:max_len] + "\n... [TRUNCATED]"


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _dt_to_utc(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(timezone.utc)


def _dt_to_iso_utc(value: Any) -> Optional[str]:
    dt = _dt_to_utc(value)
    return dt.isoformat() if dt else None


def _delta_seconds(later: Any, earlier: Any) -> Optional[float]:
    left = _dt_to_utc(later)
    right = _dt_to_utc(earlier)
    if not left or not right:
        return None
    return round((left - right).total_seconds(), 3)


def _resolve_session_dir(session_dir: Optional[str], session_id: Optional[str]) -> Path:
    if session_dir:
        path = Path(session_dir)
    elif session_id:
        path = REPO_ROOT / "data" / session_id
    else:
        raise ValueError("Provide --session-dir or --session-id")
    if not path.exists():
        raise FileNotFoundError(f"Session directory not found: {path}")
    return path.resolve()


def _build_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current_user: Optional[dict[str, Any]] = None
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role == "user":
            current_user = item
        elif role == "assistant" and current_user:
            turns.append({"user": current_user, "assistant": item})
            current_user = None
    return turns


def _select_turn(turns: list[dict[str, Any]], turn_index: int) -> dict[str, Any]:
    if turn_index < 1 or turn_index > len(turns):
        raise IndexError(f"Turn {turn_index} out of range; total turns={len(turns)}")
    selected = dict(turns[turn_index - 1])
    selected["turn_index"] = turn_index
    return selected


def _pick_content_file(session_dir: Path) -> Optional[Path]:
    content_dir = session_dir / "content"
    if not content_dir.exists():
        return None
    candidates = sorted([p for p in content_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    return candidates[0] if candidates else None


def _find_source_doc_id(doc_status: dict[str, Any], full_docs: dict[str, Any]) -> Optional[str]:
    candidates: list[tuple[datetime, str]] = []
    for doc_id, meta in doc_status.items():
        content = str((full_docs.get(doc_id) or {}).get("content") or "")
        if "User Question:" in content:
            continue
        created = _dt_to_utc(meta.get("created_at") or (full_docs.get(doc_id) or {}).get("create_time"))
        if created:
            candidates.append((created, doc_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_chat_doc_id(full_docs: dict[str, Any], question: str) -> Optional[str]:
    normalized = _normalize_text(question)
    for doc_id, meta in full_docs.items():
        content = str(meta.get("content") or "")
        if "User Question:" not in content:
            continue
        if normalized and normalized in _normalize_text(content):
            return doc_id
    return None


def _build_doc_timeline(doc_status: dict[str, Any], full_docs: dict[str, Any], question: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    question_norm = _normalize_text(question)
    for doc_id, meta in doc_status.items():
        full_doc = full_docs.get(doc_id) or {}
        content = str(full_doc.get("content") or "")
        items.append(
            {
                "doc_id": doc_id,
                "kind": "chat_turn" if "User Question:" in content else "source",
                "matched_selected_turn": bool(question_norm and question_norm in _normalize_text(content)),
                "status": meta.get("status"),
                "created_at_utc": _dt_to_iso_utc(meta.get("created_at") or full_doc.get("create_time")),
                "updated_at_utc": _dt_to_iso_utc(meta.get("updated_at") or full_doc.get("update_time")),
                "processing_start_utc": _dt_to_iso_utc((meta.get("metadata") or {}).get("processing_start_time")),
                "processing_end_utc": _dt_to_iso_utc((meta.get("metadata") or {}).get("processing_end_time")),
                "chunks": meta.get("chunks_list") or [],
                "content_preview": _truncate(content or meta.get("content_summary") or "", 500),
            }
        )
    items.sort(key=lambda item: _dt_to_utc(item["created_at_utc"]) or datetime.max.replace(tzinfo=timezone.utc))
    return items


def _parse_extract_payload(payload: Any) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for raw_line in str(payload or "").splitlines():
        line = raw_line.strip()
        if not line or line == "<|COMPLETE|>":
            continue
        parts = line.split("<|#|>")
        if line.startswith("entity<|#|>") and len(parts) >= 4:
            entities.append({"name": parts[1], "entity_type": parts[2], "description": parts[3]})
        elif line.startswith("relation<|#|>") and len(parts) >= 5:
            relations.append({"source": parts[1], "target": parts[2], "keywords": parts[3], "description": parts[4]})
    return {"entities": entities, "relations": relations}


def _summarize_cache_entry(cache_id: str, entry: dict[str, Any], parse_extract: bool = False) -> dict[str, Any]:
    item = {
        "cache_id": cache_id,
        "cache_type": entry.get("cache_type"),
        "chunk_id": entry.get("chunk_id"),
        "create_time_utc": _dt_to_iso_utc(entry.get("create_time")),
        "update_time_utc": _dt_to_iso_utc(entry.get("update_time")),
        "original_prompt_excerpt": _truncate(entry.get("original_prompt") or "", 5000),
        "return_excerpt": _truncate(entry.get("return") or "", 5000),
        "queryparam": entry.get("queryparam"),
    }
    if parse_extract:
        item["parsed_extract"] = _parse_extract_payload(entry.get("return"))
    return item


def _cache_entries_from_ids(cache_store: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cache_id in ids or []:
        entry = cache_store.get(cache_id)
        if isinstance(entry, dict):
            items.append(_summarize_cache_entry(cache_id, entry, parse_extract=entry.get("cache_type") == "extract"))
    return items


def _find_cache_entries_by_prompt(
    cache_store: dict[str, Any],
    prompt_text: str,
    cache_types: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    normalized = _normalize_text(prompt_text)
    results: list[dict[str, Any]] = []
    for cache_id, entry in cache_store.items():
        if not isinstance(entry, dict):
            continue
        if cache_types and entry.get("cache_type") not in cache_types:
            continue
        if normalized and normalized in _normalize_text(entry.get("original_prompt") or ""):
            results.append(_summarize_cache_entry(cache_id, entry, parse_extract=entry.get("cache_type") == "extract"))
    results.sort(key=lambda item: _dt_to_utc(item.get("create_time_utc")) or datetime.min.replace(tzinfo=timezone.utc))
    return results


def _entities_for_chunk(entity_chunks: dict[str, Any], chunk_id: Optional[str]) -> list[str]:
    if not chunk_id:
        return []
    names: list[str] = []
    for name, meta in entity_chunks.items():
        if chunk_id in (meta.get("chunk_ids") or []):
            names.append(name)
    return sorted(names)


def _relations_for_doc(full_relations: dict[str, Any], doc_id: Optional[str]) -> list[list[str]]:
    if not doc_id:
        return []
    return (full_relations.get(doc_id) or {}).get("relation_pairs") or []


def _load_code_snapshots(code_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(code_dir.glob("*.json")):
        if "_trace_pipeline" in path.name:
            continue
        payload = _safe_read_json(path)
        if isinstance(payload, dict) and ("nodes" in payload or "relationships" in payload):
            items.append({"path": path, "payload": payload})
    return items


def _pick_source_snapshot(snapshots: list[dict[str, Any]], content_file: Optional[Path]) -> Optional[dict[str, Any]]:
    if content_file:
        for item in snapshots:
            if item["payload"].get("source_name") == content_file.name:
                return item
    for item in snapshots:
        if item["payload"].get("source_type") == "file":
            return item
    return snapshots[0] if snapshots else None


def _find_alias_or_name_not_in_snapshot(snapshot: dict[str, Any], source_text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    source_norm = _normalize_text(source_text)
    for node in snapshot.get("nodes") or []:
        label = str(node.get("label") or "").strip()
        if label and _normalize_text(label) not in source_norm:
            hits.append({"kind": "label", "entity": label, "reason": "node_label_not_in_source_text"})
        for alias in node.get("aliases") or []:
            alias_text = str(alias or "").strip()
            if alias_text and _normalize_text(alias_text) not in source_norm:
                hits.append(
                    {
                        "kind": "alias",
                        "entity": label or "<unknown>",
                        "value": alias_text,
                        "reason": "alias_not_in_source_text",
                    }
                )
    return hits


def _pick_snapshot_relations(snapshot: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    return list(snapshot.get("relationships") or [])[:limit]


def _find_knowledge_graph_entities(knowledge_graph: dict[str, Any], interesting_names: set[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for _, entity in (knowledge_graph.get("entities") or {}).items():
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "")
        aliases = [str(x) for x in entity.get("aliases") or []]
        matched = [value for value in [name, *aliases] if value in interesting_names]
        if matched:
            items.append(
                {
                    "name": name,
                    "aliases": aliases,
                    "matched_terms": matched,
                    "description": entity.get("description"),
                    "document_ids": entity.get("document_ids") or [],
                    "sources": entity.get("sources") or [],
                }
            )
    return items


def _find_knowledge_graph_relations(knowledge_graph: dict[str, Any], interesting_names: set[str]) -> list[dict[str, Any]]:
    id_to_name = {
        entity_id: str(entity.get("name") or "")
        for entity_id, entity in (knowledge_graph.get("entities") or {}).items()
        if isinstance(entity, dict)
    }
    items: list[dict[str, Any]] = []
    for _, relation in (knowledge_graph.get("relationships") or {}).items():
        if not isinstance(relation, dict):
            continue
        src = id_to_name.get(str(relation.get("source_id") or ""), "")
        tgt = id_to_name.get(str(relation.get("target_id") or ""), "")
        if src in interesting_names or tgt in interesting_names:
            items.append(
                {
                    "source": src,
                    "target": tgt,
                    "type": relation.get("type"),
                    "description": relation.get("description"),
                    "document_ids": relation.get("document_ids") or [],
                    "sources": relation.get("sources") or [],
                }
            )
    return items


def _graphml_hits(graph_path: Path, interesting_names: set[str]) -> dict[str, Any]:
    text = _safe_read_text(graph_path)
    if not text:
        return {"nodes": [], "edges": []}
    nodes = [name for name in sorted(interesting_names) if name and f'<node id="{name}">' in text]
    edges = []
    for source, target in re.findall(r'<edge source="([^"]+)" target="([^"]+)">', text):
        if source in interesting_names or target in interesting_names:
            edges.append({"source": source, "target": target})
    return {"nodes": nodes, "edges": edges}


def _find_line_hits(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    text = _safe_read_text(path)
    if not text:
        return []
    lines = text.splitlines()
    hits: list[dict[str, Any]] = []
    for pattern in patterns:
        for idx, line in enumerate(lines, start=1):
            if pattern in line:
                hits.append({"pattern": pattern, "line": idx, "excerpt": line.strip()})
                break
    return hits


def _build_code_clues() -> dict[str, Any]:
    query_path = REPO_ROOT / "docthinker" / "server" / "routers" / "query.py"
    ingest_path = REPO_ROOT / "docthinker" / "server" / "routers" / "ingest.py"
    processor_path = REPO_ROOT / "docthinker" / "cognitive" / "processor.py"
    return {
        "query_router": {
            "path": str(query_path),
            "hits": _find_line_hits(
                query_path,
                [
                    'text_to_ingest = f"User Question: {question}\\nAssistant Answer: {answer}"',
                    "await state.ingestion_service.ingest_text(text_to_ingest, session_id=session_id)",
                    "fallback_prompt = (",
                ],
            ),
        },
        "ingest_router": {
            "path": str(ingest_path),
            "hits": _find_line_hits(
                ingest_path,
                [
                    "insight = await state.cognitive_processor.process(content, source_type=source_type)",
                    'f"[Cognitive Analysis]:\\n"',
                    "await _write_session_code_snapshot(",
                ],
            ),
        },
        "cognitive_processor": {
            "path": str(processor_path),
            "hits": _find_line_hits(
                processor_path,
                [
                    "Extract entities with type, description, confidence, aliases, attributes.",
                    '{"name": "...", "entity_type": "...", "description": "...", "confidence": 0.0, "aliases": [], "attributes": {}}',
                    "response = await self.llm_func(prompt)",
                ],
            ),
        },
    }


class AsyncModelRouter:
    def __init__(self, client: Any, models: list[str], max_concurrency: int = 32):
        self.client = client
        self.models = [model for model in models if model]
        self.max_concurrency = max(1, int(max_concurrency))
        self._cursor = 0
        self._cursor_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def _next_start_index(self) -> int:
        async with self._cursor_lock:
            idx = self._cursor % len(self.models)
            self._cursor += 1
            return idx

    async def chat_completion(self, *, messages: list[dict[str, Any]], max_tokens: int = 2048) -> Any:
        if not self.models:
            raise ValueError("No LLM models configured")
        start = await self._next_start_index()
        last_err: Exception | None = None
        async with self._semaphore:
            for offset in range(len(self.models)):
                model = self.models[(start + offset) % len(self.models)]
                try:
                    return await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                except Exception as exc:
                    last_err = exc
        if last_err:
            raise last_err
        raise RuntimeError("Model router failed without an explicit exception")


async def _build_model_funcs(trace: dict[str, Any]) -> tuple[Any, Any, Any]:
    import numpy as np

    from docthinker.providers import get_embed_client, get_vlm_client, load_settings
    from graphcore.coregraph.utils import EmbeddingFunc

    settings = load_settings()
    embed_client = get_embed_client(settings)
    llm_client = get_vlm_client(settings)
    router = AsyncModelRouter(
        client=llm_client,
        models=settings.llm_models or [settings.llm_model],
        max_concurrency=settings.llm_router_max_concurrency,
    )

    async def embedding_func_impl(texts: list[str]) -> Any:
        response = await embed_client.embeddings.create(model=settings.embed_model, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    async def traced_llm(prompt: str, system_prompt: Optional[str] = None, **_: Any) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        started = datetime.now().isoformat()
        response = await router.chat_completion(messages=messages, max_tokens=2048)
        ended = datetime.now().isoformat()
        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""
        trace.setdefault("llm_calls", []).append(
            {
                "started_at": started,
                "ended_at": ended,
                "system_prompt": _truncate(system_prompt or "", 6000),
                "prompt": _truncate(prompt or "", 12000),
                "response": _truncate(content or "", 12000),
            }
        )
        return content

    embedding_func = EmbeddingFunc(
        embedding_dim=settings.embed_dim,
        max_token_size=8192,
        func=embedding_func_impl,
    )
    return settings, embedding_func, traced_llm


def _split_prompt(prompt: str) -> tuple[str, str]:
    marker = "\n\n---User Query---\n"
    if marker not in prompt:
        return "", prompt
    return prompt.split(marker, 1)


def _build_root_cause_hypotheses(
    *,
    turn: dict[str, Any],
    source_doc: dict[str, Any],
    question_query_caches: list[dict[str, Any]],
    source_snapshot_findings: list[dict[str, str]],
    chat_doc_entities: list[str],
    chat_doc_relations: list[list[str]],
) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    source_done_after_answer = _delta_seconds(source_doc.get("updated_at_utc"), turn["assistant"].get("timestamp"))
    if source_done_after_answer is not None and source_done_after_answer > 0:
        hypotheses.append(
            {
                "title": "The first answer was generated before source ingestion finished",
                "confidence": "high",
                "evidence": [
                    f"assistant_timestamp_utc={_dt_to_iso_utc(turn['assistant'].get('timestamp'))}",
                    f"source_doc_updated_at_utc={source_doc.get('updated_at_utc')}",
                    f"source_finished_after_answer_seconds={source_done_after_answer}",
                ],
                "fix_hint": "Block querying until ingest completes, or return a processing status instead of generating an answer.",
            }
        )
    if not question_query_caches and "### References" not in str(turn["assistant"].get("content") or ""):
        hypotheses.append(
            {
                "title": "The first answer likely did not come from the grounded RAG path",
                "confidence": "medium_high",
                "evidence": [
                    "No persisted cache_type=query entry for the first question.",
                    "The answer lacks the reference section seen in later grounded responses.",
                    "query.py contains a fallback direct-LLM path.",
                ],
                "fix_hint": "Persist the actual query path and raw prompt, and distinguish grounded RAG from fallback LLM.",
            }
        )
    if chat_doc_entities or chat_doc_relations:
        hypotheses.append(
            {
                "title": "The wrong answer was ingested back into session knowledge",
                "confidence": "high",
                "evidence": [
                    f"chat_doc_entities={chat_doc_entities}",
                    f"chat_doc_relations={chat_doc_relations}",
                    "query.py ingests every chat turn back into knowledge in the background.",
                ],
                "fix_hint": "Do not ingest unverified assistant answers by default; add provenance and confidence gating.",
            }
        )
    if source_snapshot_findings:
        hypotheses.append(
            {
                "title": "Cognitive preprocessing already injected unsupported aliases or descriptions",
                "confidence": "medium",
                "evidence": [json.dumps(item, ensure_ascii=False) for item in source_snapshot_findings[:8]],
                "fix_hint": "Require extracted aliases and descriptions to be grounded by explicit source spans.",
            }
        )
    return hypotheses


async def _replay_current_query(session_dir: Path, question: str, mode: str) -> dict[str, Any]:
    from docthinker import DocThinker, DocThinkerConfig
    from graphcore.coregraph import QueryParam

    trace: dict[str, Any] = {"llm_calls": []}
    settings, embedding_func, llm_func = await _build_model_funcs(trace)
    rag = DocThinker(
        config=DocThinkerConfig(
            working_dir=str(session_dir / "knowledge"),
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
        llm_model_func=llm_func,
        embedding_func=embedding_func,
        graphcore_kwargs={
            "llm_model_max_async": settings.graphcore_llm_max_async,
            "embedding_func_max_async": settings.graphcore_embedding_max_async,
            "max_parallel_insert": settings.graphcore_max_parallel_insert,
        },
    )
    await rag._ensure_graphcore_initialized()
    prompt = await rag.graphcore.aquery(question, param=QueryParam(mode=mode, enable_rerank=True, only_need_prompt=True))
    context = await rag.graphcore.aquery(question, param=QueryParam(mode=mode, enable_rerank=True, only_need_context=True))
    raw_data = await rag.graphcore.aquery_data(question, param=QueryParam(mode=mode, enable_rerank=True))
    system_prompt, user_query = _split_prompt(str(prompt or ""))
    replayed_answer = await llm_func(user_query, system_prompt=system_prompt or None) if prompt else ""
    await rag.finalize_storages()
    rag.close = lambda: None
    return {
        "note": "This replay uses the current knowledge directory, not the exact historical answer-time prompt.",
        "mode": mode,
        "prompt": _truncate(prompt or "", 20000),
        "context": _truncate(context or "", 20000),
        "raw_data": raw_data,
        "replayed_answer": replayed_answer,
        "llm_calls": trace.get("llm_calls") or [],
    }


def _build_static_report(session_dir: Path, turn_index: int, mode: str) -> dict[str, Any]:
    talk_payload = _safe_read_json(session_dir / "talk" / "talk.json") or {}
    turns = _build_turns(talk_payload.get("messages") or [])
    turn = _select_turn(turns, turn_index)
    content_file = _pick_content_file(session_dir)
    source_text = _safe_read_text(content_file) if content_file else ""

    knowledge_dir = session_dir / "knowledge"
    code_dir = session_dir / "code"
    doc_status = _safe_read_json(knowledge_dir / "kv_store_doc_status.json") or {}
    full_docs = _safe_read_json(knowledge_dir / "kv_store_full_docs.json") or {}
    text_chunks = _safe_read_json(knowledge_dir / "kv_store_text_chunks.json") or {}
    llm_cache = _safe_read_json(knowledge_dir / "kv_store_llm_response_cache.json") or {}
    entity_chunks = _safe_read_json(knowledge_dir / "kv_store_entity_chunks.json") or {}
    full_entities = _safe_read_json(knowledge_dir / "kv_store_full_entities.json") or {}
    full_relations = _safe_read_json(knowledge_dir / "kv_store_full_relations.json") or {}
    knowledge_graph = _safe_read_json(knowledge_dir / "knowledge_graph.json") or {}

    question = str(turn["user"].get("content") or "")
    answer = str(turn["assistant"].get("content") or "")
    timeline = _build_doc_timeline(doc_status, full_docs, question)
    source_doc_id = _find_source_doc_id(doc_status, full_docs)
    chat_doc_id = _find_chat_doc_id(full_docs, question)
    source_doc = next((item for item in timeline if item["doc_id"] == source_doc_id), {})
    chat_doc = next((item for item in timeline if item["doc_id"] == chat_doc_id), {})

    source_chunk_id = ((doc_status.get(source_doc_id) or {}).get("chunks_list") or [None])[0]
    chat_chunk_id = ((doc_status.get(chat_doc_id) or {}).get("chunks_list") or [None])[0]
    source_chunk = text_chunks.get(source_chunk_id) or {}
    chat_chunk = text_chunks.get(chat_chunk_id) or {}

    qa_text = f"User Question: {question}\nAssistant Answer: {answer}"
    question_keyword_caches = _find_cache_entries_by_prompt(llm_cache, question, {"keywords"})
    question_query_caches = _find_cache_entries_by_prompt(llm_cache, question, {"query"})
    source_extract_caches = _cache_entries_from_ids(llm_cache, source_chunk.get("llm_cache_list") or [])
    chat_extract_caches = _cache_entries_from_ids(llm_cache, chat_chunk.get("llm_cache_list") or [])
    chat_extract_by_prompt = _find_cache_entries_by_prompt(llm_cache, qa_text, {"extract"})

    snapshots = _load_code_snapshots(code_dir)
    source_snapshot_item = _pick_source_snapshot(snapshots, content_file)
    source_snapshot = (source_snapshot_item or {}).get("payload") or {}
    source_snapshot_findings = _find_alias_or_name_not_in_snapshot(source_snapshot, source_text)

    chat_doc_entities = (full_entities.get(chat_doc_id) or {}).get("entity_names") or []
    chat_doc_relations = _relations_for_doc(full_relations, chat_doc_id)
    interesting_names = set(chat_doc_entities)
    for finding in source_snapshot_findings:
        interesting_names.add(str(finding.get("value") or finding.get("entity") or ""))
    interesting_names.discard("")

    return {
        "generated_at": datetime.now().isoformat(),
        "session_dir": str(session_dir),
        "mode": mode,
        "turn": {
            "index": turn_index,
            "user": turn["user"],
            "assistant": turn["assistant"],
            "user_timestamp_utc": _dt_to_iso_utc(turn["user"].get("timestamp")),
            "assistant_timestamp_utc": _dt_to_iso_utc(turn["assistant"].get("timestamp")),
            "answer_latency_seconds": _delta_seconds(turn["assistant"].get("timestamp"), turn["user"].get("timestamp")),
        },
        "session_artifacts": {
            "talk_file": str(session_dir / "talk" / "talk.json"),
            "content_file": str(content_file) if content_file else None,
            "source_doc_id": source_doc_id,
            "source_chunk_id": source_chunk_id,
            "chat_doc_id": chat_doc_id,
            "chat_chunk_id": chat_chunk_id,
        },
        "timeline": {
            "docs": timeline,
            "source_doc_finished_after_answer_seconds": _delta_seconds(source_doc.get("updated_at_utc"), turn["assistant"].get("timestamp")),
            "chat_doc_created_after_answer_seconds": _delta_seconds(chat_doc.get("created_at_utc"), turn["assistant"].get("timestamp")),
        },
        "historical_prompt_availability": {
            "historical_answer_prompt_available": False,
            "reason": "No exact answer-time system prompt was found in persisted artifacts.",
            "nearest_persisted_artifacts": {
                "question_keywords_caches": question_keyword_caches,
                "question_query_caches": question_query_caches,
                "chat_turn_extract_caches": chat_extract_by_prompt,
            },
        },
        "persisted_llm_artifacts": {
            "question_keywords_caches": question_keyword_caches,
            "question_query_caches": question_query_caches,
            "source_chunk_extract_caches": source_extract_caches,
            "chat_turn_extract_caches": chat_extract_caches,
            "chat_turn_extract_caches_by_prompt": chat_extract_by_prompt,
        },
        "source_snapshot": {
            "snapshot_file": str((source_snapshot_item or {}).get("path") or ""),
            "source_name": source_snapshot.get("source_name"),
            "created_at": source_snapshot.get("created_at"),
            "suspicious_names_or_aliases": source_snapshot_findings,
            "snapshot_relationships_preview": _pick_snapshot_relations(source_snapshot),
            "node_count": len(source_snapshot.get("nodes") or []),
            "relationship_count": len(source_snapshot.get("relationships") or []),
        },
        "graph_pollution": {
            "source_doc_full_entities": (full_entities.get(source_doc_id) or {}).get("entity_names") or [],
            "source_doc_full_relations": _relations_for_doc(full_relations, source_doc_id),
            "chat_doc_full_entities": chat_doc_entities,
            "chat_doc_full_relations": chat_doc_relations,
            "source_chunk_entity_index_hits": _entities_for_chunk(entity_chunks, source_chunk_id),
            "chat_chunk_entity_index_hits": _entities_for_chunk(entity_chunks, chat_chunk_id),
            "graphml_hits": _graphml_hits(knowledge_dir / "graph_chunk_entity_relation.graphml", interesting_names),
            "local_knowledge_graph_entities": _find_knowledge_graph_entities(knowledge_graph, interesting_names),
            "local_knowledge_graph_relations": _find_knowledge_graph_relations(knowledge_graph, interesting_names),
        },
        "code_clues": _build_code_clues(),
        "root_cause_hypotheses": _build_root_cause_hypotheses(
            turn=turn,
            source_doc=source_doc,
            question_query_caches=question_query_caches,
            source_snapshot_findings=source_snapshot_findings,
            chat_doc_entities=chat_doc_entities,
            chat_doc_relations=chat_doc_relations,
        ),
        "raw_content_previews": {
            "source_doc_preview": _truncate((full_docs.get(source_doc_id) or {}).get("content") or "", 3000),
            "chat_doc_preview": _truncate((full_docs.get(chat_doc_id) or {}).get("content") or "", 3000),
            "source_chunk_preview": _truncate(source_chunk.get("content") or "", 3000),
            "chat_chunk_preview": _truncate(chat_chunk.get("content") or "", 3000),
        },
    }


async def arun(args: argparse.Namespace) -> Path:
    session_dir = _resolve_session_dir(args.session_dir, args.session_id)
    report = _build_static_report(session_dir, args.turn, args.mode)
    if args.replay:
        try:
            report["current_replay"] = await _replay_current_query(
                session_dir,
                str(report["turn"]["user"].get("content") or ""),
                args.mode,
            )
        except Exception as exc:
            report["current_replay"] = {"error": str(exc)}

    output_path = Path(args.output) if args.output else session_dir / "code" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_trace_pipeline_turn{args.turn}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a session turn through persisted DocThinker and GraphCore artifacts")
    parser.add_argument("--session-dir", help="Absolute or relative session directory, e.g. data/#00004")
    parser.add_argument("--session-id", help="Session id under data/, e.g. #00004")
    parser.add_argument("--turn", type=int, default=1, help="1-based turn index to trace")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    parser.add_argument("--replay", action="store_true", help="Replay the current query against the current knowledge dir")
    parser.add_argument("--output", help="Optional output JSON path")
    out_path = asyncio.run(arun(parser.parse_args()))
    print(f"Trace written to: {out_path}")


if __name__ == "__main__":
    main()
