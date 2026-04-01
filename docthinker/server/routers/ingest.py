import asyncio
import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict, Iterable

import numpy as np

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, Form, Request

from ..schemas import IngestRequest, SignalIngestRequest
from ..state import state
from docthinker.hypergraph.utils import compute_mdhash_id
from docthinker.utils import separate_content

_log = logging.getLogger("docthinker.ingest")

router = APIRouter()


def _read_text_auto_encoding(path: Path) -> str:
    """Read a text file with automatic encoding detection.

    Tries UTF-8 first (strict), then falls back to charset-normalizer
    for robust detection of GBK/GB2312/GB18030 and other encodings.
    Never silently drops bytes.
    """
    raw = path.read_bytes()
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        result = from_bytes(raw).best()
        if result is not None:
            _log.info("[encoding] '%s' detected as %s", path.name, result.encoding)
            return str(result)
    except Exception:
        pass
    for enc in ("gbk", "gb18030", "big5", "euc-jp", "euc-kr", "cp1252"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


async def _sync_kg_entity_ids(session_id: str) -> None:
    """Sync KG entity IDs from session's GraphCore into state.kg_entity_ids for neuro_memory linkage."""
    try:
        if not state.ingestion_service or not state.session_manager:
            return
        config = state.ingestion_service.create_rag_config()
        session_rag = state.session_manager.get_session_rag(session_id, config)
        gc = getattr(session_rag, "graphcore", None)
        if gc is None:
            return
        graph = getattr(gc, "chunk_entity_relation_graph", None)
        if graph is None:
            return
        labels = await graph.get_all_labels()
        if labels:
            state.kg_entity_ids.update(labels)
            _log.info("[kg_sync] synced %d entity IDs from session %s (total: %d)",
                      len(labels), session_id, len(state.kg_entity_ids))
    except Exception as e:
        _log.warning("[kg_sync] failed for session %s: %s", session_id, e)


async def _run_density_clustering(sid: str) -> None:
    """Run density clustering on KG node embeddings and save cluster summaries."""
    if not state.session_manager or not state.ingestion_service:
        return
    session = state.session_manager.get_session(sid)
    if not session:
        return
    metadata = session.get("metadata") or {}
    knowledge_dir = metadata.get("knowledge_dir") or session.get("knowledge_dir")
    if not knowledge_dir:
        return

    config = state.ingestion_service.create_rag_config()
    graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {}) if state.rag_instance else {}
    session_rag = state.session_manager.get_session_rag(sid, config, graphcore_kwargs)
    if state.rag_instance:
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
    await session_rag._ensure_graphcore_initialized()
    gc = session_rag.graphcore
    if gc is None:
        return

    graph = gc.chunk_entity_relation_graph
    nodes_data = await graph.get_all_nodes()
    if len(nodes_data) < 4:
        _log.info("[cluster] only %d nodes for session %s, skipping clustering", len(nodes_data), sid)
        return

    # Collect embeddings from the entities VDB
    try:
        vdb = gc.entities_vdb
        # NanoVectorDBStorage exposes an async `client_storage` property
        # that returns the underlying NanoVectorDB.__storage dict.
        if hasattr(vdb, "client_storage"):
            raw_storage = await vdb.client_storage
        else:
            _log.warning("[cluster] VDB has no client_storage accessor for session %s", sid)
            return

        raw = raw_storage.get("data", []) if isinstance(raw_storage, dict) else raw_storage
        if not raw or len(raw) == 0:
            _log.info("[cluster] VDB storage empty for session %s", sid)
            return

        # Build a mapping from entity name → embedding
        node_ids = [str(n.get("id") or n.get("entity_id") or "") for n in nodes_data]
        vdb_names = {}
        emb_list = []
        filtered_nodes = []
        for rec in raw:
            name = str(rec.get("entity_name") or "").strip()
            if name:
                vdb_names[name] = rec.get("__vector__")

        for i, nid in enumerate(node_ids):
            vec = vdb_names.get(nid)
            if vec is not None:
                emb_list.append(vec)
                filtered_nodes.append(nodes_data[i])

        if len(emb_list) < 4:
            _log.info("[cluster] only %d nodes with embeddings for session %s, skipping", len(emb_list), sid)
            return

        embeddings = np.array(emb_list)
    except Exception as exc:
        _log.warning("[cluster] failed to extract embeddings for session %s: %s", sid, exc)
        return

    llm_fn = getattr(session_rag, "llm_model_func", None)
    if not llm_fn:
        _log.warning("[cluster] no LLM func available for session %s", sid)
        return

    from docthinker.kg_expansion import build_cluster_summaries, save_cluster_summaries

    summaries = await build_cluster_summaries(
        filtered_nodes, embeddings, llm_fn, session_id=sid,
    )
    if summaries:
        save_cluster_summaries(summaries, Path(knowledge_dir) / "cluster_summaries.json")
        _log.info("[cluster] saved %d cluster summaries for session %s", len(summaries), sid)
    else:
        _log.info("[cluster] no dense clusters found for session %s", sid)


async def _background_edge_discovery(sid: str) -> None:
    """Run latent edge discovery in the background after ingestion.

    Discovered edges are written into the graph with ``is_discovered=1``
    so the frontend can render them in red.  This never blocks user queries.
    """
    try:
        if not state.session_manager or not state.rag_instance:
            return
        session = state.session_manager.get_session(sid)
        if not session:
            return

        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(sid, config, graphcore_kwargs)
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        gc = session_rag.graphcore
        if gc is None:
            return

        graph = gc.chunk_entity_relation_graph
        nodes_data = await graph.get_all_nodes()
        edges_data = await graph.get_all_edges()

        if len(nodes_data) < 3:
            _log.info("[edge_discovery] only %d nodes for session %s, skipping", len(nodes_data), sid)
            return

        from docthinker.kg_expansion.edge_discovery import discover_edges

        llm_fn = getattr(session_rag, "llm_model_func", None)
        if not llm_fn:
            return

        discovered = await discover_edges(
            nodes_data, edges_data, llm_fn,
            window_size=30, overlap=10, max_parallel=2,
        )

        if not discovered:
            _log.info("[edge_discovery] no new edges discovered for session %s", sid)
            return

        # Write discovered edges into the graph
        added = 0
        for edge in discovered:
            if await graph.has_edge(edge.source, edge.target):
                continue
            await graph.upsert_edge(edge.source, edge.target, {
                "keywords": edge.keywords,
                "description": edge.description,
                "weight": "0.5",
                "is_discovered": "1",
                "source_id": "edge_discovery",
            })
            added += 1

        if added > 0:
            await graph.index_done_callback(force_save=True)

            # Also index discovered edges into the relationships VDB
            if gc.relationships_vdb and session_rag.embedding_func:
                vdb_data = {}
                for edge in discovered:
                    edge_key = f"{edge.source}-{edge.target}"
                    content = f"{edge.source} {edge.keywords} {edge.target}: {edge.description}"
                    vdb_data[edge_key] = {
                        "src_id": edge.source,
                        "tgt_id": edge.target,
                        "content": content,
                    }
                if vdb_data:
                    await gc.relationships_vdb.upsert(vdb_data)
                    await gc.relationships_vdb.index_done_callback()

        _log.info("[edge_discovery:bg] session %s — discovered %d, persisted %d edges",
                  sid, len(discovered), added)

    except Exception as exc:
        _log.warning("[edge_discovery:bg] failed for session %s (non-fatal): %s", sid, exc)


async def _background_self_study(sid: str) -> None:
    """Run KG self-study loop in the background after ingestion.

    The self-study loop lets the LLM autonomously quiz and densify the KG,
    improving retrieval quality before user queries arrive.
    """
    try:
        if not state.session_manager or not state.rag_instance:
            return
        session = state.session_manager.get_session(sid)
        if not session:
            return

        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(sid, config, graphcore_kwargs)
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        gc = session_rag.graphcore
        if gc is None:
            return

        graph = gc.chunk_entity_relation_graph
        all_nodes = await graph.get_all_nodes()
        all_edges = await graph.get_all_edges()

        if len(all_nodes) < 5:
            _log.info("[self_study] only %d nodes for session %s, skipping",
                      len(all_nodes), sid)
            return

        llm_fn = getattr(session_rag, "llm_model_func", None)
        if not llm_fn:
            return

        from docthinker.kg_self_study.orchestrator import (
            SelfStudyConfig,
            SelfStudyOrchestrator,
        )

        workdir = getattr(gc, "workspace", None) or "./data/_system"

        async def kg_query_func(question: str):
            try:
                result = await gc.aquery_data(question)
                return {
                    "entities": result.get("entities", []),
                    "relations": result.get("relations", []),
                    "chunks": result.get("chunks", []),
                }
            except Exception:
                return {"entities": [], "relations": [], "chunks": []}

        async def kg_write_func(operations: dict):
            for edge in operations.get("new_edges", []):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                if not src or not tgt or src == tgt:
                    continue
                if await graph.has_edge(src, tgt):
                    continue
                await graph.upsert_edge(src, tgt, {
                    "keywords": edge.get("keywords", edge.get("relation", "")),
                    "description": edge.get("relation", ""),
                    "weight": str(edge.get("confidence", 0.5)),
                    "is_discovered": "1",
                    "source_id": "self_study",
                })

            for upd in operations.get("entity_updates", []):
                entity_name = upd.get("entity", "")
                if not entity_name:
                    continue
                if upd.get("action") == "enrich_description":
                    node = await graph.get_node(entity_name)
                    if node:
                        old_desc = node.get("description", "")
                        new_content = upd.get("new_content", "")
                        if new_content and new_content not in old_desc:
                            node["description"] = f"{old_desc} | {new_content}"
                            await graph.upsert_node(entity_name, node)

            await graph.index_done_callback(force_save=True)

        async def kg_read_nodes():
            return await graph.get_all_nodes()

        async def kg_read_edges():
            return await graph.get_all_edges()

        study_config = SelfStudyConfig(
            max_rounds=3,
            max_tokens=30000,
            questions_per_round=2,
            experience_store_path=f"{workdir}/experiences_{sid}.json",
        )

        orchestrator = SelfStudyOrchestrator(
            llm_func=llm_fn,
            kg_query_func=kg_query_func,
            kg_write_func=kg_write_func,
            kg_read_nodes_func=kg_read_nodes,
            kg_read_edges_func=kg_read_edges,
            config=study_config,
        )

        result = await orchestrator.run_session()
        _log.info(
            "[self_study:bg] session %s — %d rounds, %d new edges, "
            "%d updates, %d experiences in %.1fs",
            sid, len(result.rounds), result.total_new_edges,
            result.total_entity_updates, result.total_experiences,
            result.elapsed_seconds,
        )

    except Exception as exc:
        _log.warning("[self_study:bg] failed for session %s (non-fatal): %s",
                     sid, exc, exc_info=True)


def _truncate_text(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _serialize_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


def _build_signal_text(request: SignalIngestRequest) -> str:
    payload_text = _serialize_payload(request.payload)
    payload_text = _truncate_text(payload_text, 8000)
    meta: Dict[str, Any] = {}
    if request.source_uri:
        meta["source_uri"] = request.source_uri
    if request.timestamp:
        meta["timestamp"] = request.timestamp
    if request.tags:
        meta["tags"] = request.tags
    if request.metadata:
        meta["metadata"] = request.metadata
    meta_text = _serialize_payload(meta) if meta else ""
    if meta_text:
        meta_text = _truncate_text(meta_text, 2000)
        return (
            f"Signal Modality: {request.modality or 'unknown'}\n"
            f"Source Type: {request.source_type}\n"
            f"Payload:\n{payload_text}\n\n"
            f"Meta:\n{meta_text}"
        )
    return (
        f"Signal Modality: {request.modality or 'unknown'}\n"
        f"Source Type: {request.source_type}\n"
        f"Payload:\n{payload_text}"
    )


def _load_content_list(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    content_list = data.get("content_list") if isinstance(data, dict) else None
    if content_list is None:
        if isinstance(data, list):
            content_list = data
        else:
            raise ValueError(f"Unrecognised JSON format in {json_path}")
    base_dir = json_path.parent
    for block in content_list:
        if isinstance(block, dict) and "img_path" in block:
            img_path = Path(block["img_path"])
            if not img_path.is_absolute():
                img_path = (base_dir / img_path).resolve()
            block["img_path"] = img_path.as_posix()
    return content_list


def _collect_content_list_groups(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in paths:
        stem = path.stem
        if stem.endswith("_content_list") and "_part" in stem:
            doc_id = stem.split("_part")[0]
        elif stem.endswith("_content_list"):
            doc_id = stem[: -len("_content_list")]
        else:
            doc_id = stem
        grouped.setdefault(doc_id, []).append(path)
    for doc_id, items in grouped.items():
        grouped[doc_id] = sorted(items)
    return grouped


def _safe_snapshot_name(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return "source"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    cleaned = cleaned.strip("._-")
    return cleaned[:64] or "source"


def _split_text_for_snapshot(text: str, chunk_size: int = 1200, overlap: int = 180) -> List[Dict[str, Any]]:
    source = (text or "").strip()
    if not source:
        return []
    if chunk_size <= 0:
        chunk_size = 1200
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: List[Dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(source):
        end = min(len(source), start + chunk_size)
        chunk_text = source[start:end]
        chunks.append(
            {
                "chunk_id": compute_mdhash_id(chunk_text, prefix="chunk-"),
                "index": index,
                "start": start,
                "end": end,
                "text": chunk_text,
            }
        )
        if end >= len(source):
            break
        start = end - overlap
        index += 1
    return chunks


def _build_snapshot_nodes(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for entity in metadata.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = (entity.get("name") or "").strip()
        if not name:
            continue
        entity_type = (entity.get("entity_type") or "UNKNOWN").strip() or "UNKNOWN"
        node_id = compute_mdhash_id(f"{name.lower()}|{entity_type.lower()}", prefix="node-")
        nodes.append(
            {
                "id": node_id,
                "label": name,
                "entity_type": entity_type,
                "description": entity.get("description") or "",
                "confidence": float(entity.get("confidence") or 0.0),
                "aliases": entity.get("aliases") or [],
                "attributes": entity.get("attributes") or {},
            }
        )
    return nodes


def _build_snapshot_edges(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    grouped = (
        ("relations", False),
        ("inferred_relations", True),
    )
    for key, inferred in grouped:
        for relation in metadata.get(key) or []:
            if not isinstance(relation, dict):
                continue
            src = (relation.get("source") or "").strip()
            tgt = (relation.get("target") or "").strip()
            rel = (relation.get("relation") or "").strip()
            if not src or not tgt or not rel:
                continue
            edge_id = compute_mdhash_id(
                f"{src.lower()}|{rel.lower()}|{tgt.lower()}|{int(inferred)}",
                prefix="edge-",
            )
            edges.append(
                {
                    "id": edge_id,
                    "source": src,
                    "target": tgt,
                    "relation": rel,
                    "description": relation.get("description") or "",
                    "confidence": float(relation.get("confidence") or 0.0),
                    "inferred": inferred,
                }
            )
    return edges


async def _write_session_code_snapshot(
    *,
    session_id: Optional[str],
    source_type: str,
    source_name: str,
    original_text: str,
    processed_text: str,
    metadata: Dict[str, Any],
) -> None:
    if not session_id or not state.session_manager:
        return
    try:
        code_dir = state.session_manager.get_session_code_dir(session_id)
    except Exception:
        return

    chunk_source = original_text or processed_text
    chunks = _split_text_for_snapshot(chunk_source)
    nodes = _build_snapshot_nodes(metadata)
    edges = _build_snapshot_edges(metadata)

    payload = {
        "session_id": session_id,
        "source_type": source_type,
        "source_name": source_name,
        "created_at": datetime.now().isoformat(),
        "raw_text_length": len(original_text or ""),
        "processed_text_length": len(processed_text or ""),
        "chunks": chunks,
        "nodes": nodes,
        "relationships": edges,
        "metadata": {
            "summary": metadata.get("summary"),
            "reasoning": metadata.get("reasoning"),
            "key_points": metadata.get("key_points") or [],
            "concepts": metadata.get("concepts") or [],
            "hypotheses": metadata.get("hypotheses") or [],
            "action_items": metadata.get("action_items") or [],
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = _safe_snapshot_name(Path(source_name).stem or source_name or "source")
    out_path = code_dir / f"{ts}_{safe_name}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _allocate_upload_target(session_id: str, filename: str) -> Path:
    if not session_id:
        raise ValueError("session_id is required")
    if not state.session_manager:
        raise ValueError("Session manager is not initialized")
    return state.session_manager.allocate_session_file_path(session_id, filename)


async def _extract_session_id(session_id: Optional[str], request: Optional[Request]) -> Optional[str]:
    sid = (session_id or "").strip()
    if sid:
        return sid
    if request is None:
        return None
    sid = (request.query_params.get("session_id") or "").strip()
    if sid:
        return sid
    try:
        form = await request.form()
        sid = str(form.get("session_id") or "").strip()
        if sid:
            return sid
    except Exception:
        pass
    return None


async def _process_text_for_ingest(
    content: str, source_type: str, *, skip_cognitive: bool = False,
) -> tuple[str, Dict[str, Any]]:
    """Pre-process text before feeding to GraphCore.

    For file-based ingestion (image, file, content_list) the cognitive
    processor is **skipped** because GraphCore already performs entity/
    relation extraction.  Running the cognitive processor would add 4
    sequential LLM calls (~60s) that duplicate GraphCore's work.
    """
    processed_text = content
    metadata: Dict[str, Any] = {"source_type": source_type, "type": "text"}

    if skip_cognitive or source_type in ("image", "file", "content_list"):
        _log.info("[cognitive] skipped (redundant for %s — GraphCore handles extraction)", source_type)
        return processed_text, metadata

    if state.cognitive_processor:
        t0 = time.perf_counter()
        try:
            insight = await state.cognitive_processor.process(content, source_type=source_type)
            elapsed = time.perf_counter() - t0
            _log.info("[cognitive] completed in %.2fs | entities=%d relations=%d",
                      elapsed, len(insight.entities), len(insight.relations))

            link_names = ", ".join([l.name for l in insight.potential_links[:10]])
            entity_names = ", ".join([e.name for e in insight.entities[:20]])
            relation_pairs = ", ".join([f"{r.source}->{r.relation}->{r.target}" for r in insight.relations[:20]])
            inferred_pairs = ", ".join(
                [f"{r.source}->{r.relation}->{r.target}" for r in insight.inferred_relations[:20]]
            )
            processed_text = (
                f"Source: {source_type}\n"
                f"Content:\n{content}\n\n"
                f"[Cognitive Analysis]:\n"
                f"Summary: {insight.summary}\n"
                f"Reasoning: {insight.reasoning}\n"
                f"Key Points: {', '.join(insight.key_points)}\n"
                f"Concepts: {', '.join(insight.concepts)}\n"
                f"Entities: {entity_names}\n"
                f"Relations: {relation_pairs}\n"
                f"Inferred Relations: {inferred_pairs}\n"
                f"Hypotheses: {', '.join(insight.hypotheses)}\n"
                f"Potential Links: {link_names}"
            )
            metadata.update(
                {
                    "summary": insight.summary,
                    "reasoning": insight.reasoning,
                    "key_points": insight.key_points,
                    "concepts": insight.concepts,
                    "entities": [e.dict() for e in insight.entities],
                    "relations": [r.dict() for r in insight.relations],
                    "inferred_relations": [r.dict() for r in insight.inferred_relations],
                    "hypotheses": insight.hypotheses,
                    "action_items": insight.action_items,
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            _log.warning("[cognitive] failed after %.2fs: %s", elapsed, exc)
    return processed_text, metadata


def _build_entity_text(name: str, description: str) -> str:
    text = (name or "").strip()
    desc = (description or "").strip()
    if desc:
        return f"{text}. {desc}"
    return text


def _build_document_macro_text(text: str, metadata: Dict[str, Any]) -> str:
    parts: List[str] = []
    summary = metadata.get("summary")
    reasoning = metadata.get("reasoning")
    key_points = metadata.get("key_points") or []
    concepts = metadata.get("concepts") or []
    hypotheses = metadata.get("hypotheses") or []
    action_items = metadata.get("action_items") or []
    if summary:
        parts.append(f"Summary: {summary}")
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")
    if key_points:
        parts.append(f"Key Points: {', '.join([str(p) for p in key_points])}")
    if concepts:
        parts.append(f"Concepts: {', '.join([str(c) for c in concepts])}")
    if hypotheses:
        parts.append(f"Hypotheses: {', '.join([str(h) for h in hypotheses])}")
    if action_items:
        parts.append(f"Action Items: {', '.join([str(a) for a in action_items])}")
    parts.append(f"Content: {_truncate_text(text, 2000)}")
    return "\n".join([p for p in parts if p])


def _extract_macro_terms(metadata: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for item in (metadata.get("concepts") or []):
        if isinstance(item, str) and item.strip():
            terms.append(item.strip().lower())
    for item in (metadata.get("key_points") or []):
        if isinstance(item, str) and item.strip():
            terms.append(item.strip().lower())
    return sorted(set(terms))


async def _auto_link_macro_documents(
    knowledge_graph: Any,
    embedding_func: Any,
    doc_entity: Any,
    macro_text: str,
    macro_terms: List[str],
    document_id: Optional[str],
    min_similarity: float = 0.78,
    max_existing: int = 200,
    max_links: int = 5,
) -> None:
    if not embedding_func or not doc_entity or not macro_text:
        return
    existing_docs = [
        e for e in knowledge_graph.entities.values()
        if getattr(e, "type", "") == "DOCUMENT" and e.id != doc_entity.id
    ]
    if not existing_docs:
        return
    if len(existing_docs) > max_existing:
        existing_docs = existing_docs[:max_existing]
    texts = [macro_text]
    for doc in existing_docs:
        doc_text = ""
        if isinstance(doc.properties, dict):
            doc_text = doc.properties.get("macro_text") or ""
        if not doc_text:
            doc_text = doc.description or doc.name
        texts.append(doc_text)
    try:
        embeddings = await embedding_func(texts)
    except Exception:
        return
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        return
    anchor = vectors[0]
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)
    others = vectors[1:]
    other_norms = np.linalg.norm(others, axis=1, keepdims=True) + 1e-9
    others = others / other_norms
    similarities = others @ anchor
    macro_term_set = set(macro_terms or [])
    ranked = np.argsort(similarities)[::-1]
    added = 0
    for idx in ranked:
        emb_score = float(similarities[int(idx)])
        if emb_score < min_similarity:
            break
        other = existing_docs[int(idx)]
        other_terms = []
        if isinstance(other.properties, dict):
            other_terms = other.properties.get("macro_terms") or []
        other_term_set = set([str(t).lower() for t in other_terms])
        if macro_term_set or other_term_set:
            overlap = len(macro_term_set & other_term_set)
            union = len(macro_term_set | other_term_set)
            term_score = overlap / union if union > 0 else 0.0
        else:
            term_score = 0.0
        score = emb_score * 0.7 + term_score * 0.3
        if score < min_similarity:
            continue
        if knowledge_graph.get_relationship(doc_entity.id, other.id, "analogous_to") or knowledge_graph.get_relationship(
            other.id, doc_entity.id, "analogous_to"
        ):
            continue
        knowledge_graph.add_relationship(
            source_id=doc_entity.id,
            target_id=other.id,
            type="analogous_to",
            properties={
                "method": "macro_similarity",
                "similarity": score,
                "embedding_similarity": emb_score,
                "term_overlap": term_score,
            },
            document_id=document_id,
            description="auto related by macro similarity",
            confidence=score,
            source="auto:macro",
            validate=score >= 0.86,
        )
        added += 1
        if added >= max_links:
            break


async def _auto_link_related_entities(
    knowledge_graph: Any,
    embedding_func: Any,
    new_entities: List[Any],
    document_id: Optional[str],
    min_similarity: float = 0.84,
    max_existing: int = 200,
    max_links_per_entity: int = 5,
) -> None:
    if not new_entities or not embedding_func:
        return
    new_entity_ids = {entity.id for entity in new_entities}
    existing_entities = [e for e in knowledge_graph.entities.values() if e.id not in new_entity_ids]
    if not existing_entities:
        return
    if len(existing_entities) > max_existing:
        existing_entities = existing_entities[:max_existing]
    texts = [
        _build_entity_text(entity.name, getattr(entity, "description", ""))
        for entity in new_entities
    ] + [
        _build_entity_text(entity.name, getattr(entity, "description", ""))
        for entity in existing_entities
    ]
    try:
        embeddings = await embedding_func(texts)
    except Exception:
        return
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        return
    new_vectors = vectors[: len(new_entities)]
    existing_vectors = vectors[len(new_entities) :]
    new_norms = np.linalg.norm(new_vectors, axis=1, keepdims=True) + 1e-9
    existing_norms = np.linalg.norm(existing_vectors, axis=1, keepdims=True) + 1e-9
    new_vectors = new_vectors / new_norms
    existing_vectors = existing_vectors / existing_norms
    for index, entity in enumerate(new_entities):
        similarities = existing_vectors @ new_vectors[index]
        order = np.argsort(similarities)[::-1]
        added = 0
        for idx in order:
            score = float(similarities[idx])
            if score < min_similarity:
                break
            other = existing_entities[int(idx)]
            if knowledge_graph.get_relationship(entity.id, other.id, "related_to") or knowledge_graph.get_relationship(
                other.id, entity.id, "related_to"
            ):
                continue
            knowledge_graph.add_relationship(
                source_id=entity.id,
                target_id=other.id,
                type="related_to",
                properties={"method": "embedding_similarity", "similarity": score},
                document_id=document_id,
                description="auto related by semantic similarity",
                confidence=score,
                source="auto:embedding",
                validate=score >= 0.92,
            )
            added += 1
            if added >= max_links_per_entity:
                break


async def _resolve_rag_for_session(session_id: Optional[str]) -> Any:
    rag_instance = state.rag_instance
    if not rag_instance:
        return None
    if not session_id or not state.session_manager:
        return None
    try:
        session_rag = state.session_manager.get_session_rag(
            session_id,
            rag_instance.config,
            getattr(rag_instance, "graphcore_kwargs", {}),
        )
        session_rag.llm_model_func = rag_instance.llm_model_func
        session_rag.embedding_func = rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        return session_rag
    except Exception:
        return None


async def _update_local_knowledge_graph(text: str, metadata: Dict[str, Any], session_id: Optional[str] = None) -> None:
    rag_instance = await _resolve_rag_for_session(session_id)
    if not rag_instance or not getattr(rag_instance, "knowledge_graph", None):
        return
    knowledge_graph = rag_instance.knowledge_graph
    entities_data = metadata.get("entities") or []
    relations = metadata.get("relations") or []
    inferred_relations = metadata.get("inferred_relations") or []
    if not entities_data and not relations and not inferred_relations:
        return
    document_id = compute_mdhash_id(text, prefix="doc-")
    macro_text = _build_document_macro_text(text, metadata)
    macro_terms = _extract_macro_terms(metadata)
    summary = metadata.get("summary") or ""
    doc_name = f"doc:{document_id}"
    doc_entity = knowledge_graph.add_entity(
        name=doc_name,
        type="DOCUMENT",
        properties={
            "macro_text": macro_text,
            "macro_terms": macro_terms,
            "summary": summary,
        },
        document_id=document_id,
        description=summary or _truncate_text(text, 400),
        confidence=0.6,
        source="cognitive",
    )
    name_type_map = {}
    new_entities = []
    for entity in entities_data:
        if not isinstance(entity, dict):
            continue
        name = (entity.get("name") or "").strip()
        if not name:
            continue
        entity_type = entity.get("entity_type") or "UNKNOWN"
        name_type_map[name.lower()] = entity_type
        confidence = float(entity.get("confidence") or 0.0)
        item = knowledge_graph.add_entity(
            name=name,
            type=entity_type,
            properties=entity.get("attributes") or {},
            document_id=document_id,
            description=entity.get("description") or "",
            confidence=max(confidence, 0.1),
            aliases=entity.get("aliases") or [],
            source="cognitive",
        )
        new_entities.append(item)
        try:
            knowledge_graph.add_relationship(
                source_id=item.id,
                target_id=doc_entity.id,
                type="mentioned_in",
                properties={},
                document_id=document_id,
                description="auto linked entity to document",
                confidence=0.8,
                source="auto:doc",
                validate=False,
            )
        except Exception:
            pass

    def _get_or_create_entity(name: str) -> Optional[Any]:
        if not name:
            return None
        entity_type = name_type_map.get(name.lower())
        entity = knowledge_graph.get_entity_by_name(name, entity_type) or knowledge_graph.get_entity_by_name(name)
        if entity:
            return entity
        return knowledge_graph.add_entity(
            name=name,
            type=entity_type or "UNKNOWN",
            document_id=document_id,
            description="",
            confidence=0.1,
            source="cognitive",
        )

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        src = (relation.get("source") or "").strip()
        tgt = (relation.get("target") or "").strip()
        rel_type = (relation.get("relation") or "").strip()
        if not src or not tgt or not rel_type:
            continue
        src_entity = _get_or_create_entity(src)
        tgt_entity = _get_or_create_entity(tgt)
        if not src_entity or not tgt_entity:
            continue
        properties = {}
        if relation.get("evidence"):
            properties["evidence"] = relation.get("evidence")
        confidence = float(relation.get("confidence") or 0.0)
        knowledge_graph.add_relationship(
            source_id=src_entity.id,
            target_id=tgt_entity.id,
            type=rel_type,
            properties=properties,
            document_id=document_id,
            description=relation.get("description") or "",
            confidence=max(confidence, 0.1),
            source="cognitive",
            validate=confidence >= 0.9,
        )

    for relation in inferred_relations:
        if not isinstance(relation, dict):
            continue
        src = (relation.get("source") or "").strip()
        tgt = (relation.get("target") or "").strip()
        rel_type = (relation.get("relation") or "").strip()
        if not src or not tgt or not rel_type:
            continue
        src_entity = _get_or_create_entity(src)
        tgt_entity = _get_or_create_entity(tgt)
        if not src_entity or not tgt_entity:
            continue
        properties = {"inferred": True}
        if relation.get("evidence"):
            properties["evidence"] = relation.get("evidence")
        confidence = float(relation.get("confidence") or 0.0)
        knowledge_graph.add_relationship(
            source_id=src_entity.id,
            target_id=tgt_entity.id,
            type=rel_type,
            properties=properties,
            document_id=document_id,
            description=relation.get("description") or "",
            confidence=max(confidence, 0.1),
            source="cognitive:inferred",
            validate=False,
        )

    try:
        await _auto_link_related_entities(
            knowledge_graph=knowledge_graph,
            embedding_func=rag_instance.embedding_func,
            new_entities=new_entities,
            document_id=document_id,
        )
    except Exception:
        pass
    try:
        await _auto_link_macro_documents(
            knowledge_graph=knowledge_graph,
            embedding_func=rag_instance.embedding_func,
            doc_entity=doc_entity,
            macro_text=macro_text,
            macro_terms=macro_terms,
            document_id=document_id,
        )
    except Exception:
        pass


async def _update_local_knowledge_base(
    text: str,
    metadata: Dict[str, Any],
    *,
    source_type: str,
    session_id: Optional[str],
) -> None:
    rag_instance = await _resolve_rag_for_session(session_id)
    if not rag_instance or not getattr(rag_instance, "knowledge_base_manager", None):
        return
    try:
        rag_instance.add_cognitive_memory(
            text=text,
            metadata=metadata,
            source_type=source_type,
            session_id=session_id,
        )
    except Exception:
        pass


async def _process_image_for_ingest(image_path: str) -> str:
    if not state.rag_instance or not getattr(state.rag_instance, "vision_model_func", None):
        raise RuntimeError("Vision model is not configured")
    prompt = "Describe key entities, events, and relationships in this image for knowledge graph extraction."
    description = await state.rag_instance.vision_model_func(prompt, image_data=image_path)
    return f"Source: image\nContent:\n{description}"


@router.post("/ingest/stream")
async def ingest_stream(request: IngestRequest, background_tasks: BackgroundTasks):
    if not state.ingestion_service:
        raise HTTPException(status_code=500, detail="Ingestion service not initialized")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    async def _process_and_ingest(content: str, source_type: str, session_id: Optional[str]):
        processed_text, metadata = await _process_text_for_ingest(content, source_type)
        try:
            await state.ingestion_service.ingest_text(processed_text, session_id=session_id)
        except Exception:
            pass
        try:
            await _update_local_knowledge_graph(processed_text, metadata, session_id=session_id)
        except Exception:
            pass
        try:
            await _update_local_knowledge_base(
                content,
                metadata,
                source_type=source_type,
                session_id=session_id,
            )
        except Exception:
            pass
        try:
            await _write_session_code_snapshot(
                session_id=session_id,
                source_type=source_type,
                source_name=f"stream_{source_type}",
                original_text=content,
                processed_text=processed_text,
                metadata=metadata,
            )
        except Exception:
            pass

    async def _stream_ingest_task(content: str, source_type: str, session_id: Optional[str]):
        await _process_and_ingest(content, source_type, session_id)

    background_tasks.add_task(_stream_ingest_task, request.content, request.source_type, request.session_id)
    return {"status": "processing", "message": "Stream accepted for cognitive processing"}


@router.post("/ingest/signal")
async def ingest_signal(request: SignalIngestRequest, background_tasks: BackgroundTasks):
    if not state.ingestion_service:
        raise HTTPException(status_code=500, detail="Ingestion service not initialized")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    async def _process_and_ingest(signal_text: str, source_type: str, session_id: Optional[str]):
        processed_text, metadata = await _process_text_for_ingest(signal_text, source_type)
        try:
            await state.ingestion_service.ingest_text(processed_text, session_id=session_id)
        except Exception:
            pass
        try:
            await _update_local_knowledge_graph(processed_text, metadata, session_id=session_id)
        except Exception:
            pass
        try:
            await _update_local_knowledge_base(
                signal_text,
                metadata,
                source_type=source_type,
                session_id=session_id,
            )
        except Exception:
            pass
        try:
            await _write_session_code_snapshot(
                session_id=session_id,
                source_type=source_type,
                source_name=f"signal_{source_type}",
                original_text=signal_text,
                processed_text=processed_text,
                metadata=metadata,
            )
        except Exception:
            pass

    signal_text = _build_signal_text(request)
    source_type = request.modality or request.source_type or "signal"
    background_tasks.add_task(_process_and_ingest, signal_text, source_type, request.session_id)
    return {"status": "processing", "message": "Signal accepted for cognitive processing"}


@router.post("/ingest")
async def ingest_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    request: Request = None,
):
    if not state.ingestion_service or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    session_id = await _extract_session_id(session_id, request)
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    uploaded_files: List[str] = []
    try:
        for file in files:
            file_path = _allocate_upload_target(session_id, file.filename)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            await file.close()
            uploaded_files.append(str(file_path))

            try:
                state.session_manager.add_document_record(
                    session_id,
                    file.filename,
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    file_ext=file_path.suffix.lower(),
                )
            except Exception:
                pass

        async def _background_file_processing(file_paths: List[str], sid: str):
            _bg_t0 = time.perf_counter()
            _log.info("[ingest] background processing started | files=%d sid=%s", len(file_paths), sid)
            try:
                for path_str in file_paths:
                    try:
                        if state.session_manager:
                            state.session_manager.set_document_status(
                                sid, Path(path_str).name, "processing"
                            )
                    except Exception:
                        pass

                image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
                text_exts = {".txt", ".md"}

                content_list_paths: List[Path] = []
                image_paths: List[Path] = []
                text_paths: List[Path] = []
                complex_paths: List[Path] = []

                for path_str in file_paths:
                    path = Path(path_str)
                    if path.name.endswith("_content_list.json"):
                        content_list_paths.append(path)
                    elif path.suffix.lower() in image_exts:
                        image_paths.append(path)
                    elif path.suffix.lower() in text_exts:
                        text_paths.append(path)
                    else:
                        complex_paths.append(path)

                # 1) Structured content-list files
                if content_list_paths:
                    grouped = _collect_content_list_groups(content_list_paths)
                    for doc_id, paths in grouped.items():
                        combined: List[dict[str, Any]] = []
                        for p in paths:
                            try:
                                content = _load_content_list(p)
                                if content:
                                    combined.extend(content)
                            except Exception:
                                pass

                        if not combined:
                            continue

                        lines: List[str] = []
                        for item in combined:
                            if isinstance(item, str):
                                if item.strip():
                                    lines.append(item.strip())
                                continue
                            if not isinstance(item, dict):
                                continue
                            for key in ("text", "content", "caption", "ocr_text", "title"):
                                value = item.get(key)
                                if isinstance(value, str) and value.strip():
                                    lines.append(value.strip())
                                    break

                        merged_text = "\n".join(lines).strip()
                        if not merged_text:
                            continue

                        processed_text, metadata = await _process_text_for_ingest(merged_text, "content_list")
                        await state.ingestion_service.ingest_text(processed_text, session_id=sid)
                        await _update_local_knowledge_graph(processed_text, metadata, session_id=sid)
                        await _update_local_knowledge_base(
                            merged_text,
                            metadata,
                            source_type="content_list",
                            session_id=sid,
                        )
                        await _write_session_code_snapshot(
                            session_id=sid,
                            source_type="content_list",
                            source_name=f"{doc_id}_content_list",
                            original_text=merged_text,
                            processed_text=processed_text,
                            metadata=metadata,
                        )
                    for p in content_list_paths:
                        try:
                            if state.session_manager:
                                state.session_manager.set_document_status(
                                    sid, p.name, "processed"
                                )
                        except Exception:
                            pass

                # 2) Images
                for img_path in image_paths:
                    try:
                        image_text = await _process_image_for_ingest(str(img_path))
                        processed_text, metadata = await _process_text_for_ingest(image_text, "image")
                        await state.ingestion_service.ingest_text(processed_text, session_id=sid)
                        await _update_local_knowledge_graph(processed_text, metadata, session_id=sid)
                        await _update_local_knowledge_base(
                            image_text,
                            metadata,
                            source_type="image",
                            session_id=sid,
                        )
                        await _write_session_code_snapshot(
                            session_id=sid,
                            source_type="image",
                            source_name=img_path.name,
                            original_text=image_text,
                            processed_text=processed_text,
                            metadata=metadata,
                        )
                        if state.session_manager:
                            state.session_manager.set_document_status(
                                sid, img_path.name, "processed"
                            )
                    except Exception:
                        if state.session_manager:
                            state.session_manager.set_document_status(
                                sid, img_path.name, "failed"
                            )

                # 3) Plain text files — skip CognitiveProcessor (GraphCore handles entity extraction)
                for txt_path in text_paths:
                    _txt_t0 = time.perf_counter()
                    _log.info("[TXT] start processing '%s' (%d chars)", txt_path.name, 0)
                    try:
                        content = _read_text_auto_encoding(txt_path)
                        _log.info("[TXT] '%s' read: %d chars", txt_path.name, len(content))

                        _t1 = time.perf_counter()
                        processed_text, metadata = await _process_text_for_ingest(
                            content, "file", skip_cognitive=True,
                        )
                        _log.info("[TXT T+%.2fs] cognitive/prep done", time.perf_counter() - _txt_t0)

                        _t2 = time.perf_counter()
                        await state.ingestion_service.ingest_text(
                            content, session_id=sid, file_path=txt_path.name,
                        )
                        _log.info("[TXT T+%.2fs] GraphCore ainsert done (%.2fs)",
                                  time.perf_counter() - _txt_t0, time.perf_counter() - _t2)

                        _t3 = time.perf_counter()
                        kg_task = _update_local_knowledge_graph(processed_text, metadata, session_id=sid)
                        kb_task = _update_local_knowledge_base(
                            content, metadata, source_type="file", session_id=sid,
                        )
                        snap_task = _write_session_code_snapshot(
                            session_id=sid,
                            source_type="file",
                            source_name=txt_path.name,
                            original_text=content,
                            processed_text=processed_text,
                            metadata=metadata,
                        )
                        await asyncio.gather(kg_task, kb_task, snap_task)
                        _log.info("[TXT T+%.2fs] KG+KB+snapshot done (%.2fs)",
                                  time.perf_counter() - _txt_t0, time.perf_counter() - _t3)

                        if state.session_manager:
                            state.session_manager.set_document_status(
                                sid, txt_path.name, "processed"
                            )
                        _log.info("[TXT T+%.2fs] '%s' TOTAL processing complete",
                                  time.perf_counter() - _txt_t0, txt_path.name)
                    except Exception as exc:
                        _log.error("[TXT T+%.2fs] '%s' FAILED: %s",
                                   time.perf_counter() - _txt_t0, txt_path.name, exc)
                        if state.session_manager:
                            state.session_manager.set_document_status(
                                sid, txt_path.name, "failed"
                            )

                # 4) Complex files (PDF/Doc): delegate to session-scoped ingestion pipeline once.
                if complex_paths:
                    try:
                        await state.ingestion_service.ingest_files(
                            [str(p) for p in complex_paths],
                            session_id=sid,
                        )
                        for p in complex_paths:
                            if state.session_manager:
                                state.session_manager.set_document_status(
                                    sid, p.name, "processed"
                                )
                    except Exception:
                        for p in complex_paths:
                            if state.session_manager:
                                state.session_manager.set_document_status(
                                    sid, p.name, "failed"
                                )
                        raise

                # ── Post-processing: sync KG entity IDs + density clustering ──
                await _sync_kg_entity_ids(sid)

                try:
                    await _run_density_clustering(sid)
                except Exception as exc:
                    _log.warning("[ingest] density clustering failed (non-fatal): %s", exc)

                _log.info("[ingest] background processing COMPLETE for session %s", sid)

                # Fire-and-forget: discover latent edges in the background.
                # This does NOT block the user — they can query immediately.
                asyncio.create_task(_background_edge_discovery(sid))

                # Fire-and-forget: KG self-study loop (test-time scaling on KG)
                asyncio.create_task(_background_self_study(sid))

            except Exception as e:
                _log.error("[ingest] background processing error: %s", e, exc_info=True)
                for path_str in file_paths:
                    try:
                        if state.session_manager:
                            state.session_manager.set_document_status(
                                sid, Path(path_str).name, "failed"
                            )
                    except Exception:
                        pass

        background_tasks.add_task(_background_file_processing, uploaded_files, session_id)

        return {
            "success": True,
            "status": "processing",
            "message": f"Uploaded {len(uploaded_files)} file(s), background processing started",
            "files": [{"name": Path(f).name} for f in uploaded_files],
            "session_id": session_id,
            "background_processing": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    request: Request = None,
):
    """Alias for /ingest to match frontend expectations"""
    return await ingest_files(background_tasks, files, session_id, request)
