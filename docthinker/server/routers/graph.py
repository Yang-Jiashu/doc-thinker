from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse

from ..schemas import EntityRelationshipRequest, RelationshipRequest
from ..state import state
from ..memory import get_session_memory_engine
from docthinker.kg_expansion import ExpandedNodeManager
from docthinker.image_assets import is_image_node, resolve_graph_node_color


router = APIRouter()


def _get_memory_engine_or_raise(session_id: Optional[str]):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    engine = get_session_memory_engine(session_id)
    if engine is None:
        raise HTTPException(status_code=501, detail="Memory engine not initialized")
    return engine


async def _get_session_rag_or_raise(session_id: Optional[str]):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        return session_rag
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")


def _get_expanded_node_manager_or_raise(session_id: Optional[str]) -> ExpandedNodeManager:
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    session = state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    metadata = session.get("metadata") or {}
    knowledge_dir = metadata.get("knowledge_dir") or session.get("knowledge_dir")
    if not knowledge_dir:
        raise HTTPException(status_code=500, detail="knowledge_dir not found in session metadata")

    if not hasattr(state, "expanded_node_managers") or state.expanded_node_managers is None:
        state.expanded_node_managers = {}
    if not hasattr(state, "expanded_node_lock") or state.expanded_node_lock is None:
        from threading import RLock

        state.expanded_node_lock = RLock()

    storage_path = Path(str(knowledge_dir)) / "expanded_nodes.json"
    lock = state.expanded_node_lock
    with lock:
        mgr = state.expanded_node_managers.get(session_id)
        if mgr is None:
            mgr = ExpandedNodeManager(storage_path=storage_path)
            state.expanded_node_managers[session_id] = mgr
        return mgr


def _pick_root_entity_ids(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    *,
    limit: int = 6,
) -> List[str]:
    degree: Dict[str, int] = {}
    for edge in edges_data:
        src = str(edge.get("source") or "").strip()
        tgt = str(edge.get("target") or "").strip()
        if src:
            degree[src] = degree.get(src, 0) + 1
        if tgt:
            degree[tgt] = degree.get(tgt, 0) + 1

    candidates = []
    for n in nodes_data:
        entity = str(n.get("id") or n.get("entity_id") or "").strip()
        if not entity:
            continue
        is_expanded = str(n.get("source_id") or "").strip() == "llm_expansion" or str(
            n.get("is_expanded") or ""
        ).strip() in {"1", "true", "True"}
        if is_expanded:
            continue
        candidates.append((degree.get(entity, 0), entity))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [entity for _, entity in candidates[: max(1, int(limit))]]


def _is_expanded_node(node_data: Dict[str, Any]) -> bool:
    ie = node_data.get("is_expanded")
    if ie is not None and ie != "":
        if ie == 1 or ie == "1" or str(ie).strip() == "1":
            return True
    return str(node_data.get("source_id") or "").strip() == "llm_expansion"


@router.post("/config")
async def update_config(payload: Dict[str, Any] = Body(...)):
    """Update system configuration"""
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    config_type = payload.get("type")
    config_data = payload.get("data", {})
    
    try:
        if config_type == "kg":
            # Update Knowledge Graph configuration
            if "kg-storage" in config_data:
                state.rag_instance.config.knowledge_graph_storage_type = config_data["kg-storage"]
            
            if "kg-path" in config_data:
                state.rag_instance.config.knowledge_graph_path = config_data["kg-path"]
            
            if "entity-threshold" in config_data:
                state.rag_instance.config.entity_disambiguation_threshold = float(config_data["entity-threshold"])
            
            if "rel-threshold" in config_data:
                state.rag_instance.config.relationship_validation_threshold = float(config_data["rel-threshold"])
                
            if "enable-auto-validation" in config_data:
                state.rag_instance.config.enable_auto_validation = config_data["enable-auto-validation"] == "on"
            
            # New dual mode parameters
            if "graph-construction-mode" in config_data:
                state.rag_instance.config.graph_construction_mode = config_data["graph-construction-mode"]
                # Also update orchestrator if it exists
                if hasattr(state, "orchestrator") and state.orchestrator:
                    if hasattr(state.orchestrator, "hyper_system") and state.orchestrator.hyper_system:
                        state.orchestrator.hyper_system.graph_construction_mode = config_data["graph-construction-mode"]
            
            if "spacy-model" in config_data:
                state.rag_instance.config.spacy_model = config_data["spacy-model"]
                # Also update orchestrator if it exists
                if hasattr(state, "orchestrator") and state.orchestrator:
                    if hasattr(state.orchestrator, "hyper_system") and state.orchestrator.hyper_system:
                        state.orchestrator.hyper_system.spacy_model = config_data["spacy-model"]

            return {"success": True, "message": "Knowledge graph configuration updated"}
            
        elif config_type == "ui":
            # UI config might not be directly updateable in backend RAG instance
            # but we could store it if needed
            return {"success": True, "message": "UI configuration received (not all fields are persistent)"}
            
        elif config_type == "api":
            # API config might require restart to take effect
            return {"success": True, "message": "API configuration received (restart may be required)"}
            
        else:
            return {"success": False, "message": f"Unknown configuration type: {config_type}"}
            
    except Exception as e:
        return {"success": False, "message": f"Error updating configuration: {str(e)}"}


@router.get("/knowledge-graph/stats-all")
async def get_all_graph_stats():
    """Return node/edge counts for all session graphs."""
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    result: Dict[str, Any] = {"sessions": {}}
    for s in state.session_manager.list_sessions():
        sid = s.get("id")
        if not sid:
            continue
        try:
            config = state.rag_instance.config
            graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
            session_rag = state.session_manager.get_session_rag(sid, config, graphcore_kwargs)
            session_rag.llm_model_func = state.rag_instance.llm_model_func
            session_rag.embedding_func = state.rag_instance.embedding_func
            await session_rag._ensure_graphcore_initialized()
            SG = session_rag.graphcore.chunk_entity_relation_graph
            snd = await SG.get_all_nodes()
            sed = await SG.get_all_edges()
            result["sessions"][sid] = {
                "nodes": len(snd),
                "edges": len(sed),
                "title": s.get("title", "unknown"),
            }
            if hasattr(SG, "_graphml_xml_file"):
                result["sessions"][sid]["path"] = str(getattr(SG, "_graphml_xml_file", ""))
        except Exception as e:
            result["sessions"][sid] = {"error": str(e), "title": s.get("title", "unknown")}
    return result


@router.get("/knowledge-graph/data")
async def get_graph_data(session_id: Optional[str] = None):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(session_id, config, graphcore_kwargs)
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session graph not found: {e}")

    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()

        nodes = []
        edges = []
        max_nodes = 5000

        img_path_by_id: Dict[str, str] = {}
        for n in nodes_data:
            nid = n.get("id") or n.get("entity_id") or ""
            raw = str(n.get("img_path") or "").strip()
            if nid and raw and Path(raw).is_file():
                img_path_by_id[nid] = raw

        asset_img_for: Dict[str, str] = {}
        for e in edges_data:
            kw = str(e.get("keywords") or "").strip().lower()
            if kw in {"image_asset_of", "image_related_to"}:
                src = e.get("source", "")
                tgt = e.get("target", "")
                if src in img_path_by_id and tgt not in img_path_by_id:
                    asset_img_for.setdefault(tgt, img_path_by_id[src])
                elif tgt in img_path_by_id and src not in img_path_by_id:
                    asset_img_for.setdefault(src, img_path_by_id[tgt])

        expanded_nodes = [n for n in nodes_data if _is_expanded_node(n)]
        other_nodes = [n for n in nodes_data if not _is_expanded_node(n)]
        nodes_to_use = expanded_nodes + other_nodes[: max(0, max_nodes - len(expanded_nodes))]
        for node_info in nodes_to_use:
            node_id = node_info.get("id") or node_info.get("entity_id") or ""
            if not node_id:
                continue
            is_expanded = _is_expanded_node(node_info)
            _is_img = is_image_node(node_info)
            node_entry: Dict[str, Any] = {
                "id": node_id,
                "label": node_id,
                "type": node_info.get("entity_type", "unknown"),
                "size": 20,
                "color": resolve_graph_node_color(
                    node_info, is_expanded=is_expanded
                ),
                "is_expanded": is_expanded,
                "is_image_node": _is_img,
            }
            if _is_img:
                resolved_path = img_path_by_id.get(node_id) or asset_img_for.get(node_id) or ""
                if resolved_path:
                    node_entry["has_image"] = True
                    node_entry["image_file"] = Path(resolved_path).name
                else:
                    node_entry["has_image"] = False
            nodes.append(node_entry)

        node_ids = set(n["id"] for n in nodes)
        for edge_info in edges_data:
            u = edge_info["source"]
            v = edge_info["target"]
            if u in node_ids and v in node_ids:
                edges.append(
                    {
                        "id": f"{u}-{v}",
                        "source": u,
                        "target": v,
                        "label": edge_info.get("keywords", "related"),
                        "description": edge_info.get("description", ""),
                        "color": "#95a5a6",
                        "width": 1,
                    }
                )

        expanded_in_response = sum(1 for x in nodes if x.get("is_expanded"))
        image_nodes_in_response = sum(1 for x in nodes if x.get("is_image_node"))
        meta = {
            "total_nodes": len(nodes_data),
            "total_edges": len(edges_data),
            "session_id": session_id,
            "nodes_returned": len(nodes),
            "expanded_in_response": expanded_in_response,
            "image_nodes_in_response": image_nodes_in_response,
        }
        if hasattr(G, "_graphml_xml_file"):
            meta["graph_file"] = str(getattr(G, "_graphml_xml_file", ""))
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": meta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract graph data: {str(e)}")


@router.get("/knowledge-graph/image/{session_id}/{filename}")
async def serve_image_node(session_id: str, filename: str):
    """Serve an image file from a session's multimodal/images directory."""
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    session = state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    knowledge_dir = session.get("knowledge_dir") or ""
    if not knowledge_dir:
        raise HTTPException(status_code=404, detail="Knowledge dir not found")

    safe_name = Path(filename).name
    img_path = Path(knowledge_dir) / "multimodal" / "images" / safe_name

    if not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image file not found: {safe_name}")

    _MIME = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }
    mime = _MIME.get(img_path.suffix.lower(), "image/png")
    return FileResponse(str(img_path), media_type=mime)


@router.post("/knowledge-graph/expand")
async def expand_knowledge_graph(payload: Dict[str, Any] = Body(default={})):
    """Expand a session knowledge graph with LLM-generated candidate nodes."""
    session_id = payload.get("session_id")
    angle_indices = payload.get("angle_indices")
    apply = payload.get("apply", True)
    root_entity_ids = payload.get("root_entity_ids")
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")
    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {e}")

    llm_fn = getattr(target_rag, "llm_model_func", None)
    if not llm_fn:
        raise HTTPException(status_code=500, detail="LLM not available")
    embed_fn = getattr(target_rag, "embedding_func", None)
    if embed_fn and hasattr(embed_fn, "func"):
        embed_fn = embed_fn.func

    try:
        from docthinker.kg_expansion import KGExpander

        expander = KGExpander(
            llm_func=llm_fn,
            embedding_func=embed_fn,
            min_per_angle=15,
            semantic_dedup_threshold=1.0,
        )
        result = await expander.expand(
            nodes_data,
            edges_data,
            angle_indices=angle_indices if angle_indices is not None else [0, 1, 5],
            apply_to_graph=G if apply else None,
            session_id=session_id,
        )
        manager = _get_expanded_node_manager_or_raise(session_id)
        if isinstance(root_entity_ids, list) and root_entity_ids:
            root_ids = [str(x).strip() for x in root_entity_ids if str(x).strip()]
        else:
            root_ids = _pick_root_entity_ids(nodes_data, edges_data)

        lifecycle = manager.upsert_candidates(
            [*(result.get("added") or []), *(result.get("suggested") or [])],
            default_root_ids=root_ids,
            source="llm_expansion",
        )

        return {
            "success": True,
            **result,
            "root_entity_ids": root_ids,
            "lifecycle": lifecycle,
        }
    except Exception as e:
        err = str(e)
        raise HTTPException(status_code=500, detail=err or "Expansion failed")


@router.get("/knowledge-graph/debug-expanded")
async def debug_expanded_nodes(session_id: Optional[str] = None):
    """Return diagnostics for expanded nodes in a session graph."""
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")

    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()

        expanded = [
            {"id": n.get("id") or n.get("entity_id"), "is_expanded": n.get("is_expanded")}
            for n in nodes_data
            if _is_expanded_node(n)
        ]
        total = len(nodes_data)
        storage_info = {}
        if hasattr(G, "_graphml_xml_file"):
            storage_info["graph_file"] = getattr(G, "_graphml_xml_file", "N/A")

        return {
            "expanded_count": len(expanded),
            "total_nodes": total,
            "expanded_sample": expanded[:20],
            "storage_info": storage_info,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats(session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
        entity_types = sorted(
            {
                str(n.get("entity_type") or "unknown")
                for n in nodes_data
            }
        )
        relationship_types = sorted(
            {
                str(e.get("keywords") or e.get("description") or "related")
                for e in edges_data
            }
        )
        return {
            "session_id": session_id,
            "total_entities": len(nodes_data),
            "total_relationships": len(edges_data),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/expanded-nodes")
async def list_expanded_nodes(
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
):
    manager = _get_expanded_node_manager_or_raise(session_id)
    nodes = manager.list_nodes(status=status, limit=limit)
    return {
        "session_id": session_id,
        "status": status,
        "count": len(nodes),
        "nodes": nodes,
    }


@router.post("/knowledge-graph/expanded-nodes/match")
async def match_expanded_nodes(payload: Dict[str, Any] = Body(default={})):
    session_id = payload.get("session_id")
    query = str(payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or 2)
    memory_terms = payload.get("memory_terms") or []
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    manager = _get_expanded_node_manager_or_raise(session_id)
    matches = manager.match_nodes(
        query=query,
        top_k=max(1, top_k),
        memory_terms=memory_terms if isinstance(memory_terms, list) else [],
    )
    if matches:
        manager.mark_hits([m.get("entity", "") for m in matches])
    instruction = manager.build_forced_instruction(matches, limit=min(2, max(1, top_k)))
    return {
        "session_id": session_id,
        "query": query,
        "count": len(matches),
        "matches": matches,
        "instruction": instruction,
    }


@router.post("/knowledge-graph/entity")
async def add_entity(request: EntityRelationshipRequest):
    target_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        props = dict(request.properties or {})
        props.setdefault("entity_type", request.entity_type)
        props.setdefault("source_id", request.document_id)
        await G.upsert_node(request.entity_name, props)
        await G.index_done_callback()
        return {
            "status": "success",
            "entity": {
                "id": request.entity_name,
                "name": request.entity_name,
                "type": props.get("entity_type", request.entity_type),
                "properties": props,
                "document_ids": [request.document_id] if request.document_id else [],
                "session_id": request.session_id,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/relationship")
async def add_relationship(request: RelationshipRequest):
    target_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if not await G.has_node(request.source_entity) or not await G.has_node(request.target_entity):
            raise HTTPException(status_code=404, detail="Source or target entity not found")
        props = dict(request.properties or {})
        props.setdefault("keywords", request.relationship_type)
        props.setdefault("description", request.relationship_type)
        props.setdefault("source_id", request.document_id)
        await G.upsert_edge(request.source_entity, request.target_entity, props)
        await G.index_done_callback()
        return {
            "status": "success",
            "relationship": {
                "id": f"{request.source_entity}-{request.target_entity}",
                "source_id": request.source_entity,
                "target_id": request.target_entity,
                "type": request.relationship_type,
                "properties": props,
                "document_ids": [request.document_id] if request.document_id else [],
                "session_id": request.session_id,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-graph/entity/{entity_name}")
async def update_entity(entity_name: str, properties: Dict[str, Any], session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if await G.has_node(entity_name):
            await G.upsert_node(entity_name, properties)
            await G.index_done_callback()
            return {"status": "success", "message": f"Entity {entity_name} updated", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Entity not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-graph/relationship")
async def delete_relationship(source: str, target: str, session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if await G.has_edge(source, target):
            await G.remove_edges([(source, target)])
            await G.index_done_callback()
            return {"status": "success", "message": f"Relationship {source}->{target} deleted", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Relationship not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
async def memory_stats(session_id: Optional[str] = None):
    """Memory engine status summary."""
    engine = _get_memory_engine_or_raise(session_id)
    try:
        episodes = engine.episode_store.all_episodes()
        edges = engine.graph.get_all_edges()
        return {
            "enabled": True,
            "session_id": session_id,
            "episodes": len(episodes),
            "edges": len(edges),
        }
    except Exception as e:
        return {"enabled": True, "session_id": session_id, "error": str(e)}


@router.get("/memory/graph-data")
async def memory_graph_data(session_id: Optional[str] = None):
    """Graph payload for memory visualization."""
    engine = _get_memory_engine_or_raise(session_id)
    try:
        graph = engine.graph
        episodes = engine.episode_store.all_episodes()
        nodes = []
        edges = []
        type_color = {"episode": "#3498db", "entity": "#2ecc71", "chunk": "#e67e22"}
        type_label = {"episode": "episode", "entity": "entity", "chunk": "chunk"}
        for nid, nd in graph.get_all_nodes():
            ntype = nd.get("type", "episode")
            label = nid
            if ntype == "episode" and nid in episodes:
                summary_text = episodes[nid].summary or ""
                summary = (summary_text or nid)[:30]
                label = summary + "..." if len(summary_text) > 30 else (summary_text or nid)
            nodes.append(
                {
                    "id": nid,
                    "label": label,
                    "type": type_label.get(ntype, ntype),
                    "size": 20,
                    "color": type_color.get(ntype, "#95a5a6"),
                }
            )
        for e in graph.get_all_edges():
            edges.append(
                {
                    "id": f"{e.source_id}-{e.edge_type.value}-{e.target_id}",
                    "source": e.source_id,
                    "target": e.target_id,
                    "label": e.edge_type.value,
                    "type": e.edge_type.value,
                    "color": "#9b59b6",
                    "width": max(1, int(e.weight * 3)),
                }
            )
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "source": "memory",
                "enabled": True,
                "session_id": session_id,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }
    except Exception as e:
        return {
            "nodes": [],
            "edges": [],
            "metadata": {"source": "memory", "enabled": True, "session_id": session_id, "error": str(e)},
        }


@router.post("/memory/consolidate")
async def memory_consolidate(session_id: Optional[str] = None, recent_n: int = 50, run_llm: bool = True):
    """Trigger one memory consolidation pass."""
    engine = _get_memory_engine_or_raise(session_id)
    try:
        result = await engine.consolidate(
            recent_n=recent_n,
            run_llm=run_llm,
        )
        try:
            dp = engine.decay_and_prune(
                decay_factor=0.9,
                max_age_days=30.0,
                min_weight=0.05,
            )
            result["decayed"] = dp.get("decayed", 0)
            result["pruned"] = dp.get("pruned", 0)
        except Exception:
            pass
        engine.save()
        return {"success": True, "session_id": session_id, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/decay-prune")
async def memory_decay_prune(
    session_id: Optional[str] = None,
    decay_factor: float = 0.9,
    max_age_days: float = 30.0,
    min_weight: float = 0.05,
):
    """Run memory edge decay and pruning."""
    engine = _get_memory_engine_or_raise(session_id)
    try:
        result = engine.decay_and_prune(
            decay_factor=decay_factor,
            max_age_days=max_age_days,
            min_weight=min_weight,
        )
        engine.save()
        return {"success": True, "session_id": session_id, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
