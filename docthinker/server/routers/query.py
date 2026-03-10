import asyncio
import json
import os
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException

from docthinker.kg_expansion import ExpandedNodeManager, extract_entities_from_text

from ..memory import get_session_memory_engine
from ..schemas import MultiDocumentQueryRequest, QueryRequest
from ..state import state


router = APIRouter()

FAST_QA_TIMEOUT_SECONDS = 30
SESSION_QUERY_TIMEOUT_SECONDS = 180
FALLBACK_LLM_TIMEOUT_SECONDS = 60


def _looks_like_file_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    keywords = [
        "文件",
        "文档",
        "附件",
        "内容",
        "总结",
        "表格",
        "图片",
        "file",
        "document",
        "attachment",
        "uploaded",
        "summarize",
        "this doc",
        "this file",
        "pdf",
    ]
    return any(k in q for k in keywords)


def _is_chat_turn_ingest_enabled() -> bool:
    flag = str(os.getenv("ENABLE_CHAT_TURN_INGEST", "0")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _get_pending_session_files(session_id: Optional[str]) -> List[Dict[str, Any]]:
    if not session_id or not state.session_manager:
        return []
    try:
        files = state.session_manager.get_files(session_id)
    except Exception:
        return []
    done_statuses = {"processed", "completed", "complete", "done", "success", "ready", "finished"}
    pending: List[Dict[str, Any]] = []
    for item in files:
        status = str(item.get("status") or "").strip().lower()
        if status and status in done_statuses:
            continue
        pending.append(item)
    return pending


def _format_thinking_process(details: Dict[str, Any], meta: Dict[str, Any]) -> str:
    lines: List[str] = []
    if meta.get("memory_mode"):
        lines.append(f"memory_mode: {meta['memory_mode']}")
    if meta.get("retrieval_instruction"):
        lines.append("retrieval_instruction: provided")
    if details.get("memory_hits") is not None:
        lines.append(f"memory_hits: {details['memory_hits']}")
    if details.get("expanded_hits") is not None:
        lines.append(f"expanded_hits: {details['expanded_hits']}")
    if details.get("mode"):
        lines.append(f"mode: {details['mode']}")
    return "\n".join(lines)


def _build_sources_from_details(details: Any, evidence: Any = None) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    if isinstance(details, dict):
        sub_answers = details.get("sub_answers")
        if isinstance(sub_answers, list):
            for ans in sub_answers:
                if not isinstance(ans, dict):
                    continue
                answer = str(ans.get("answer") or "").strip()
                context = str(ans.get("context") or "").strip()
                if not answer and not context:
                    continue
                snippet = "\n".join(x for x in [answer, context] if x)
                sources.append({"content": snippet, "confidence": float(ans.get("confidence") or 0.5)})

    if not sources and isinstance(evidence, dict):
        raw_prompt = str(evidence.get("raw_prompt") or "").strip()
        if raw_prompt:
            snippet = raw_prompt if len(raw_prompt) <= 800 else f"{raw_prompt[:800]}..."
            sources.append({"content": snippet, "confidence": 0.4})

    return sources[:5]


def _load_content_list(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict) and "content_list" in data:
        content_list = data.get("content_list") or []
    elif isinstance(data, list):
        content_list = data
    else:
        return []

    base_dir = path.parent
    for block in content_list:
        if isinstance(block, dict) and "img_path" in block:
            img_path = Path(str(block["img_path"]))
            if not img_path.is_absolute():
                img_path = (base_dir / img_path).resolve()
            block["img_path"] = img_path.as_posix()
    return content_list


def _extract_text_from_content_list(content_list: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for block in content_list:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif btype == "table" and block.get("table_html"):
            parts.append(str(block["table_html"]))
        elif btype == "equation" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _count_pages_from_content_list(content_list: List[Dict[str, Any]]) -> int:
    max_page = -1
    for block in content_list:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx")
        if isinstance(page_idx, int):
            max_page = max(max_page, page_idx)
    return max_page + 1 if max_page >= 0 else 0


def _find_latest_content_list(output_dir: Path, stem: str) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = list(output_dir.rglob(f"{stem}_content_list.json"))
    if not candidates:
        # MinerU may emit ASCII-safe hashed stems when source filename is non-ASCII.
        candidates = list(output_dir.rglob("*_content_list.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_cjk_font(size: int = 18):
    """Try to locate a CJK-capable TrueType font on the system."""
    from PIL import ImageFont

    candidate_paths = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/mingliu.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for fp in candidate_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return None


def _render_text_to_image(text: str, output_dir: Path, name: str) -> Optional[Path]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    font_size = 18
    cjk_font = _find_cjk_font(font_size)
    using_cjk = cjk_font is not None
    font = cjk_font if using_cjk else ImageFont.load_default()

    chars_per_line = 60
    lines: List[str] = []
    for paragraph in text.replace("\r", "").split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        while paragraph:
            lines.append(paragraph[:chars_per_line])
            paragraph = paragraph[chars_per_line:]

    if not lines or len(lines) > 200:
        return None

    line_height = font_size + 8 if using_cjk else (font.getbbox("A")[3] - font.getbbox("A")[1] + 6)
    margin = 20
    width = 1200
    height = max(200, margin * 2 + line_height * len(lines))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height

    output_path = output_dir / f"{name}_quick_qa.png"
    image.save(output_path, "PNG")
    return output_path


async def _try_fast_qa(request: QueryRequest) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not request.session_id:
        return None, None
    if not state.session_manager or not state.rag_instance:
        return None, None
    if not getattr(state.rag_instance, "vision_model_func", None):
        return None, None

    files = state.session_manager.get_files(request.session_id)
    if not files:
        return None, None

    target = next((f for f in files if f.get("file_path")), None)
    if not target:
        return None, None

    file_path = Path(str(target.get("file_path")))
    if not file_path.exists():
        return None, None

    file_size = target.get("file_size")
    if not isinstance(file_size, int):
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = None

    file_ext = (target.get("file_ext") or file_path.suffix).lower()
    max_bytes = 300 * 1024
    max_chars = 4000
    max_pages = 2
    short_by_size = file_size is not None and file_size <= max_bytes

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    text_exts = {".txt", ".md"}

    image_path: Optional[Path] = None
    text_content = ""
    page_count = 0

    if file_ext in image_exts and short_by_size:
        image_path = file_path
    else:
        if file_ext in text_exts:
            try:
                text_content = file_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                text_content = ""
        else:
            try:
                output_dir = Path(state.rag_instance.config.parser_output_dir)
                session = state.session_manager.get_session(request.session_id)
                session_work_dir = Path((session or {}).get("path") or "")
                if not output_dir.is_absolute() and session_work_dir:
                    output_dir = session_work_dir / output_dir
                content_list_path = _find_latest_content_list(output_dir, file_path.stem)
                if content_list_path:
                    content_list = _load_content_list(content_list_path)
                    text_content = _extract_text_from_content_list(content_list)
                    page_count = _count_pages_from_content_list(content_list)
                    if not text_content:
                        for block in content_list:
                            if not isinstance(block, dict):
                                continue
                            if str(block.get("type") or "").lower() != "image":
                                continue
                            img = str(block.get("img_path") or "").strip()
                            if not img:
                                continue
                            candidate = Path(img)
                            if candidate.exists():
                                image_path = candidate
                                break
            except Exception:
                text_content = ""

        if text_content and len(text_content) <= max_chars:
            if file_ext in text_exts:
                prompt = (
                    f"以下是文档「{file_path.name}」的内容：\n\n"
                    f"{text_content}\n\n"
                    f"请根据文档内容回答用户问题：{request.question}"
                )
                try:
                    answer = await state.rag_instance.llm_model_func(prompt)
                except Exception:
                    return None, None
                if answer:
                    return answer, {"fast_qa": True, "file": file_path.name, "mode": "text_direct"}
                return None, None

            if (short_by_size or page_count <= max_pages) and text_content.strip() and image_path is None:
                session = state.session_manager.get_session(request.session_id)
                output_base = Path(session["path"]) if session and session.get("path") else Path(tempfile.mkdtemp())
                image_path = _render_text_to_image(text_content, output_base / "quick_qa", file_path.stem)

    if not image_path or not image_path.exists():
        return None, None

    prompt = f"Please answer the question based on the document image: {request.question}"
    try:
        answer = await state.rag_instance.vision_model_func(prompt, image_data=str(image_path))
    except Exception:
        return None, None

    if not answer:
        return None, None

    return answer, {"fast_qa": True, "file": file_path.name}


async def _get_session_rag_or_raise(session_id: Optional[str]):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required; global knowledge is disabled")
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    config = state.rag_instance.config
    graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
    session_rag = state.session_manager.get_session_rag(session_id, config, graphcore_kwargs)
    session_rag.llm_model_func = state.rag_instance.llm_model_func
    session_rag.embedding_func = state.rag_instance.embedding_func
    await session_rag._ensure_graphcore_initialized()
    return session_rag


async def _ingest_chat_turn(question: str, answer: str, session_id: Optional[str], timestamp: Optional[float] = None):
    if not state.ingestion_service:
        return
    text_to_ingest = f"User Question: {question}\nAssistant Answer: {answer}"
    if session_id:
        try:
            await state.ingestion_service.ingest_text(text_to_ingest, session_id=session_id)
        except Exception:
            pass


def _get_expanded_node_manager(session_id: Optional[str]) -> Optional[ExpandedNodeManager]:
    if not session_id or not state.session_manager:
        return None
    session = state.session_manager.get_session(session_id)
    if not session:
        return None

    metadata = session.get("metadata") or {}
    knowledge_dir = metadata.get("knowledge_dir") or session.get("knowledge_dir")
    if not knowledge_dir:
        return None

    if not hasattr(state, "expanded_node_managers") or state.expanded_node_managers is None:
        state.expanded_node_managers = {}
    if not hasattr(state, "expanded_node_lock") or state.expanded_node_lock is None:
        from threading import RLock

        state.expanded_node_lock = RLock()

    with state.expanded_node_lock:
        manager = state.expanded_node_managers.get(session_id)
        if manager is not None:
            return manager
        manager = ExpandedNodeManager(Path(str(knowledge_dir)) / "expanded_nodes.json")
        state.expanded_node_managers[session_id] = manager
        return manager


def _merge_retrieval_instruction(*instructions: Optional[str]) -> str:
    parts = []
    for ins in instructions:
        text = str(ins or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


async def _promote_expanded_nodes_in_background(
    session_id: Optional[str],
    question: str,
    answer: str,
    matched_nodes: List[Dict[str, Any]],
):
    if not session_id or not matched_nodes:
        return

    manager = _get_expanded_node_manager(session_id)
    if manager is None:
        return

    entities = extract_entities_from_text(answer, max_entities=12)
    usage = manager.record_response_usage(answer=answer, matches=matched_nodes, attached_entities=entities)
    promoted_names = usage.get("promoted") or []
    if not promoted_names:
        return

    try:
        session_rag = await _get_session_rag_or_raise(session_id)
        G = session_rag.graphcore.chunk_entity_relation_graph
        changed = False

        for name in promoted_names:
            record = manager.get(name)
            if not record:
                continue
            roots = [str(x).strip() for x in (record.get("root_ids") or []) if str(x).strip()]

            await G.upsert_node(
                name,
                {
                    "entity_id": name,
                    "entity_type": "concept",
                    "description": record.get("reason") or name,
                    "source_id": "promoted_expansion",
                    "is_expanded": "0",
                },
            )
            changed = True

            for ent in entities[:8]:
                if not ent or ent == name:
                    continue
                await G.upsert_node(
                    ent,
                    {
                        "entity_id": ent,
                        "entity_type": "concept",
                        "description": f"Extracted from answer for expansion node {name}",
                        "source_id": "answer_entity",
                    },
                )
                await G.upsert_edge(
                    name,
                    ent,
                    {
                        "keywords": "co_mentioned",
                        "description": f"Assistant answer associated {name} with {ent}",
                        "source_id": "answer_entity",
                    },
                )
                changed = True

            for root in roots[:6]:
                if not root or root == name:
                    continue
                await G.upsert_edge(
                    name,
                    root,
                    {
                        "keywords": "expanded_from_root",
                        "description": f"Promoted expansion node linked to root node {root}",
                        "source_id": "llm_expansion",
                    },
                )
                changed = True

        if changed and hasattr(G, "index_done_callback"):
            try:
                await G.index_done_callback(force_save=True)
            except TypeError:
                await G.index_done_callback()
    except Exception:
        return


@router.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    print(f"DEBUG: Received query: {request.question}, session_id: {request.session_id}")
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required; global knowledge is disabled")

    session_rag = await _get_session_rag_or_raise(request.session_id)
    effective_memory_mode = "session"

    identity_keywords = ["who are you", "your name", "hello", "hi"]
    question_lower = (request.question or "").lower()
    is_identity_query = any(k in question_lower for k in identity_keywords) and len(request.question) < 40

    try:
        state.session_manager.add_message(request.session_id, "user", request.question)
    except Exception:
        pass

    if _looks_like_file_question(request.question):
        pending_files = _get_pending_session_files(request.session_id)
        if pending_files:
            pending_names = [str(x.get("filename") or "unknown") for x in pending_files[:3]]
            suffix = " ..." if len(pending_files) > 3 else ""
            return {
                "answer": (
                    "文件正在后台解析，请稍后再问。\n"
                    f"当前待完成: {', '.join(pending_names)}{suffix}"
                ),
                "query": request.question,
                "mode": request.mode,
                "session_id": request.session_id,
                "memory_mode": effective_memory_mode,
                "thinking_process": "pending_file_processing",
                "sources": [],
                "answer_mode": "pending_file_processing",
                "expanded_matches": [],
                "retrieval_instruction_applied": False,
            }

    answer = ""
    sources: List[Dict[str, Any]] = []
    thinking_process: Optional[str] = None
    answer_mode = "rag"

    expanded_matches: List[Dict[str, Any]] = []
    expanded_instruction = ""
    merged_instruction = str(request.retrieval_instruction or "").strip()
    expanded_manager: Optional[ExpandedNodeManager] = None

    if request.enable_expanded_matching and not is_identity_query:
        expanded_manager = _get_expanded_node_manager(request.session_id)
        if expanded_manager:
            expanded_matches = expanded_manager.match_nodes(
                query=request.question,
                top_k=max(1, request.expanded_top_k),
                min_score=max(0.0, request.expanded_min_score),
            )
            if expanded_matches:
                expanded_manager.mark_hits([m.get("entity", "") for m in expanded_matches])
                expanded_instruction = expanded_manager.build_forced_instruction(
                    expanded_matches,
                    limit=min(2, max(1, request.expanded_top_k)),
                )
                merged_instruction = _merge_retrieval_instruction(request.retrieval_instruction, expanded_instruction)

    try:
        if is_identity_query:
            system_prompt = (
                "You are DocThinker, an assistant for document understanding, knowledge retrieval, and "
                "structured reasoning. Answer briefly and clearly."
            )
            llm_resp = await state.rag_instance.llm_model_func(
                f"{system_prompt}\n\nUser: {request.question}\nAssistant:"
            )
            answer = llm_resp if llm_resp else "I can help with document understanding and knowledge retrieval."
            answer_mode = "identity"

        elif request.session_id and _looks_like_file_question(request.question):
            try:
                fast_answer, fast_meta = await asyncio.wait_for(_try_fast_qa(request), timeout=FAST_QA_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                fast_answer, fast_meta = None, None
            if fast_answer:
                answer = fast_answer
                answer_mode = "fast_qa"
                thinking_process = "fast_qa"
                sources = (
                    [{"content": f"document: {fast_meta.get('file', '')}", "confidence": 0.6}] if fast_meta else []
                )

        if not answer and request.enable_thinking:
            analogies: List[Tuple[Any, float, Optional[str]]] = []
            memory_terms: List[str] = []
            memory_engine = get_session_memory_engine(request.session_id)
            if memory_engine:
                try:
                    analogies = await memory_engine.retrieve_analogies(
                        request.question,
                        top_k=5,
                        then_spread=True,
                        spread_top_k=3,
                    )
                    for ep, _, _ in analogies[:5]:
                        memory_terms.extend(list(ep.concepts or [])[:6])
                        memory_terms.extend(list(ep.entity_ids or [])[:6])
                    ep_ids = [ep.episode_id for ep, _, _ in analogies]
                    kg_ids = getattr(state, "kg_entity_ids", set())
                    ent_ids: List[str] = []
                    for ep, _, _ in analogies:
                        ent_ids.extend(ep.entity_ids or [])
                    ent_ids = [e for e in dict.fromkeys(ent_ids) if e and e in kg_ids]
                    if ep_ids or ent_ids:
                        try:
                            memory_engine.record_co_activation(ep_ids, ent_ids)
                            memory_engine.save()
                        except Exception:
                            pass
                except Exception:
                    pass

            if request.enable_expanded_matching and expanded_manager:
                refined = expanded_manager.match_nodes(
                    query=request.question,
                    top_k=max(1, request.expanded_top_k),
                    memory_terms=memory_terms,
                    min_score=max(0.0, request.expanded_min_score),
                )
                if refined:
                    expanded_matches = refined
                    expanded_manager.mark_hits([m.get("entity", "") for m in expanded_matches])
                    expanded_instruction = expanded_manager.build_forced_instruction(
                        expanded_matches,
                        limit=min(2, max(1, request.expanded_top_k)),
                    )
                    merged_instruction = _merge_retrieval_instruction(request.retrieval_instruction, expanded_instruction)

            try:
                answer = await asyncio.wait_for(
                    session_rag.aquery(
                        query=request.question,
                        mode=request.mode,
                        enable_rerank=request.enable_rerank,
                        enable_image_asset_activation=request.enable_image_asset_activation,
                        image_activation_threshold=request.image_activation_threshold,
                        image_activation_top_k=request.image_activation_top_k,
                        user_prompt=merged_instruction or None,
                    ),
                    timeout=SESSION_QUERY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                answer = ""
            except Exception:
                answer = ""

            answer_mode = "session_thinking"
            thinking_process = _format_thinking_process(
                {
                    "memory_hits": len(analogies),
                    "mode": request.mode,
                    "expanded_hits": len(expanded_matches),
                },
                {
                    "memory_mode": effective_memory_mode,
                    "retrieval_instruction": merged_instruction,
                },
            )
            if hasattr(session_rag, "get_last_query_evidence"):
                sources = _build_sources_from_details({}, session_rag.get_last_query_evidence())

        elif not answer:
            try:
                answer = await asyncio.wait_for(
                    session_rag.aquery(
                        query=request.question,
                        mode=request.mode,
                        enable_rerank=request.enable_rerank,
                        enable_image_asset_activation=request.enable_image_asset_activation,
                        image_activation_threshold=request.image_activation_threshold,
                        image_activation_top_k=request.image_activation_top_k,
                        user_prompt=merged_instruction or None,
                    ),
                    timeout=SESSION_QUERY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                answer = ""

        if not answer:
            answer = "抱歉，我暂时没有检索到足够信息。"

        negative_responses = ["i don't know", "不知道", "没找到", "sorry", "抱歉", "无法回答"]
        if (
            not is_identity_query
            and any(n in (answer or "").lower() for n in negative_responses)
            and len(answer or "") < 120
        ):
            fallback_prompt = (
                f"用户问题: {request.question}\n"
                "请给出一个清晰、可执行的回答；若信息不足，请说明缺口并给出下一步建议。"
            )
            try:
                llm_resp = await asyncio.wait_for(
                    state.rag_instance.llm_model_func(fallback_prompt),
                    timeout=FALLBACK_LLM_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                llm_resp = None
            if llm_resp:
                answer = llm_resp

        try:
            state.session_manager.add_message(request.session_id, "assistant", answer)
        except Exception:
            pass

        chat_turn_ts = time.time()
        if _is_chat_turn_ingest_enabled():
            background_tasks.add_task(
                _ingest_chat_turn,
                request.question,
                answer,
                request.session_id,
                chat_turn_ts,
            )
        if expanded_matches:
            background_tasks.add_task(
                _promote_expanded_nodes_in_background,
                request.session_id,
                request.question,
                answer,
                expanded_matches,
            )

        if not sources and hasattr(session_rag, "get_last_query_evidence"):
            evidence = session_rag.get_last_query_evidence()
            sources = _build_sources_from_details({}, evidence)

        return {
            "answer": answer,
            "query": request.question,
            "mode": request.mode,
            "session_id": request.session_id,
            "memory_mode": effective_memory_mode,
            "thinking_process": thinking_process,
            "sources": sources,
            "answer_mode": answer_mode,
            "expanded_matches": expanded_matches[: min(2, len(expanded_matches))],
            "retrieval_instruction_applied": bool(merged_instruction),
        }
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "api_key" in err_msg or "authentication" in err_msg:
            raise HTTPException(status_code=401, detail="API key invalid or missing.")
        if "403" in err_msg:
            raise HTTPException(status_code=403, detail="API access forbidden.")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/text")
async def query_text(request: QueryRequest, background_tasks: BackgroundTasks):
    """Alias for /query to match frontend expectations."""
    return await query(request, background_tasks)


@router.post("/query/multi-document")
async def query_multi_document(request: MultiDocumentQueryRequest):
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    session_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        result = await session_rag.aquery_multi_document_enhanced(
            query=request.question,
            mode=request.mode,
            enable_rerank=request.enable_rerank,
        )
        return {
            "answer": result["answer"],
            "query": request.question,
            "mode": request.mode,
            "related_documents": result["related_documents"],
            "extracted_entities": result["extracted_entities"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
