"""
Utility functions for DocThinker

Contains helper functions for content separation, text insertion, API integrations,
and other utilities.
"""
#分离文本和多模态内容，插入纯文本和多模态内容到graphcore.coregraph中，对重排序内容处理分批，返回格式，帮助graphcore.coregraph来实现。
import asyncio
import base64
import json
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import aiohttp

from graphcore.coregraph.utils import logger


def _remap_missing_image_path(image_path: str) -> Path | None:
    """
    Try to remap an outdated absolute image path to a new base directory.

    Strategy:
    1) If OLD_DATA_ROOT/NEW_DATA_ROOT are set and the path is under OLD_DATA_ROOT,
       rebuild under NEW_DATA_ROOT.
    2) If the path contains a "data" segment, rebuild it under the current project's
       data directory (or DATA_ROOT_OVERRIDE env) and see if it exists.
    """
    try:
        path = Path(image_path)
        if path.exists():
            return path

        old_root_env = os.getenv("OLD_DATA_ROOT")
        new_root_env = os.getenv("NEW_DATA_ROOT")
        if old_root_env and new_root_env:
            try:
                rel = path.relative_to(Path(old_root_env))
                candidate = Path(new_root_env) / rel
                logger.debug(f"Remap candidate (OLD/NEW): {candidate}")
                if candidate.exists():
                    return candidate
            except Exception as exc:
                logger.debug(f"OLD/NEW remap skipped: {exc}")

        parts = path.parts
        if "data" in parts:
            idx = parts.index("data")
            rel_after_data = Path(*parts[idx + 1 :])
            data_root_env = os.getenv("DATA_ROOT_OVERRIDE")
            default_data_root = Path(data_root_env) if data_root_env else Path(__file__).resolve().parents[1] / "data"
            candidate = default_data_root / rel_after_data
            logger.debug(f"Remap candidate (data-root): {candidate}")
            if candidate.exists():
                return candidate
    except Exception as exc:
        logger.debug(f"Remap failed: {exc}")
        return None
    return None


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content with optional page marker for later citation
            text = item.get("text", "")
            if text.strip():
                page_idx = item.get("page_idx")
                if page_idx is not None:
                    text_parts.append(f"[page_idx:{page_idx}] {text}")
                else:
                    text_parts.append(text)
        else:
            # Skip empty footers/headers — they waste LLM calls for zero value
            if content_type in ("footer", "header") and not (item.get("text") or "").strip():
                logger.debug(f"  Skipping empty {content_type} block (page {item.get('page_idx', '?')})")
                continue
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        path = Path(image_path)
        if not path.exists():
            remapped = _remap_missing_image_path(image_path)
            if remapped and remapped.exists():
                logger.debug(f"Image path remapped for encoding: {image_path} -> {remapped}")
                path = remapped
            else:
                raise FileNotFoundError(f"{image_path}")

        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists
        if not path.exists():
            remapped = _remap_missing_image_path(image_path)
            if remapped and remapped.exists():
                logger.debug(f"Image path remapped: {image_path} -> {remapped}")
                path = remapped
                image_path = str(remapped)
            else:
                logger.warning(f"Image file not found: {image_path}")
                return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


async def insert_text_content(
    coregraph,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    Insert pure text content into GraphCore

    Args:
        coregraph: GraphCore instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
    """
    logger.info("Starting text content insertion into GraphCore...")

    # Use GraphCore's insert method with all parameters
    await coregraph.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    coregraph,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into GraphCore

    Args:
        coregraph: GraphCore instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into GraphCore...")

    # Use GraphCore's insert method with all parameters
    try:
        await coregraph.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the docthinker branch of coregraph"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])


def create_bltcy_rerank_func(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    separator: str = " [SEP] ",
    timeout: int = 60,
):
    """
    Create a rerank function that calls an OpenAI-compatible rerank endpoint.

    Args:
        api_key: Optional API key. If omitted, falls back to environment variables
                 (LLM_BINDING_API_KEY, OPENAI_API_KEY).
        base_url: Optional base URL. Defaults to RERANK_HOST, EMBEDDING_BINDING_HOST,
                  LLM_BINDING_HOST, OPENAI_BASE_URL, or https://api.openai.com/v1.
        model_name: Rerank model name (default: BAAI/bge-reranker-v2-m3).
        separator: Separator between query and document when forming rerank inputs.
        timeout: Request timeout in seconds.

    Returns:
        Callable suitable for GraphCore's rerank_model_func or None when API key is missing.
    """
    resolved_api_key = api_key or os.getenv("RERANK_API_KEY")
    if not resolved_api_key:
        logger.warning(
            "Rerank is disabled because RERANK_API_KEY is not set."
        )
        return None

    default_base = "https://api.openai.com/v1"
    resolved_base = (
        base_url
        or os.getenv("RERANK_HOST")
        or os.getenv("EMBEDDING_BINDING_HOST")
        or os.getenv("LLM_BINDING_HOST")
        or os.getenv("OPENAI_BASE_URL")
        or default_base
    )
    resolved_model = (
        model_name
        if model_name is not None
        else os.getenv("RERANK_MODEL")
    )
    if not resolved_model:
        logger.info("Rerank is disabled because RERANK_MODEL is empty.")
        return None
    resolved_base_stripped = resolved_base.rstrip("/")
    is_dashscope_native = "services/rerank" in resolved_base_stripped.lower()
    is_dashscope = "dashscope" in resolved_base_stripped.lower()

    if is_dashscope_native:
        endpoint = resolved_base_stripped
    elif is_dashscope:
        endpoint = resolved_base_stripped + "/reranks"
    else:
        base_for_endpoint = resolved_base_stripped
        if base_for_endpoint.endswith("/chat/completions"):
            base_for_endpoint = base_for_endpoint[: -len("/chat/completions")]
        if "api.openai.com" in base_for_endpoint.lower():
            logger.info(
                "Rerank is disabled for OpenAI base URL; /rerank is not guaranteed on this endpoint."
            )
            return None
        endpoint = base_for_endpoint + "/rerank"

    session: aiohttp.ClientSession | None = None
    session_lock = asyncio.Lock()

    async def _get_session() -> aiohttp.ClientSession:
        nonlocal session
        async with session_lock:
            if session and not session.closed:
                return session
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(limit=80, limit_per_host=40, ttl_dns_cache=300)
            session = aiohttp.ClientSession(timeout=timeout_cfg, connector=connector)
            return session

    async def rerank(query: str, documents: List[str], **kwargs) -> List[Dict[str, float]]:
        if not documents:
            return []

        request_docs: List[str] = []
        for doc in documents:
            if isinstance(doc, str):
                request_docs.append(doc)
            elif isinstance(doc, dict) and "text" in doc:
                request_docs.append(str(doc["text"]))
            else:
                request_docs.append(str(doc))

        max_docs_per_request = 25
        batches: List[List[str]] = [
            request_docs[i : i + max_docs_per_request]
            for i in range(0, len(request_docs), max_docs_per_request)
        ]
        if len(batches) > 1:
            logger.warning(
                "Rerank request documents length %s exceeds API limit %s. "
                "Splitting into %s batched requests.",
                len(request_docs),
                max_docs_per_request,
                len(batches),
            )

        scores: List[float] = [0.0] * len(request_docs)
        headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

        for batch_index, batch_docs in enumerate(batches):
            if is_dashscope_native:
                payload = {
                    "model": resolved_model,
                    "input": {"query": query, "documents": batch_docs},
                    "parameters": {},
                }
                if "top_n" in kwargs and kwargs["top_n"] is not None:
                    payload["parameters"]["top_n"] = min(int(kwargs["top_n"]), len(batch_docs))
            else:
                payload = {
                    "model": resolved_model,
                    "query": query,
                    "documents": batch_docs,
                }
                if "top_n" in kwargs and kwargs["top_n"] is not None:
                    payload["top_n"] = min(int(kwargs["top_n"]), len(batch_docs))

            try:
                client = await _get_session()
                async with client.post(endpoint, headers=headers, data=json.dumps(payload)) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.error("Rerank request failed with status %s: %s", resp.status, body)
                        raise RuntimeError(f"Rerank HTTP {resp.status}")
                    response_json = await resp.json()
            except Exception as exc:
                logger.error("Rerank request failed: %s", exc)
                raise

            if is_dashscope_native:
                data_items = response_json.get("output", {}).get("results", [])
            else:
                data_items = response_json.get("data", [])
            if not data_items:
                data_items = response_json.get("results", [])
            if len(data_items) != len(batch_docs):
                logger.warning(
                    "Rerank response count mismatch: expected %s, got %s",
                    len(batch_docs),
                    len(data_items),
                )
                logger.debug("Rerank raw response: %s", response_json)

            for offset, item in enumerate(data_items):
                score = item.get("score")
                if score is None:
                    score = item.get("relevance_score")
                if score is None and isinstance(item.get("scores"), list):
                    score = item["scores"][0]
                if score is None:
                    raise ValueError(
                        "Rerank response missing 'score' or 'relevance_score' field."
                    )
                global_index = batch_index * max_docs_per_request + offset
                if global_index < len(scores):
                    scores[global_index] = float(score)

        ranked = [
            {"index": idx, "relevance_score": float(score)}
            for idx, score in enumerate(scores)
        ]
        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_n = kwargs.get("top_n")
        if top_n is not None:
            try:
                return ranked[: max(1, int(top_n))]
            except Exception:
                return ranked
        return ranked

    logger.info(
        "Rerank function configured with model '%s' at '%s'",
        resolved_model,
        endpoint,
    )
    return rerank
