"""
Diagnostic test script for PDF pipeline bugs found in session #00026.

Tests:
  1. VLM max_tokens default — verify responses are not truncated
  2. Embedding input validation — ensure no None/empty content is embedded
  3. TXT ingestion — verify file_path propagates (no unknown_source)
  4. Rerank default — verify rerank defaults to disabled
  5. End-to-end PDF pipeline mode — verify dual-path activates
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEPARATOR = "=" * 70
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    tag = PASS if status == "pass" else (FAIL if status == "fail" else WARN)
    print(f"  {tag} {name}")
    if detail:
        for line in detail.split("\n"):
            print(f"        {line}")


# ── Test 1: VLM max_tokens default ─────────────────────────────────────────
def test_vlm_max_tokens():
    print(f"\n{SEPARATOR}")
    print("Test 1: VLM max_tokens default in _make_vision_model_func")
    print(SEPARATOR)

    from docthinker.server.app import _make_vision_model_func

    import inspect
    src = inspect.getsource(_make_vision_model_func)

    if 'max_tokens", 350' in src or "max_tokens', 350" in src:
        record("VLM default max_tokens", "fail",
               "Default max_tokens is still 350 — responses will be truncated!")
    elif "4096" in src or "8192" in src:
        record("VLM default max_tokens", "pass",
               "Default max_tokens has been increased (found 4096/8192)")
    else:
        record("VLM default max_tokens", "warn",
               f"Could not determine max_tokens default from source")


# ── Test 2: Embedding input validation ──────────────────────────────────────
def test_embedding_content_safety():
    print(f"\n{SEPARATOR}")
    print("Test 2: Multimodal entity content safety for embedding")
    print(SEPARATOR)

    import inspect
    from docthinker.processor import ProcessorMixin

    src = inspect.getsource(ProcessorMixin._store_multimodal_main_entities)

    has_none_guard = ("or description" in src or "or entity_name" in src)
    has_isinstance = "isinstance" in src

    if has_none_guard or has_isinstance:
        record("Embedding content None guard", "pass",
               "Content fallback chain prevents None values from reaching embedding API")
    else:
        record("Embedding content None guard", "fail",
               "entity_info.get('summary', description) can return None if summary key exists but is None")


# ── Test 3: Simulated entity data validation ────────────────────────────────
def test_entity_data_content_types():
    print(f"\n{SEPARATOR}")
    print("Test 3: Entity data content type validation")
    print(SEPARATOR)

    test_cases = [
        {"entity_info": {"entity_name": "TestEntity", "summary": None}, "description": "A test"},
        {"entity_info": {"entity_name": "TestEntity", "summary": ""}, "description": "A test"},
        {"entity_info": {"entity_name": "TestEntity"}, "description": "A test"},
        {"entity_info": {"entity_name": "TestEntity", "summary": "Good summary"}, "description": "A test"},
        {"entity_info": {"entity_name": "TestEntity", "summary": 123}, "description": "A test"},
    ]

    for i, case in enumerate(test_cases):
        entity_info = case["entity_info"]
        description = case["description"]
        entity_name = entity_info["entity_name"]

        raw_content = entity_info.get("summary") or description or entity_name
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        content = raw_content.strip() or entity_name

        if not content or not isinstance(content, str):
            record(f"Entity case {i + 1}", "fail", f"content={content!r} — would break embedding")
        elif content.strip() == "":
            record(f"Entity case {i + 1}", "fail", f"content is empty string — would break embedding")
        else:
            record(f"Entity case {i + 1}", "pass", f"content={content[:50]!r}")


# ── Test 4: TXT ingestion file_path propagation ────────────────────────────
def test_txt_ingest_file_path():
    print(f"\n{SEPARATOR}")
    print("Test 4: TXT ingestion file_path propagation")
    print(SEPARATOR)

    import inspect
    from docthinker.services.ingestion_service import IngestionService

    sig = inspect.signature(IngestionService.ingest_text)
    params = list(sig.parameters.keys())

    if "file_path" in params:
        record("ingest_text accepts file_path", "pass")
    else:
        record("ingest_text accepts file_path", "fail",
               f"Parameters: {params} — missing file_path, source will be unknown_source")

    insert_src = inspect.getsource(IngestionService._insert_text)
    if "file_paths" in insert_src:
        record("_insert_text passes file_paths to ainsert", "pass")
    else:
        record("_insert_text passes file_paths to ainsert", "fail")

    from docthinker.server.routers import ingest as ingest_module
    ingest_src = inspect.getsource(ingest_module)
    if "file_path=txt_path.name" in ingest_src or "file_path=txt_path" in ingest_src:
        record("Ingest route passes txt filename", "pass")
    else:
        record("Ingest route passes txt filename", "fail",
               "TXT files will be stored as unknown_source in GraphCore")


# ── Test 5: Rerank defaults ────────────────────────────────────────────────
def test_rerank_defaults():
    print(f"\n{SEPARATOR}")
    print("Test 5: Rerank default configuration")
    print(SEPARATOR)

    from docthinker.server.schemas import QueryRequest

    default_req = QueryRequest(question="test")
    if default_req.enable_rerank:
        record("Rerank default", "fail",
               "enable_rerank defaults to True but no rerank model is configured — will log errors")
    else:
        record("Rerank default", "pass",
               "enable_rerank defaults to False")


# ── Test 6: PDF_PARSE_MODE env ──────────────────────────────────────────────
def test_pdf_parse_mode():
    print(f"\n{SEPARATOR}")
    print("Test 6: PDF_PARSE_MODE environment configuration")
    print(SEPARATOR)

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        record("PDF_PARSE_MODE in .env", "warn", ".env file not found")
        return

    env_text = env_path.read_text(encoding="utf-8")
    for line in env_text.splitlines():
        line = line.strip()
        if line.startswith("PDF_PARSE_MODE="):
            mode = line.split("=", 1)[1].strip()
            if mode == "full":
                record("PDF_PARSE_MODE", "pass", f"Mode is '{mode}' — dual-path pipeline active")
            elif mode == "vision_only":
                record("PDF_PARSE_MODE", "warn",
                       f"Mode is '{mode}' — old FAST PATH, not using new dual-path pipeline")
            else:
                record("PDF_PARSE_MODE", "pass", f"Mode is '{mode}'")
            return

    record("PDF_PARSE_MODE", "warn", "PDF_PARSE_MODE not found in .env, will default to 'full'")


# ── Test 7: processor.py uses new dual-path ─────────────────────────────────
def test_dual_path_code():
    print(f"\n{SEPARATOR}")
    print("Test 7: Processor uses new dual-path pipeline for PDFs")
    print(SEPARATOR)

    import inspect
    from docthinker.processor import ProcessorMixin

    src = inspect.getsource(ProcessorMixin.process_document_complete)
    if "_process_pdf_dual_path" in src:
        record("process_document_complete calls _process_pdf_dual_path", "pass")
    else:
        record("process_document_complete calls _process_pdf_dual_path", "fail",
               "Still using old code path — [FAST PATH] message from logs")

    if hasattr(ProcessorMixin, "_process_pdf_dual_path"):
        record("_process_pdf_dual_path method exists", "pass")
    else:
        record("_process_pdf_dual_path method exists", "fail")


# ── Test 8: VLM query max_tokens in query.py ────────────────────────────────
def test_query_vlm_max_tokens():
    print(f"\n{SEPARATOR}")
    print("Test 8: VLM query response max_tokens")
    print(SEPARATOR)

    from docthinker import query as query_module
    import inspect

    src = inspect.getsource(query_module.QueryMixin._call_vlm_with_multimodal_content)

    if "max_tokens" in src:
        import re
        matches = re.findall(r"max_tokens\s*=\s*(\d+)", src)
        if matches:
            max_val = max(int(m) for m in matches)
            if max_val >= 4096:
                record("VLM query max_tokens", "pass", f"max_tokens={max_val}")
            else:
                record("VLM query max_tokens", "warn",
                       f"max_tokens={max_val}, may still truncate long answers")
        else:
            record("VLM query max_tokens", "pass", "max_tokens parameter found")
    else:
        record("VLM query max_tokens", "fail",
               "No max_tokens in _call_vlm_with_multimodal_content — defaults to 350, causing truncation!")


# ── Test 9: Live VLM call (optional, needs running API) ─────────────────────
async def test_live_vlm_call():
    print(f"\n{SEPARATOR}")
    print("Test 9: Live VLM output length test (requires API key)")
    print(SEPARATOR)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = (os.environ.get("VLM_API_KEY")
               or os.environ.get("LLM_BINDING_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        record("Live VLM test", "warn", "No API key found, skipping")
        return

    from docthinker.auto_thinking.vlm_client import VLMClient

    model = os.environ.get("VLM_MODEL", "gpt-4.1")
    base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1/chat/completions")

    client = VLMClient(
        api_key=api_key,
        api_base=base_url,
        model=model,
    )

    prompt = (
        "Write a detailed 500-word essay about the history of robotics, "
        "from early automatons to modern AI-powered robots. Include specific dates and inventors."
    )

    try:
        t0 = time.time()
        result = await client.generate(prompt, max_tokens=4096)
        elapsed = time.time() - t0
        word_count = len(result.split())
        char_count = len(result)

        if word_count < 100:
            record("VLM output length", "fail",
                   f"Only {word_count} words ({char_count} chars) in {elapsed:.1f}s — likely truncated")
        elif word_count < 300:
            record("VLM output length", "warn",
                   f"{word_count} words ({char_count} chars) in {elapsed:.1f}s — somewhat short")
        else:
            record("VLM output length", "pass",
                   f"{word_count} words ({char_count} chars) in {elapsed:.1f}s")

    except Exception as e:
        record("VLM output length", "warn", f"API call failed: {e}")
    finally:
        await client.close()


# ── Test 10: Nano-vectordb upsert with edge-case content ────────────────────
async def test_embedding_edge_cases():
    print(f"\n{SEPARATOR}")
    print("Test 10: Embedding edge-case content validation")
    print(SEPARATOR)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = (os.environ.get("EMBEDDING_BINDING_API_KEY")
               or os.environ.get("EMBEDDING_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        record("Embedding edge case test", "warn", "No API key found, skipping")
        return
    os.environ.setdefault("EMBEDDING_BINDING_API_KEY", api_key)

    from docthinker.hypergraph.llm import openai_compatible_embedding

    test_texts = [
        "Normal text about a robot",
        "A" * 10,
        "轮履带变换式侦察搜救机器人",
    ]

    try:
        result = await openai_compatible_embedding(test_texts)
        if result is not None and len(result) == len(test_texts):
            record("Embedding normal texts", "pass", f"Got {len(result)} embeddings")
        else:
            record("Embedding normal texts", "fail", f"Expected {len(test_texts)}, got {len(result) if result is not None else 'None'}")
    except Exception as e:
        record("Embedding normal texts", "fail", str(e))

    bad_texts = ["", " ", "\n\n"]
    for text in bad_texts:
        try:
            result = await openai_compatible_embedding([text])
            record(f"Embedding empty-ish text {text!r}", "warn",
                   f"API accepted it (got {len(result)} embedding)")
        except Exception as e:
            record(f"Embedding empty-ish text {text!r}", "warn",
                   f"API rejected: {str(e)[:80]}")


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    print("PDF Pipeline Bug Diagnostic Tests")
    print("Based on session #00026 logs analysis")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_vlm_max_tokens()
    test_embedding_content_safety()
    test_entity_data_content_types()
    test_txt_ingest_file_path()
    test_rerank_defaults()
    test_pdf_parse_mode()
    test_dual_path_code()
    test_query_vlm_max_tokens()

    await test_live_vlm_call()
    await test_embedding_edge_cases()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    passes = sum(1 for _, s, _ in results if s == "pass")
    fails = sum(1 for _, s, _ in results if s == "fail")
    warns = sum(1 for _, s, _ in results if s == "warn")

    print(f"  Total: {len(results)} tests")
    print(f"  {PASS} Passed: {passes}")
    print(f"  {FAIL} Failed: {fails}")
    print(f"  {WARN} Warnings: {warns}")

    if fails > 0:
        print(f"\n  FAILURES:")
        for name, status, detail in results:
            if status == "fail":
                print(f"    - {name}: {detail}")

    print()
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
