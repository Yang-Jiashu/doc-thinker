"""
PDF Processing Pipeline Diagnostic Script
==========================================
Tests every stage of the PDF → Knowledge Graph pipeline:
  1. MinerU PDF parsing (text extraction quality)
  2. GPT Vision fallback (when MinerU fails)
  3. Content separation (text vs multimodal)
  4. Entity extraction LLM connectivity
  5. Embedding function health
  6. Knowledge base population check
  7. Rerank status check

Usage:
    python scripts/test_pdf_pipeline.py [--pdf PATH_TO_PDF] [--session SESSION_ID]
"""

import asyncio
import base64
import json
import os
import sys
import time
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def ok(msg: str):
    print(f"  [OK]   {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")


def info(msg: str):
    print(f"  [INFO] {msg}")


def warn(msg: str):
    print(f"  [WARN] {msg}")


# ── Stage 1: MinerU PDF Parsing ──────────────────────────────────────────
def test_mineru_parsing(pdf_path: Path):
    banner("Stage 1: MinerU PDF Parsing")

    if not pdf_path.exists():
        fail(f"PDF not found: {pdf_path}")
        return [], 0

    info(f"PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")

    try:
        from docthinker.parser import MineruParser
        parser = MineruParser()
        install_ok = parser.check_installation()
        if install_ok:
            ok("MinerU 2.0 installed")
        else:
            fail("MinerU 2.0 NOT installed")
            return [], 0
    except Exception as e:
        fail(f"MinerU import error: {e}")
        return [], 0

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        try:
            content_list = parser.parse_pdf(
                pdf_path=pdf_path,
                output_dir=tmpdir,
                method="auto",
            )
            elapsed = time.time() - t0
            ok(f"Parsing completed in {elapsed:.1f}s — {len(content_list)} content blocks")
        except Exception as e:
            elapsed = time.time() - t0
            fail(f"Parsing failed after {elapsed:.1f}s: {e}")
            return [], 0

    # Analyze content blocks
    type_counts = {}
    text_chars = 0
    for block in content_list:
        bt = block.get("type", "unknown")
        type_counts[bt] = type_counts.get(bt, 0) + 1
        if bt == "text":
            text_chars += len(block.get("text", ""))

    info(f"Block type distribution: {type_counts}")
    info(f"Total text characters: {text_chars}")

    text_score = sum(
        1 for b in content_list
        if b.get("type") == "text" and (b.get("text") or "").strip()
    )
    if text_score == 0:
        warn("MinerU extracted 0 text blocks — GPT Vision fallback would activate")
    else:
        ok(f"{text_score} text blocks with content")

    # Print first text block preview
    for block in content_list:
        if block.get("type") == "text" and block.get("text", "").strip():
            preview = block["text"][:200].replace("\n", " ")
            info(f"First text preview: {preview}...")
            break

    return content_list, text_chars


# ── Stage 2: GPT Vision PDF Fallback ────────────────────────────────────
async def test_gpt_vision_fallback(pdf_path: Path):
    banner("Stage 2: GPT Vision PDF Fallback (1 page test)")

    try:
        import fitz
        ok("PyMuPDF (fitz) available")
    except ImportError:
        fail("PyMuPDF not installed — run: pip install pymupdf")
        return

    try:
        doc = fitz.open(str(pdf_path))
        ok(f"PDF opened: {len(doc)} pages")
    except Exception as e:
        fail(f"Cannot open PDF: {e}")
        return

    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    info(f"Page 1 image: {len(img_bytes)} bytes, {pix.width}x{pix.height}")

    # Write temp image
    tmp = Path(tempfile.mktemp(suffix=".png"))
    tmp.write_bytes(img_bytes)

    # Test GPT Vision
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    vlm_host = os.getenv("LLM_VLM_HOST") or os.getenv("LLM_BINDING_HOST") or "https://api.openai.com/v1"
    vlm_model = os.getenv("VLM_MODEL") or "gpt-4.1"

    if not api_key:
        fail("No API key found in env (LLM_BINDING_API_KEY / OPENAI_API_KEY)")
        doc.close()
        tmp.unlink(missing_ok=True)
        return

    info(f"VLM endpoint: {vlm_host}")
    info(f"VLM model: {vlm_model}")

    try:
        from docthinker.auto_thinking.vlm_client import VLMClient
        client = VLMClient(api_key=api_key, api_base=vlm_host, model=vlm_model)

        prompt = (
            "这是一个PDF文档的第1页截图。"
            "请将图片中所有的文字内容完整、准确地提取出来。"
            "保持原文的段落和层次结构。只输出提取到的文字内容，不要添加解释。"
        )

        t0 = time.time()
        answer = await client.generate(prompt, images=[str(tmp)])
        elapsed = time.time() - t0

        if answer and answer.strip():
            ok(f"GPT Vision extracted {len(answer)} chars in {elapsed:.1f}s")
            preview = answer[:300].replace("\n", " ")
            info(f"Content preview: {preview}...")
        else:
            fail("GPT Vision returned empty response")

        await client.close()
    except Exception as e:
        fail(f"GPT Vision call failed: {e}")
    finally:
        doc.close()
        tmp.unlink(missing_ok=True)


# ── Stage 3: Entity Extraction LLM Connectivity ─────────────────────────
async def test_llm_connectivity():
    banner("Stage 3: Entity Extraction LLM Connectivity")

    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST") or "https://api.openai.com/v1"
    extraction_model = os.getenv("EXTRACTION_LLM_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    llm_model = os.getenv("LLM_MODEL") or "gpt-4o-mini"

    info(f"LLM host: {base_url}")
    info(f"LLM model: {llm_model}")
    info(f"Extraction model: {extraction_model}")

    if not api_key:
        fail("No API key")
        return

    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        t0 = time.time()
        resp = await client.chat.completions.create(
            model=extraction_model,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=10,
        )
        elapsed = time.time() - t0
        answer = resp.choices[0].message.content
        ok(f"LLM responded in {elapsed:.1f}s: {answer}")
        await client.close()
    except Exception as e:
        fail(f"LLM call failed: {e}")


# ── Stage 4: Embedding Function Health ───────────────────────────────────
async def test_embedding():
    banner("Stage 4: Embedding Function Health")

    api_key = os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("EMBEDDING_BINDING_HOST") or "https://api.openai.com/v1"
    model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-large"
    dim = int(os.getenv("EMBEDDING_DIM") or 3072)

    info(f"Embedding host: {base_url}")
    info(f"Embedding model: {model} (dim={dim})")

    if not api_key:
        fail("No API key")
        return

    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        t0 = time.time()
        resp = await client.embeddings.create(
            model=model,
            input=["计算机系统概述"],
        )
        elapsed = time.time() - t0
        vec = resp.data[0].embedding
        ok(f"Embedding returned {len(vec)}-dim vector in {elapsed:.1f}s")
        if len(vec) != dim:
            warn(f"Expected dim={dim}, got {len(vec)}")
        await client.close()
    except Exception as e:
        fail(f"Embedding call failed: {e}")


# ── Stage 5: Knowledge Base Population Check ────────────────────────────
def test_knowledge_base(session_id: str):
    banner(f"Stage 5: Knowledge Base Check (session {session_id})")

    kb_dir = PROJECT_ROOT / "data" / session_id / "knowledge"
    if not kb_dir.exists():
        warn(f"Knowledge directory not found: {kb_dir}")
        return

    files_to_check = [
        "knowledge_graph.json",
        "kv_store_text_chunks.json",
        "kv_store_full_entities.json",
        "vdb_entities.json",
        "vdb_chunks.json",
    ]

    for fname in files_to_check:
        fpath = kb_dir / fname
        if not fpath.exists():
            warn(f"  {fname}: not found")
            continue

        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                count = len(data)
            elif isinstance(data, list):
                count = len(data)
            else:
                count = "?"
            size_kb = fpath.stat().st_size / 1024
            info(f"  {fname}: {count} entries, {size_kb:.1f} KB")
        except Exception as e:
            warn(f"  {fname}: read error — {e}")


# ── Stage 6: Rerank Status ──────────────────────────────────────────────
def test_rerank_config():
    banner("Stage 6: Rerank Configuration")

    rerank_model = os.getenv("RERANK_MODEL") or ""
    rerank_key = os.getenv("RERANK_API_KEY") or ""

    if not rerank_model and not rerank_key:
        ok("Rerank is disabled (no model/key configured) — no 404 errors expected")
    else:
        warn(f"Rerank model: '{rerank_model}' — may cause 404 if endpoint doesn't support it")


# ── Stage 7: Concurrency Configuration ──────────────────────────────────
def test_concurrency_config():
    banner("Stage 7: Concurrency Configuration")

    max_async = os.getenv("MAX_ASYNC") or "4"
    embed_async = os.getenv("EMBEDDING_FUNC_MAX_ASYNC") or "16"
    max_parallel = os.getenv("MAX_PARALLEL_INSERT") or "2"

    info(f"MAX_ASYNC (LLM concurrency): {max_async}")
    info(f"EMBEDDING_FUNC_MAX_ASYNC: {embed_async}")
    info(f"MAX_PARALLEL_INSERT: {max_parallel}")

    if int(max_async) < 8:
        warn("MAX_ASYNC < 8 is quite low for OpenAI; recommend 16+")
    else:
        ok(f"MAX_ASYNC={max_async} is reasonable")

    if int(embed_async) < 8:
        warn("EMBEDDING_FUNC_MAX_ASYNC < 8; recommend 16+")
    else:
        ok(f"EMBEDDING_FUNC_MAX_ASYNC={embed_async} is good")


# ── Stage 8: MinerU Output Inspection ───────────────────────────────────
def test_mineru_output(session_id: str):
    banner(f"Stage 8: MinerU Output Inspection (session {session_id})")

    output_dir = PROJECT_ROOT / "data" / session_id / "knowledge" / "output"
    if not output_dir.exists():
        warn(f"Output directory not found: {output_dir}")
        return

    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info(f"Run directory: {run_dir.name}")

        # Find content_list.json
        for cl_file in run_dir.rglob("*content_list.json"):
            try:
                data = json.loads(cl_file.read_text(encoding="utf-8"))
                types = {}
                text_chars = 0
                for item in data:
                    t = item.get("type", "unknown")
                    types[t] = types.get(t, 0) + 1
                    if t == "text":
                        text_chars += len(item.get("text", ""))
                info(f"  {cl_file.name}: {len(data)} blocks, types={types}, text_chars={text_chars}")
                if text_chars == 0:
                    warn("  No text extracted in this run!")
            except Exception as e:
                warn(f"  Cannot read {cl_file.name}: {e}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF Pipeline Diagnostic")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file")
    parser.add_argument("--session", type=str, default="#00020", help="Session ID")
    args = parser.parse_args()

    if args.pdf:
        pdf_path = Path(args.pdf)
    else:
        # Find first PDF in session content dir
        content_dir = PROJECT_ROOT / "data" / args.session / "content"
        pdfs = list(content_dir.glob("*.pdf")) if content_dir.exists() else []
        if pdfs:
            pdf_path = pdfs[0]
        else:
            pdf_path = Path("nonexistent.pdf")

    print(f"\nPDF Pipeline Diagnostic — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PDF: {pdf_path}")
    print(f"Session: {args.session}")

    # Run all stages
    content_list, text_chars = test_mineru_parsing(pdf_path)
    await test_gpt_vision_fallback(pdf_path)
    await test_llm_connectivity()
    await test_embedding()
    test_knowledge_base(args.session)
    test_rerank_config()
    test_concurrency_config()
    test_mineru_output(args.session)

    banner("Summary")
    if text_chars == 0:
        warn("MinerU extracted 0 text → GPT Vision fallback is CRITICAL for this PDF")
        info("After restarting server, the new fallback will automatically extract text via GPT-4 Vision")
    else:
        ok(f"MinerU extracted {text_chars} chars of text")

    info("Next steps:")
    info("  1. Restart the server to pick up new .env settings (GPT models, concurrency)")
    info("  2. Re-upload the PDF in a new session to test the full pipeline with fixes")
    info("  3. Check that entity extraction uses gpt-4.1 (not qwen3-8b)")


if __name__ == "__main__":
    asyncio.run(main())
