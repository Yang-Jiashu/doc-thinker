"""
Test script for the dual-path PDF pipeline (MinerU + GPT Vision).

Tests each module independently and then the full orchestrated pipeline.
Uses the IEEE paper PDF as the test document.

Usage:
    python scripts/test_pdf_dual_pipeline.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

TEST_PDF = PROJECT_ROOT / "data" / "#00024" / "code" / "New_IEEEtran_how-to.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "#00024" / "knowledge"

DIVIDER = "=" * 72


def _preview(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... ({len(text) - max_chars} more chars)"


# ──────────────────────────────────────────────────────────────────────
# Test 1: MinerU Extractor only
# ──────────────────────────────────────────────────────────────────────
async def test_mineru_extractor():
    print(f"\n{DIVIDER}")
    print("TEST 1: MineruExtractor (layout-aware text + multimodal extraction)")
    print(DIVIDER)

    from docthinker.pdf_pipeline.mineru_extractor import MineruExtractor

    ext = MineruExtractor(
        parse_method="auto",
        output_dir=str(OUTPUT_DIR / "output"),
        skip_ocr_retry=True,
    )

    t0 = time.time()
    result = await ext.extract(str(TEST_PDF))
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Text blocks: {len(result.text_blocks)}")
    print(f"  Multimodal blocks: {len(result.multimodal_blocks)}")
    print(f"  Total text chars: {sum(len(b.get('text', '')) for b in result.text_blocks)}")

    if result.text_blocks:
        print(f"\n  [First text block preview]")
        print(f"  {_preview(result.text_blocks[0].get('text', ''))}")

    mm_types = {}
    for b in result.multimodal_blocks:
        t = b.get("type", "?")
        mm_types[t] = mm_types.get(t, 0) + 1
    if mm_types:
        print(f"\n  Multimodal types: {mm_types}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Test 2: GPT Vision Extractor only
# ──────────────────────────────────────────────────────────────────────
async def test_vision_extractor():
    print(f"\n{DIVIDER}")
    print("TEST 2: VisionExtractor (GPT Vision page-by-page text extraction)")
    print(DIVIDER)

    vision_func = await _get_vision_func()
    if not vision_func:
        print("  SKIP: No vision_model_func available")
        return None

    from docthinker.pdf_pipeline.vision_extractor import VisionExtractor

    ext = VisionExtractor(
        vision_model_func=vision_func,
        concurrency=4,
        dpi=200,
    )

    vision_dir = str(OUTPUT_DIR / "vision_pages_test")

    t0 = time.time()
    result = await ext.extract(str(TEST_PDF), image_output_dir=vision_dir)
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Text blocks: {len(result.text_blocks)}")
    print(f"  Page images saved: {len(result.page_images)}")
    print(f"  Total text chars: {sum(len(b.get('text', '')) for b in result.text_blocks)}")

    if result.text_blocks:
        print(f"\n  [First page text preview]")
        print(f"  {_preview(result.text_blocks[0].get('text', ''))}")

    if result.page_images:
        print(f"\n  Page images: {result.page_images[:3]}{'...' if len(result.page_images) > 3 else ''}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Test 3: Full orchestrated pipeline
# ──────────────────────────────────────────────────────────────────────
async def test_orchestrator_full():
    print(f"\n{DIVIDER}")
    print("TEST 3: PDFPipelineOrchestrator mode='full' (MinerU + Vision parallel)")
    print(DIVIDER)

    vision_func = await _get_vision_func()
    from docthinker.pdf_pipeline import PDFPipelineOrchestrator
    from docthinker.pdf_pipeline.mineru_extractor import MineruExtractor
    from docthinker.pdf_pipeline.vision_extractor import VisionExtractor

    orch = PDFPipelineOrchestrator(
        mode="full",
        mineru_extractor=MineruExtractor(
            parse_method="auto",
            output_dir=str(OUTPUT_DIR / "output"),
            skip_ocr_retry=True,
        ),
        vision_extractor=VisionExtractor(vision_model_func=vision_func),
    )

    t0 = time.time()
    result = await orch.process(
        pdf_path=str(TEST_PDF),
        mineru_output_dir=str(OUTPUT_DIR / "output"),
        vision_image_dir=str(OUTPUT_DIR / "vision_pages"),
    )
    elapsed = time.time() - t0

    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"  MinerU elapsed: {result.mineru_result.elapsed_seconds:.1f}s")
    print(f"  Vision elapsed: {result.vision_result.elapsed_seconds:.1f}s")
    print(f"  Merged text blocks: {len(result.text_blocks)}")
    print(f"  Merged multimodal blocks: {len(result.multimodal_blocks)}")
    print(f"  Total text chars: {len(result.text_content)}")
    print(f"  Page images: {len(result.page_images)}")

    print(f"\n  --- MinerU contribution ---")
    print(f"  Text blocks: {len(result.mineru_result.text_blocks)}")
    print(f"  Multimodal blocks: {len(result.mineru_result.multimodal_blocks)}")

    print(f"\n  --- Vision contribution ---")
    print(f"  Text blocks: {len(result.vision_result.text_blocks)}")
    print(f"  Page images: {len(result.vision_result.page_images)}")

    if result.text_content:
        print(f"\n  [Merged text preview (first 800 chars)]")
        print(f"  {_preview(result.text_content, 800)}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Test 4: Fast mode (MinerU only)
# ──────────────────────────────────────────────────────────────────────
async def test_orchestrator_fast():
    print(f"\n{DIVIDER}")
    print("TEST 4: PDFPipelineOrchestrator mode='fast' (MinerU only)")
    print(DIVIDER)

    from docthinker.pdf_pipeline import PDFPipelineOrchestrator
    from docthinker.pdf_pipeline.mineru_extractor import MineruExtractor

    orch = PDFPipelineOrchestrator(
        mode="fast",
        mineru_extractor=MineruExtractor(
            parse_method="auto",
            output_dir=str(OUTPUT_DIR / "output"),
            skip_ocr_retry=True,
        ),
    )

    t0 = time.time()
    result = await orch.process(
        pdf_path=str(TEST_PDF),
        mineru_output_dir=str(OUTPUT_DIR / "output"),
    )
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Text blocks: {len(result.text_blocks)}")
    print(f"  Multimodal blocks: {len(result.multimodal_blocks)}")
    print(f"  Text chars: {len(result.text_content)}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Test 5: Vision-only mode
# ──────────────────────────────────────────────────────────────────────
async def test_orchestrator_vision_only():
    print(f"\n{DIVIDER}")
    print("TEST 5: PDFPipelineOrchestrator mode='vision_only' (GPT Vision only)")
    print(DIVIDER)

    vision_func = await _get_vision_func()
    if not vision_func:
        print("  SKIP: No vision_model_func")
        return None

    from docthinker.pdf_pipeline import PDFPipelineOrchestrator
    from docthinker.pdf_pipeline.vision_extractor import VisionExtractor

    orch = PDFPipelineOrchestrator(
        mode="vision_only",
        vision_extractor=VisionExtractor(vision_model_func=vision_func),
    )

    t0 = time.time()
    result = await orch.process(
        pdf_path=str(TEST_PDF),
        vision_image_dir=str(OUTPUT_DIR / "vision_pages_test_only"),
    )
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Text blocks: {len(result.text_blocks)}")
    print(f"  Page images: {len(result.page_images)}")
    print(f"  Text chars: {len(result.text_content)}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Test 6: Text dedup quality
# ──────────────────────────────────────────────────────────────────────
async def test_dedup():
    print(f"\n{DIVIDER}")
    print("TEST 6: Text deduplication logic")
    print(DIVIDER)

    from docthinker.pdf_pipeline.orchestrator import _dedup_text_blocks

    primary = [
        {"text": "This is a test paragraph about computer architecture and system design."},
        {"text": "The IEEE standard defines formatting rules for technical papers."},
    ]
    secondary = [
        {"text": "This is a test paragraph about computer architecture and system design."},
        {"text": "A completely new paragraph about networking protocols."},
        {"text": "The IEEE standard defines formatting rules for technical papers and publications."},
    ]

    unique = _dedup_text_blocks(primary, secondary, similarity_threshold=0.8)
    print(f"  Primary blocks: {len(primary)}")
    print(f"  Secondary blocks: {len(secondary)}")
    print(f"  Unique after dedup: {len(unique)}")
    for u in unique:
        print(f"    - {u['text'][:80]}...")

    assert len(unique) <= len(secondary), "Dedup should not add blocks"
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────
# Helper: get vision model func from settings
# ──────────────────────────────────────────────────────────────────────
async def _get_vision_func():
    try:
        from docthinker.auto_thinking.vlm_client import VLMClient
        vlm = VLMClient()
        if not vlm.host or not vlm.api_key:
            print("  [Vision] No VLM host/key configured")
            return None

        async def vision_func(prompt: str, image_data: str = None, **kw):
            return await vlm.ask(prompt, image_data=image_data)

        test = await vision_func("Say 'hello' in one word.")
        if test:
            print(f"  [Vision] Model OK — test response: {test[:50]}")
            return vision_func
        return None
    except Exception as e:
        print(f"  [Vision] Init failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{'#' * 72}")
    print(f"# PDF Dual-Path Pipeline Test Suite")
    print(f"# Test PDF: {TEST_PDF}")
    print(f"# PDF exists: {TEST_PDF.exists()}")
    if TEST_PDF.exists():
        print(f"# PDF size: {TEST_PDF.stat().st_size / 1024:.1f} KB")
    print(f"# PDF_PARSE_MODE: {os.environ.get('PDF_PARSE_MODE', 'not set')}")
    print(f"{'#' * 72}")

    if not TEST_PDF.exists():
        print("\nERROR: Test PDF not found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    await test_dedup()

    print(f"\n{'>' * 72}")
    print(f"> Running MinerU extractor test...")
    print(f"{'>' * 72}")
    mineru_result = await test_mineru_extractor()

    print(f"\n{'>' * 72}")
    print(f"> Running Vision extractor test (1 page only for speed)...")
    print(f"{'>' * 72}")
    vision_result = await test_vision_extractor()

    print(f"\n{'>' * 72}")
    print(f"> Running full orchestrator test...")
    print(f"{'>' * 72}")
    full_result = await test_orchestrator_full()

    # Summary
    print(f"\n{'#' * 72}")
    print("# SUMMARY")
    print(f"{'#' * 72}")
    if mineru_result:
        print(f"  MinerU: {len(mineru_result.text_blocks)} text, "
              f"{len(mineru_result.multimodal_blocks)} multimodal, "
              f"{mineru_result.elapsed_seconds:.1f}s")
    if vision_result:
        print(f"  Vision: {len(vision_result.text_blocks)} text, "
              f"{len(vision_result.page_images)} images, "
              f"{vision_result.elapsed_seconds:.1f}s")
    if full_result:
        print(f"  Full:   {len(full_result.text_blocks)} text (merged), "
              f"{len(full_result.multimodal_blocks)} multimodal (merged), "
              f"{full_result.elapsed_seconds:.1f}s")
        savings = 0
        if mineru_result and vision_result:
            sequential = mineru_result.elapsed_seconds + vision_result.elapsed_seconds
            savings = sequential - full_result.elapsed_seconds
            print(f"  Parallel savings: ~{savings:.1f}s vs sequential")

    print(f"\n{'#' * 72}")
    print("# Tests complete!")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    logging_setup = True
    if logging_setup:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    asyncio.run(main())
