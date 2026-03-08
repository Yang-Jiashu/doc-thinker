"""
Validation test: confirm that the fast_qa path for .txt files
now sends text directly to LLM instead of rendering to image.
"""

import asyncio
import inspect
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_code_structure():
    """Verify the code structure has the fix applied."""
    print("=" * 60)
    print("  VALIDATION: Code Structure Check")
    print("=" * 60)

    from docthinker.server.routers.query import _try_fast_qa

    source = inspect.getsource(_try_fast_qa)

    checks = {
        "text_direct branch exists": "text_direct" in source,
        "llm_model_func called for text": "llm_model_func" in source,
        "text_exts check before render": "if file_ext in text_exts" in source,
        "CJK font still available for non-text": "_render_text_to_image" in source,
        "errors=replace used": 'errors="replace"' in source,
    }

    all_pass = True
    for name, result in checks.items():
        status = "[OK]  " if result else "[FAIL]"
        if not result:
            all_pass = False
        print(f"  {status} {name}")

    return all_pass


async def simulate_fast_qa_text_file():
    """Simulate the fast_qa path with a mock to verify text goes to LLM directly."""
    print(f"\n{'='*60}")
    print("  VALIDATION: Simulate fast_qa for .txt file")
    print("=" * 60)

    from docthinker.server.routers.query import _try_fast_qa
    from docthinker.server.schemas import QueryRequest

    test_file = PROJECT_ROOT / "data" / "#00004" / "content" / "mayun.txt"
    if not test_file.exists():
        print("  [FAIL] Test file not found")
        return False

    llm_called = False
    vlm_called = False
    llm_prompt_received = ""

    async def mock_llm(prompt, **kw):
        nonlocal llm_called, llm_prompt_received
        llm_called = True
        llm_prompt_received = prompt
        return "这是一篇关于马云早年经历的文档。"

    async def mock_vlm(prompt, *, image_data=None, **kw):
        nonlocal vlm_called
        vlm_called = True
        return "garbled text"

    mock_rag = MagicMock()
    mock_rag.llm_model_func = mock_llm
    mock_rag.vision_model_func = mock_vlm
    mock_rag.config = MagicMock()
    mock_rag.config.parser_output_dir = str(PROJECT_ROOT / "data" / "_system")

    mock_session_mgr = MagicMock()
    mock_session_mgr.get_files.return_value = [
        {"file_path": str(test_file), "file_ext": ".txt", "file_size": test_file.stat().st_size}
    ]
    mock_session_mgr.get_session.return_value = {
        "path": str(test_file.parent.parent)
    }

    from docthinker.server import state as state_mod
    original_rag = state_mod.state.rag_instance
    original_sm = state_mod.state.session_manager
    try:
        state_mod.state.rag_instance = mock_rag
        state_mod.state.session_manager = mock_session_mgr

        req = QueryRequest(question="这个文档讲了什么", session_id="#00004")
        answer, meta = await _try_fast_qa(req)

        print(f"  answer: {answer}")
        print(f"  meta:   {meta}")
        print(f"  LLM called:  {llm_called}")
        print(f"  VLM called:  {vlm_called}")

        ok_answer = answer is not None and "马云" in answer
        ok_llm = llm_called and not vlm_called
        ok_mode = meta is not None and meta.get("mode") == "text_direct"
        ok_prompt = "马云" in llm_prompt_received or "mayun" in llm_prompt_received.lower()

        print(f"\n  [{'OK' if ok_answer else 'FAIL'}]   Answer contains expected content")
        print(f"  [{'OK' if ok_llm else 'FAIL'}]   LLM called, VLM NOT called (text went directly to LLM)")
        print(f"  [{'OK' if ok_mode else 'FAIL'}]   Mode is 'text_direct'")
        print(f"  [{'OK' if ok_prompt else 'FAIL'}]   LLM received the original text content")

        return ok_answer and ok_llm and ok_mode and ok_prompt
    finally:
        state_mod.state.rag_instance = original_rag
        state_mod.state.session_manager = original_sm


async def simulate_fast_qa_image_file():
    """Verify that image files still go through VLM path."""
    print(f"\n{'='*60}")
    print("  VALIDATION: Simulate fast_qa for image file (unchanged path)")
    print("=" * 60)

    from docthinker.server.routers.query import _try_fast_qa
    from docthinker.server.schemas import QueryRequest

    fake_img = PROJECT_ROOT / "data" / "#00004" / "knowledge" / "quick_qa" / "diag" / "render_cjk_font.png"
    if not fake_img.exists():
        print("  [SKIP] No test image available")
        return True

    vlm_called = False

    async def mock_vlm(prompt, *, image_data=None, **kw):
        nonlocal vlm_called
        vlm_called = True
        return "This is a document about Ma Yun."

    mock_rag = MagicMock()
    mock_rag.vision_model_func = mock_vlm
    mock_rag.config = MagicMock()

    mock_session_mgr = MagicMock()
    mock_session_mgr.get_files.return_value = [
        {"file_path": str(fake_img), "file_ext": ".png", "file_size": fake_img.stat().st_size}
    ]

    from docthinker.server import state as state_mod
    original_rag = state_mod.state.rag_instance
    original_sm = state_mod.state.session_manager
    try:
        state_mod.state.rag_instance = mock_rag
        state_mod.state.session_manager = mock_session_mgr

        req = QueryRequest(question="这个文档讲了什么", session_id="#00004")
        answer, meta = await _try_fast_qa(req)

        print(f"  VLM called: {vlm_called}")
        print(f"  answer:     {answer}")

        ok = vlm_called and answer is not None
        print(f"\n  [{'OK' if ok else 'FAIL'}]   Image files still route to VLM correctly")
        return ok
    finally:
        state_mod.state.rag_instance = original_rag
        state_mod.state.session_manager = original_sm


def main():
    print("=" * 60)
    print("  FIX VALIDATION TEST")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    code_ok = check_code_structure()
    text_ok = asyncio.run(simulate_fast_qa_text_file())
    image_ok = asyncio.run(simulate_fast_qa_image_file())

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Code structure:  {'PASS' if code_ok else 'FAIL'}")
    print(f"  Text file path:  {'PASS' if text_ok else 'FAIL'}")
    print(f"  Image file path: {'PASS' if image_ok else 'FAIL'}")

    all_ok = code_ok and text_ok and image_ok
    print(f"\n  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
