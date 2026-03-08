"""
Comprehensive diagnostic script for tracing garbled text (乱码) in the fast_qa pipeline.

Tests each stage independently:
  Stage 1: Source file reading & encoding (byte-level analysis)
  Stage 2: Font loading & CJK rendering quality
  Stage 3: Image rendering (compare default vs CJK font)
  Stage 4: VLM API call with rendered images
  Stage 5: Full pipeline simulation (end-to-end)
"""

import asyncio
import base64
import json
import os
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = []


def report(stage: str, test: str, passed: bool, detail: str = ""):
    status = "[OK]  " if passed else "[FAIL]"
    msg = f"  {status} {stage} / {test}"
    if detail:
        for i, line in enumerate(detail.split("\n")):
            if i == 0:
                msg += f"\n         -> {line}"
            else:
                msg += f"\n            {line}"
    print(msg)
    RESULTS.append({"stage": stage, "test": test, "passed": passed, "detail": detail})


def stage_header(name: str):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


# ─── Font search (self-contained, no server import) ─────────────────
CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def find_cjk_font(size: int = 18):
    from PIL import ImageFont
    for fp in CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────
# STAGE 1: Source file reading & encoding (byte-level)
# ─────────────────────────────────────────────────────────────────────
def test_stage1_source_file():
    stage_header("STAGE 1: Source File Reading & Encoding")

    test_file = PROJECT_ROOT / "data" / "#00004" / "content" / "mayun.txt"

    report("Stage1", "file_exists", test_file.exists(), str(test_file))
    if not test_file.exists():
        return None

    raw = test_file.read_bytes()
    report("Stage1", "file_size_bytes", len(raw) > 0, f"{len(raw)} bytes")

    # BOM check
    has_utf8_bom = raw[:3] == b'\xef\xbb\xbf'
    has_utf16_le = raw[:2] == b'\xff\xfe'
    has_utf16_be = raw[:2] == b'\xfe\xff'
    report("Stage1", "bom_check",
           not has_utf16_le and not has_utf16_be,
           f"UTF-8 BOM={has_utf8_bom}, UTF-16 LE={has_utf16_le}, UTF-16 BE={has_utf16_be}")

    # Strict UTF-8 decode
    try:
        text_strict = raw.decode("utf-8")
        report("Stage1", "utf8_strict_decode", True, f"{len(text_strict)} chars")
    except UnicodeDecodeError as e:
        report("Stage1", "utf8_strict_decode", False,
               f"FAILS at byte {e.start}: {e.reason}\n"
               f"Context bytes [{max(0,e.start-5)}:{e.end+5}]: {raw[max(0,e.start-5):e.end+5].hex(' ')}")
        text_strict = None

    # read_text with errors="ignore" (what the code actually does)
    text_ignore = test_file.read_text(encoding="utf-8", errors="ignore")
    report("Stage1", "utf8_ignore_decode", True, f"{len(text_ignore)} chars")

    if text_strict is not None:
        diff = len(text_strict) - len(text_ignore)
        report("Stage1", "chars_dropped_by_ignore", diff == 0,
               f"strict={len(text_strict)} vs ignore={len(text_ignore)}, dropped={diff}")

    # CJK char count
    cjk_count = sum(1 for c in text_ignore if '\u4e00' <= c <= '\u9fff')
    report("Stage1", "cjk_char_count", cjk_count > 50, f"{cjk_count} CJK characters")

    # Print hex dump of first 100 bytes for manual inspection
    hex_preview = raw[:100].hex(' ')
    report("Stage1", "first_100_bytes_hex", True, hex_preview)

    # Print first 100 chars as unicode escapes (safe for any console encoding)
    safe_preview = text_ignore[:100].encode("unicode_escape").decode("ascii")
    report("Stage1", "first_100_chars_escaped", True, safe_preview)

    return text_ignore


# ─────────────────────────────────────────────────────────────────────
# STAGE 2: Font loading & CJK rendering quality
# ─────────────────────────────────────────────────────────────────────
def test_stage2_font():
    stage_header("STAGE 2: Font Loading & CJK Rendering Quality")

    try:
        from PIL import Image, ImageDraw, ImageFont
        report("Stage2", "pillow_import", True, "PIL imported")
    except ImportError as e:
        report("Stage2", "pillow_import", False, str(e))
        return None, False

    diag_dir = PROJECT_ROOT / "data" / "#00004" / "knowledge" / "quick_qa" / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    test_text_short = "马云祖籍浙江"
    test_text_long = "马云祖籍浙江省嵊州市（原嵊县）谷来镇，后父母移居杭州。"

    # 2a. Default font analysis
    default_font = ImageFont.load_default()
    report("Stage2", "default_font_type", True, f"{type(default_font).__name__}")

    img_def = Image.new("RGB", (800, 50), "white")
    draw_def = ImageDraw.Draw(img_def)
    draw_def.text((10, 10), test_text_long, fill="black", font=default_font)
    img_def.save(diag_dir / "render_default_font.png")

    gray_def = img_def.convert("L")
    px_def = list(gray_def.getdata())
    ink_def = sum(1 for p in px_def if p < 200)
    report("Stage2", "default_font_ink_pixels", True,
           f"{ink_def} ink pixels for default font rendering")

    # Check if default font renders recognizable CJK or just boxes
    # Render "A" and "马" separately - if they have same pixel pattern, font doesn't support CJK
    img_a = Image.new("L", (30, 30), 255)
    draw_a = ImageDraw.Draw(img_a)
    draw_a.text((5, 5), "A", fill=0, font=default_font)
    ink_a = sum(1 for p in img_a.getdata() if p < 200)

    img_ma = Image.new("L", (30, 30), 255)
    draw_ma = ImageDraw.Draw(img_ma)
    draw_ma.text((5, 5), "马", fill=0, font=default_font)
    ink_ma = sum(1 for p in img_ma.getdata() if p < 200)

    # "A" and "马" should have very different pixel patterns if CJK is supported
    report("Stage2", "default_font_cjk_vs_ascii",
           abs(ink_ma - ink_a) > 10,
           f"ink pixels: 'A'={ink_a}, '马'={ink_ma}, diff={abs(ink_ma-ink_a)}\n"
           f"(if diff<=10, default font renders CJK as tiny boxes similar to ASCII)")

    # 2b. CJK font
    cjk_font = find_cjk_font(18)
    report("Stage2", "cjk_font_found", cjk_font is not None,
           f"font={cjk_font.getname() if cjk_font and hasattr(cjk_font, 'getname') else cjk_font}")

    if cjk_font:
        img_cjk = Image.new("RGB", (800, 50), "white")
        draw_cjk = ImageDraw.Draw(img_cjk)
        draw_cjk.text((10, 10), test_text_long, fill="black", font=cjk_font)
        img_cjk.save(diag_dir / "render_cjk_font.png")

        gray_cjk = img_cjk.convert("L")
        px_cjk = list(gray_cjk.getdata())
        ink_cjk = sum(1 for p in px_cjk if p < 200)
        report("Stage2", "cjk_font_ink_pixels", ink_cjk > ink_def,
               f"{ink_cjk} ink pixels (vs default: {ink_def})")

        report("Stage2", "comparison_images_saved", True,
               f"default: {diag_dir}/render_default_font.png\n"
               f"CJK:     {diag_dir}/render_cjk_font.png")

    # 2c. What does the RUNNING server code use?
    try:
        import inspect
        from docthinker.server.routers.query import _render_text_to_image
        source = inspect.getsource(_render_text_to_image)
        uses_find_cjk = "_find_cjk_font" in source
        uses_load_default = "load_default" in source
        report("Stage2", "server_code_uses_cjk", uses_find_cjk,
               f"_find_cjk_font={uses_find_cjk}, load_default={uses_load_default}\n"
               f"(if _find_cjk_font=False, the fix has NOT been applied to the running code!)")
    except Exception as e:
        report("Stage2", "server_code_inspect", False, f"Could not inspect: {e}")

    return cjk_font, True


# ─────────────────────────────────────────────────────────────────────
# STAGE 3: Image rendering (full text, compare both fonts)
# ─────────────────────────────────────────────────────────────────────
def test_stage3_render(text: str):
    stage_header("STAGE 3: Image Rendering (Full Document)")

    if not text:
        report("Stage3", "input_text_available", False, "no text from Stage 1")
        return None

    from PIL import Image, ImageDraw, ImageFont

    diag_dir = PROJECT_ROOT / "data" / "#00004" / "knowledge" / "quick_qa" / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 3a. Render with DEFAULT font (reproduce the bug)
    default_font = ImageFont.load_default()
    lines_d = []
    for para in text.replace("\r", "").split("\n"):
        if not para.strip():
            lines_d.append("")
            continue
        lines_d.extend(textwrap.wrap(para, width=80))

    lh_d = default_font.getbbox("A")[3] - default_font.getbbox("A")[1] + 6
    h_d = max(200, 40 + lh_d * len(lines_d))
    img_d = Image.new("RGB", (1200, h_d), "white")
    draw_d = ImageDraw.Draw(img_d)
    y = 20
    for line in lines_d:
        draw_d.text((20, y), line, fill="black", font=default_font)
        y += lh_d
    default_path = diag_dir / "full_render_DEFAULT.png"
    img_d.save(default_path)

    gray_d = img_d.convert("L")
    ink_d = sum(1 for p in gray_d.getdata() if p < 200)
    report("Stage3", "default_font_full_render", True,
           f"saved: {default_path}\nink pixels: {ink_d}, size: {img_d.width}x{img_d.height}")

    # 3b. Render with CJK font (the fix)
    cjk_font = find_cjk_font(18)
    cjk_path = None
    if cjk_font:
        chars_per_line = 60
        lines_c = []
        for para in text.replace("\r", "").split("\n"):
            if not para.strip():
                lines_c.append("")
                continue
            while para:
                lines_c.append(para[:chars_per_line])
                para = para[chars_per_line:]

        lh_c = 26
        h_c = max(200, 40 + lh_c * len(lines_c))
        img_c = Image.new("RGB", (1200, h_c), "white")
        draw_c = ImageDraw.Draw(img_c)
        y = 20
        for line in lines_c:
            draw_c.text((20, y), line, fill="black", font=cjk_font)
            y += lh_c
        cjk_path = diag_dir / "full_render_CJK.png"
        img_c.save(cjk_path)

        gray_c = img_c.convert("L")
        ink_c = sum(1 for p in gray_c.getdata() if p < 200)
        report("Stage3", "cjk_font_full_render", True,
               f"saved: {cjk_path}\nink pixels: {ink_c}, size: {img_c.width}x{img_c.height}")

        report("Stage3", "cjk_has_more_ink", ink_c > ink_d * 2,
               f"CJK ink={ink_c} vs Default ink={ink_d}, ratio={ink_c/ink_d if ink_d else 'inf':.1f}x\n"
               f"(CJK should have significantly more ink if rendering real glyphs)")
    else:
        report("Stage3", "cjk_font_full_render", False, "no CJK font found!")

    # 3c. Check the OLD cached production image
    old_image = PROJECT_ROOT / "data" / "#00004" / "knowledge" / "quick_qa" / "mayun_quick_qa.png"
    if old_image.exists():
        img_old = Image.open(old_image)
        gray_old = img_old.convert("L")
        ink_old = sum(1 for p in gray_old.getdata() if p < 200)
        total_old = img_old.width * img_old.height
        ratio_old = ink_old / total_old
        report("Stage3", "old_production_image",
               ratio_old > 0.02,
               f"mayun_quick_qa.png: ink={ink_old}, total={total_old}, ratio={ratio_old:.4f}\n"
               f"size: {img_old.width}x{img_old.height}\n"
               f"(ratio<0.02 means the production image is garbled and VLM sees boxes)")
    else:
        report("Stage3", "old_production_image", True, "no old cached image")

    # 3d. Now test _render_text_to_image from the actual server code
    try:
        from docthinker.server.routers.query import _render_text_to_image
        server_path = _render_text_to_image(text, diag_dir, "server_code_test")
        if server_path and server_path.exists():
            img_s = Image.open(server_path)
            gray_s = img_s.convert("L")
            ink_s = sum(1 for p in gray_s.getdata() if p < 200)
            total_s = img_s.width * img_s.height
            ratio_s = ink_s / total_s
            report("Stage3", "server_render_function", ratio_s > 0.02,
                   f"saved: {server_path}\nink={ink_s}, ratio={ratio_s:.4f}, size={img_s.width}x{img_s.height}\n"
                   f"(this is what the ACTUAL server code produces)")
        else:
            report("Stage3", "server_render_function", False, "returned None")
    except Exception as e:
        report("Stage3", "server_render_function", False, f"Error: {e}")

    return cjk_path or default_path


# ─────────────────────────────────────────────────────────────────────
# STAGE 4: VLM API call (test with both good and bad images)
# ─────────────────────────────────────────────────────────────────────
async def test_stage4_vlm(good_image_path: Path):
    stage_header("STAGE 4: VLM API Call")

    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    vlm_host = os.getenv("LLM_VLM_HOST") or os.getenv("LLM_BINDING_HOST") or "https://api.bltcy.ai/v1"
    vlm_model = os.getenv("VLM_MODEL") or "qwen3-vl-235b-a22b"

    report("Stage4", "api_key_present", bool(api_key),
           f"key={'***' + api_key[-4:] if api_key and len(api_key)>4 else 'MISSING'}")
    report("Stage4", "vlm_endpoint", True, f"{vlm_host}, model={vlm_model}")

    if not api_key:
        report("Stage4", "SKIPPED", False, "no API key; set LLM_BINDING_API_KEY to test VLM calls")
        return

    from docthinker.auto_thinking.vlm_client import VLMClient
    client = VLMClient(api_key=api_key, api_base=vlm_host, model=vlm_model)

    garbled_indicators = [
        "乱码", "□", "不可识别", "损坏", "garbled", "unreadable", "cannot read",
        "方框", "不可读", "无法识别", "占位符", "编码错误", "无法解析",
        "corrupted", "broken", "gibberish", "unrecognizable",
    ]

    # 4a. Test with old production image (should show garbled)
    old_image = PROJECT_ROOT / "data" / "#00004" / "knowledge" / "quick_qa" / "mayun_quick_qa.png"
    if old_image.exists():
        try:
            prompt = "Please describe what you see in this image. Is the text readable?"
            t0 = time.time()
            answer = await client.generate(prompt, images=[str(old_image)], max_tokens=500)
            elapsed = time.time() - t0
            has_garbled = any(ind in answer for ind in garbled_indicators)
            report("Stage4", "old_image_vlm_response", True,
                   f"[{elapsed:.1f}s] VLM says about OLD image:\n{answer[:400]}")
            report("Stage4", "old_image_is_garbled", has_garbled,
                   f"garbled detected={has_garbled} (expected: True, confirming the bug)")
        except Exception as e:
            report("Stage4", "old_image_vlm_call", False, str(e))

    # 4b. Test with NEW CJK-rendered image (should be readable)
    diag_dir = PROJECT_ROOT / "data" / "#00004" / "knowledge" / "quick_qa" / "diag"
    cjk_image = diag_dir / "full_render_CJK.png"
    if cjk_image.exists():
        try:
            prompt = "请根据文档图片回答：这个文档讲了什么？"
            t0 = time.time()
            answer = await client.generate(prompt, images=[str(cjk_image)], max_tokens=500)
            elapsed = time.time() - t0
            has_garbled = any(ind in answer for ind in garbled_indicators)
            # Check if it mentions Ma Yun/Jack Ma content
            has_content = any(k in answer for k in ["马云", "杭州", "高考", "师范", "浙江"])
            report("Stage4", "cjk_image_vlm_response", True,
                   f"[{elapsed:.1f}s] VLM says about CJK image:\n{answer[:400]}")
            report("Stage4", "cjk_image_readable", not has_garbled and has_content,
                   f"garbled={has_garbled}, has_content={has_content}\n"
                   f"(expected: garbled=False, has_content=True)")
        except Exception as e:
            report("Stage4", "cjk_image_vlm_call", False, str(e))
    else:
        report("Stage4", "cjk_image_test", False, "CJK rendered image not found")

    # 4c. Alternative: send raw text directly to LLM instead of rendering to image
    # This tests whether skipping image rendering entirely would work
    test_file = PROJECT_ROOT / "data" / "#00004" / "content" / "mayun.txt"
    if test_file.exists():
        raw_text = test_file.read_text(encoding="utf-8", errors="ignore").strip()
        try:
            prompt = f"以下是一个文档的内容，请总结这个文档讲了什么：\n\n{raw_text}"
            t0 = time.time()
            answer = await client.generate(prompt, max_tokens=500)
            elapsed = time.time() - t0
            has_content = any(k in answer for k in ["马云", "杭州", "高考", "师范", "浙江"])
            report("Stage4", "direct_text_to_llm", has_content,
                   f"[{elapsed:.1f}s] LLM response (text-only, no image):\n{answer[:400]}")
        except Exception as e:
            report("Stage4", "direct_text_to_llm", False, str(e))

    try:
        await client.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# STAGE 5: Alternative approach analysis
# ─────────────────────────────────────────────────────────────────────
def test_stage5_alternatives():
    stage_header("STAGE 5: Root Cause & Alternative Solutions")

    # 5a. The fundamental question: WHY render text to image then send to VLM?
    report("Stage5", "architecture_analysis", True,
           "Current flow: txt -> render_to_image -> VLM (vision model)\n"
           "Problem: rendering CJK text to image requires CJK font\n"
           "Better flow: txt -> send text directly to LLM (no image needed)")

    # 5b. Check if the question routing is correct
    test_questions = [
        "这个文档讲了什么",
        "告诉我马云的祖籍",
        "summarize this document",
        "what is in this file",
    ]

    # Inline the check (avoid import issues)
    keywords = [
        "文件", "文档", "附件", "内容", "总结", "表格", "图片",
        "file", "document", "attachment", "uploaded", "summarize",
        "this doc", "this file", "pdf",
    ]
    for q in test_questions:
        q_lower = q.strip().lower()
        is_file_q = any(k in q_lower for k in keywords)
        report("Stage5", f"routes_to_fast_qa: '{q}'", True,
               f"result={is_file_q}")

    # 5c. Recommendation
    report("Stage5", "RECOMMENDATION", True,
           "For .txt files, the fast_qa path should either:\n"
           "  Option A: Use CJK font for rendering (our fix)\n"
           "  Option B: Send raw text directly to LLM without image rendering\n"
           "  Option C: Fall through to normal RAG query for .txt files\n"
           "Option B is simplest and most robust - no font dependency at all")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  GARBLED TEXT DIAGNOSTIC - Full Pipeline Trace")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Python: {sys.version}")
    print("=" * 70)

    text = test_stage1_source_file()
    cjk_font, pil_ok = test_stage2_font()
    image_path = test_stage3_render(text) if pil_ok else None
    asyncio.run(test_stage4_vlm(image_path))
    test_stage5_alternatives()

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    print(f"  Total tests: {total}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")

    if failed > 0:
        print(f"\n  FAILURES:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"    [FAIL] {r['stage']} / {r['test']}")
                if r["detail"]:
                    for line in r["detail"].split("\n"):
                        print(f"           {line}")
        print()

    # Save results
    out_path = PROJECT_ROOT / "data" / "#00004" / "code" / "diagnostic_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "results": RESULTS},
                  f, ensure_ascii=False, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
