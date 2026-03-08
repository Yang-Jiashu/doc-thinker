"""
GPT Vision Extractor — page-image based text + visual understanding.

Converts each PDF page to a PNG via PyMuPDF (``fitz``), sends the image to
a GPT Vision model for text extraction, and persists page images for
downstream VLM queries.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from docthinker.pdf_pipeline import PDFExtractionResult

logger = logging.getLogger(__name__)


class VisionExtractor:
    """Use GPT Vision to extract text and visual understanding from a PDF."""

    def __init__(
        self,
        vision_model_func: Optional[Callable] = None,
        concurrency: int = 4,
        dpi: int = 200,
    ):
        self.vision_model_func = vision_model_func
        self.concurrency = concurrency
        self.dpi = dpi

    async def extract(
        self,
        pdf_path: str,
        image_output_dir: Optional[str] = None,
    ) -> PDFExtractionResult:
        """Extract text from *pdf_path* page images via GPT Vision.

        Returns text blocks for the text pipeline and image blocks for the
        multimodal pipeline.  Page images are saved permanently in
        *image_output_dir* (defaults to ``<pdf_parent>/vision_pages``).
        """
        if not self.vision_model_func:
            logger.warning("[VisionExtractor] No vision_model_func — skipping")
            return PDFExtractionResult(source="vision")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("[VisionExtractor] PyMuPDF not installed — skipping")
            return PDFExtractionResult(source="vision")

        t0 = time.time()
        pdf = Path(pdf_path)
        if not pdf.exists() or pdf.suffix.lower() != ".pdf":
            return PDFExtractionResult(source="vision")

        if image_output_dir:
            img_dir = Path(image_output_dir)
        else:
            img_dir = pdf.parent / "vision_pages"
        img_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(pdf))
        except Exception as exc:
            logger.error(f"[VisionExtractor] Failed to open PDF: {exc}")
            return PDFExtractionResult(source="vision")

        semaphore = asyncio.Semaphore(self.concurrency)
        vision_func = self.vision_model_func

        async def _process_page(page_num: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=self.dpi)
                    img_bytes = pix.tobytes("png")

                    page_img = img_dir / f"page_{page_num + 1}.png"
                    page_img.write_bytes(img_bytes)

                    tmp = Path(tempfile.mktemp(suffix=".png"))
                    tmp.write_bytes(img_bytes)
                    try:
                        prompt = (
                            f"这是一个PDF文档的第{page_num + 1}页的截图。"
                            "请将图片中所有的文字内容完整、准确地提取出来。"
                            "保持原文的段落和层次结构。只输出提取到的文字内容，不要添加解释。"
                        )
                        answer = await vision_func(prompt, image_data=str(tmp))
                    finally:
                        tmp.unlink(missing_ok=True)

                    return {
                        "page_num": page_num,
                        "text": (answer or "").strip(),
                        "img_path": str(page_img),
                    }
                except Exception as exc:
                    logger.warning(f"[VisionExtractor] Page {page_num + 1} failed: {exc}")
                    return None

        tasks = [_process_page(i) for i in range(len(doc))]
        results = await asyncio.gather(*tasks)
        doc.close()

        text_blocks: List[Dict[str, Any]] = []
        multimodal_blocks: List[Dict[str, Any]] = []
        page_images: List[str] = []

        for r in results:
            if r is None:
                continue
            pn = r["page_num"]
            if r["text"]:
                text_blocks.append({
                    "type": "text",
                    "text": r["text"],
                    "page_idx": pn,
                })
                logger.info(
                    f"[VisionExtractor] Page {pn + 1}: {len(r['text'])} chars extracted"
                )
            multimodal_blocks.append({
                "type": "image",
                "img_path": r["img_path"],
                "page_idx": pn,
            })
            page_images.append(r["img_path"])

        elapsed = time.time() - t0
        total_chars = sum(len(b.get("text", "")) for b in text_blocks)
        logger.info(
            f"[VisionExtractor] Done in {elapsed:.1f}s — "
            f"{len(text_blocks)} text blocks ({total_chars} chars), "
            f"{len(page_images)} page images"
        )

        return PDFExtractionResult(
            text_blocks=text_blocks,
            multimodal_blocks=multimodal_blocks,
            page_images=page_images,
            elapsed_seconds=elapsed,
            source="vision",
        )
