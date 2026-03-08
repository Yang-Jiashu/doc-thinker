"""
MinerU Extractor — layout-aware PDF parsing via MinerU CLI.

Wraps the existing ``MineruParser`` to produce a ``PDFExtractionResult``.
All text blocks and multimodal items (images, tables, equations) are returned
separately so the orchestrator can route them into the appropriate pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from docthinker.pdf_pipeline import PDFExtractionResult

logger = logging.getLogger(__name__)


class MineruExtractor:
    """Run MinerU on a PDF and return structured extraction results."""

    def __init__(
        self,
        parse_method: str = "auto",
        output_dir: Optional[str] = None,
        skip_ocr_retry: bool = True,
        **parser_kwargs,
    ):
        self.parse_method = parse_method
        self.output_dir = output_dir
        self.skip_ocr_retry = skip_ocr_retry
        self.parser_kwargs = parser_kwargs

    async def extract(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
    ) -> PDFExtractionResult:
        """Parse *pdf_path* with MinerU and return a ``PDFExtractionResult``.

        Runs in a thread-pool because MinerU is a synchronous CLI tool.
        """
        from docthinker.parser import MineruParser

        t0 = time.time()
        resolved_output = output_dir or self.output_dir
        if resolved_output is None:
            resolved_output = str(Path(pdf_path).parent / "mineru_output")

        Path(resolved_output).mkdir(parents=True, exist_ok=True)

        old_skip = os.environ.get("SKIP_MINERU_OCR_RETRY")
        if self.skip_ocr_retry:
            os.environ["SKIP_MINERU_OCR_RETRY"] = "1"

        try:
            parser = MineruParser()
            content_list: List[Dict[str, Any]] = await asyncio.to_thread(
                parser.parse_pdf,
                pdf_path=pdf_path,
                output_dir=resolved_output,
                method=self.parse_method,
                **self.parser_kwargs,
            )
        except Exception as exc:
            logger.error(f"[MineruExtractor] MinerU parsing failed: {exc}")
            return PDFExtractionResult(
                elapsed_seconds=time.time() - t0, source="mineru"
            )
        finally:
            if old_skip is None:
                os.environ.pop("SKIP_MINERU_OCR_RETRY", None)
            else:
                os.environ["SKIP_MINERU_OCR_RETRY"] = old_skip

        text_blocks: List[Dict[str, Any]] = []
        multimodal_blocks: List[Dict[str, Any]] = []
        low_value = {"footer", "header", "page_number"}

        for item in content_list:
            ctype = item.get("type", "text")
            if ctype == "text":
                if (item.get("text") or "").strip():
                    text_blocks.append(item)
            else:
                if ctype in low_value and not (item.get("text") or "").strip():
                    continue
                multimodal_blocks.append(item)

        elapsed = time.time() - t0
        logger.info(
            f"[MineruExtractor] Done in {elapsed:.1f}s — "
            f"{len(text_blocks)} text blocks, {len(multimodal_blocks)} multimodal blocks"
        )

        return PDFExtractionResult(
            text_blocks=text_blocks,
            multimodal_blocks=multimodal_blocks,
            raw_content_list=content_list,
            elapsed_seconds=elapsed,
            source="mineru",
        )
