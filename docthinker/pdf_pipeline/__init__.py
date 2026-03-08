"""
PDF Pipeline — modular dual-path extraction for PDF documents.

Architecture
============
1. **MineruExtractor**   – runs MinerU locally for layout-aware text + image/table extraction.
2. **VisionExtractor**   – converts pages to images via PyMuPDF, sends to GPT Vision.
3. **PDFPipelineOrchestrator** – runs both in parallel, merges results, deduplicates.

Modes (controlled by ``PDF_PIPELINE_MODE`` env or constructor arg):
    * ``full``         – MinerU + GPT Vision in parallel (best quality, slower).
    * ``fast``         – MinerU only (no API cost, fast but lower quality on image PDFs).
    * ``vision_only``  – GPT Vision only (no local model, ~11s per page via API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PDFExtractionResult:
    """Unified result from any extractor."""

    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    multimodal_blocks: List[Dict[str, Any]] = field(default_factory=list)
    page_images: List[str] = field(default_factory=list)
    raw_content_list: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    source: str = ""

    @property
    def text_content(self) -> str:
        """Join all text blocks into a single string with page markers."""
        parts = []
        for b in self.text_blocks:
            text = b.get("text", "").strip()
            if not text:
                continue
            page = b.get("page_idx")
            if page is not None:
                parts.append(f"[page_idx:{page}] {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)

    @property
    def all_blocks(self) -> List[Dict[str, Any]]:
        return self.text_blocks + self.multimodal_blocks


from docthinker.pdf_pipeline.mineru_extractor import MineruExtractor
from docthinker.pdf_pipeline.vision_extractor import VisionExtractor
from docthinker.pdf_pipeline.orchestrator import PDFPipelineOrchestrator

__all__ = [
    "PDFExtractionResult",
    "MineruExtractor",
    "VisionExtractor",
    "PDFPipelineOrchestrator",
]
