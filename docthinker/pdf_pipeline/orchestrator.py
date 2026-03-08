"""
PDF Pipeline Orchestrator — runs MinerU and GPT Vision extractors,
merges the results, and returns unified content ready for the text
pipeline and multimodal pipeline.

Modes
-----
* ``full``         – MinerU + GPT Vision in parallel.  MinerU provides layout +
                     images/tables, GPT Vision provides text understanding.
* ``fast``         – MinerU only (no API cost).
* ``vision_only``  – GPT Vision only (no local model dependency).

Deduplication
-------------
Text from both sources is kept separate but tagged with its *source*.  The
downstream GraphCore entity merge automatically deduplicates entities with
the same name across both paths, so explicit entity-level dedup is not needed.

For text chunks, we do a lightweight char-level dedup to avoid inserting the
same paragraph twice (when MinerU and Vision both successfully extract text).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from docthinker.pdf_pipeline import PDFExtractionResult
from docthinker.pdf_pipeline.mineru_extractor import MineruExtractor
from docthinker.pdf_pipeline.vision_extractor import VisionExtractor

logger = logging.getLogger(__name__)


def _dedup_text_blocks(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    similarity_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """Return *secondary* blocks that are NOT duplicates of any *primary* block.

    Uses a simple set-of-trigrams Jaccard overlap.  This is intentionally
    lightweight — the heavy-lifting dedup happens in GraphCore entity merge.
    """
    if not primary or not secondary:
        return secondary

    def _trigrams(text: str):
        t = text.replace(" ", "").replace("\n", "")
        return {t[i : i + 3] for i in range(max(0, len(t) - 2))}

    primary_grams = [_trigrams(b.get("text", "")) for b in primary]

    unique: List[Dict[str, Any]] = []
    for sb in secondary:
        sg = _trigrams(sb.get("text", ""))
        if not sg:
            continue
        is_dup = False
        for pg in primary_grams:
            if not pg:
                continue
            intersection = len(sg & pg)
            union = len(sg | pg)
            if union > 0 and intersection / union >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(sb)

    if len(secondary) != len(unique):
        logger.info(
            f"[Dedup] Removed {len(secondary) - len(unique)} duplicate text blocks "
            f"({len(secondary)} vision → {len(unique)} unique)"
        )
    return unique


class PDFPipelineOrchestrator:
    """Orchestrate MinerU + GPT Vision for PDF processing.

    Parameters
    ----------
    mode : str
        One of ``"full"``, ``"fast"``, ``"vision_only"``.
    mineru_extractor : MineruExtractor or None
        Provide a configured extractor or let the orchestrator create a default.
    vision_extractor : VisionExtractor or None
        Provide a configured extractor or let the orchestrator create a default.
    vision_model_func : callable or None
        The async VLM function (used when auto-creating a VisionExtractor).
    """

    def __init__(
        self,
        mode: str = "full",
        mineru_extractor: Optional[MineruExtractor] = None,
        vision_extractor: Optional[VisionExtractor] = None,
        vision_model_func: Optional[Callable] = None,
    ):
        self.mode = mode
        self.mineru_extractor = mineru_extractor or MineruExtractor()
        self.vision_extractor = vision_extractor or VisionExtractor(
            vision_model_func=vision_model_func,
        )

    async def process(
        self,
        pdf_path: str,
        *,
        mineru_output_dir: Optional[str] = None,
        vision_image_dir: Optional[str] = None,
    ) -> "PDFPipelineResult":
        """Run the configured extraction pipeline and return merged results.

        Returns a ``PDFPipelineResult`` with ``text_content`` for the text
        pipeline and ``multimodal_items`` for the multimodal pipeline.
        """
        t0 = time.time()
        mode = self.mode
        logger.info(f"[PDFPipeline] Starting '{mode}' processing for {Path(pdf_path).name}")

        mineru_result = PDFExtractionResult(source="mineru")
        vision_result = PDFExtractionResult(source="vision")

        if mode == "full":
            mineru_result, vision_result = await asyncio.gather(
                self.mineru_extractor.extract(pdf_path, output_dir=mineru_output_dir),
                self.vision_extractor.extract(pdf_path, image_output_dir=vision_image_dir),
            )
        elif mode == "fast":
            mineru_result = await self.mineru_extractor.extract(
                pdf_path, output_dir=mineru_output_dir
            )
        elif mode == "vision_only":
            vision_result = await self.vision_extractor.extract(
                pdf_path, image_output_dir=vision_image_dir
            )
        else:
            logger.warning(f"[PDFPipeline] Unknown mode '{mode}', falling back to 'full'")
            mineru_result, vision_result = await asyncio.gather(
                self.mineru_extractor.extract(pdf_path, output_dir=mineru_output_dir),
                self.vision_extractor.extract(pdf_path, image_output_dir=vision_image_dir),
            )

        # --- merge text -------------------------------------------------------
        # MinerU text is the *primary* source (layout-aware, preserves structure).
        # Vision text is supplementary — only add non-duplicate paragraphs.
        merged_text_blocks = list(mineru_result.text_blocks)
        vision_unique = _dedup_text_blocks(merged_text_blocks, vision_result.text_blocks)
        merged_text_blocks.extend(vision_unique)

        # --- merge multimodal --------------------------------------------------
        # MinerU provides layout-detected images/tables/equations.
        # Vision provides full-page images for VLM queries.
        # Keep both — MinerU's are more precise, Vision's are page-level.
        merged_multimodal = list(mineru_result.multimodal_blocks)

        # Only add vision page images that aren't already covered by MinerU images
        mineru_pages = {b.get("page_idx") for b in mineru_result.multimodal_blocks if b.get("type") == "image"}
        for vb in vision_result.multimodal_blocks:
            if vb.get("type") == "image" and vb.get("page_idx") not in mineru_pages:
                merged_multimodal.append(vb)
            elif vb.get("type") != "image":
                merged_multimodal.append(vb)

        # --- build combined content_list for doc_id generation -----------------
        combined_content_list = merged_text_blocks + merged_multimodal

        elapsed = time.time() - t0
        logger.info(
            f"[PDFPipeline] Completed in {elapsed:.1f}s — "
            f"text: {len(merged_text_blocks)} blocks, "
            f"multimodal: {len(merged_multimodal)} blocks "
            f"(MinerU: {mineru_result.elapsed_seconds:.1f}s, "
            f"Vision: {vision_result.elapsed_seconds:.1f}s)"
        )

        return PDFPipelineResult(
            text_blocks=merged_text_blocks,
            multimodal_blocks=merged_multimodal,
            combined_content_list=combined_content_list,
            mineru_result=mineru_result,
            vision_result=vision_result,
            page_images=vision_result.page_images or [],
            elapsed_seconds=elapsed,
        )


class PDFPipelineResult:
    """Merged result from the PDF pipeline orchestrator."""

    __slots__ = (
        "text_blocks",
        "multimodal_blocks",
        "combined_content_list",
        "mineru_result",
        "vision_result",
        "page_images",
        "elapsed_seconds",
    )

    def __init__(
        self,
        text_blocks: List[Dict[str, Any]],
        multimodal_blocks: List[Dict[str, Any]],
        combined_content_list: List[Dict[str, Any]],
        mineru_result: PDFExtractionResult,
        vision_result: PDFExtractionResult,
        page_images: List[str],
        elapsed_seconds: float,
    ):
        self.text_blocks = text_blocks
        self.multimodal_blocks = multimodal_blocks
        self.combined_content_list = combined_content_list
        self.mineru_result = mineru_result
        self.vision_result = vision_result
        self.page_images = page_images
        self.elapsed_seconds = elapsed_seconds

    @property
    def text_content(self) -> str:
        """Join text blocks into a single string with page markers."""
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
    def content_list_for_doc_id(self) -> List[Dict[str, Any]]:
        """Content list suitable for computing a deterministic doc ID."""
        return self.combined_content_list
