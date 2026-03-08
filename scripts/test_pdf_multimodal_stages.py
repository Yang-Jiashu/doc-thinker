from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from docthinker.image_assets import ImageAssetTable, extract_image_items
from docthinker.parser import MineruParser
from docthinker.processor import ProcessorMixin
from docthinker.query import QueryMixin


class _GraphStorage:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def get_all_nodes(self):
        rows = []
        for node_id, node_data in self.nodes.items():
            row = dict(node_data)
            row["id"] = node_id
            rows.append(row)
        return rows

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)


class _GraphCore:
    def __init__(self):
        self.chunk_entity_relation_graph = _GraphStorage()


class _StageProcessor(ProcessorMixin):
    def __init__(self, working_dir: Path):
        self.working_dir = str(working_dir)
        self.graphcore = _GraphCore()
        self.logger = logging.getLogger("stage-processor")
        self.embedding_func = self._embed

    async def _embed(self, texts):
        vectors = []
        for text in texts:
            t = (text or "").lower()
            if "系统" in t or "cpu" in t or "计算机" in t:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class _StageQuery(QueryMixin):
    def __init__(self, working_dir: Path):
        self.working_dir = str(working_dir)
        self.logger = logging.getLogger("stage-query")
        self.embedding_func = self._embed

    async def _embed(self, texts):
        vectors = []
        for text in texts:
            t = (text or "").lower()
            if "系统" in t or "cpu" in t or "计算机" in t:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


async def run(pdf_path: Path, session_knowledge_dir: Path) -> int:
    print(f"[Stage 1] Parse PDF with MinerU: {pdf_path}")
    parser = MineruParser()
    content_list = parser.parse_pdf(str(pdf_path), output_dir="output", method="auto")
    print(f"  Parsed content blocks: {len(content_list)}")
    if not content_list:
        print("  ERROR: no content parsed")
        return 1

    print("[Stage 2] Extract image blocks from parsed content_list")
    image_blocks = extract_image_items(content_list)
    print(f"  Image blocks: {len(image_blocks)}")
    if not image_blocks:
        print("  WARNING: no image blocks found in this PDF")
        return 0

    print("[Stage 3] Persist image assets and upsert image nodes/edges")
    session_knowledge_dir.mkdir(parents=True, exist_ok=True)
    stage_processor = _StageProcessor(session_knowledge_dir)
    stage_processor.graphcore.chunk_entity_relation_graph.nodes["计算机系统"] = {
        "entity_type": "concept",
        "description": "计算机系统总览",
    }

    multimodal_data = []
    for idx, item in enumerate(image_blocks):
        page_idx = int(item.get("page_idx") or 0)
        multimodal_data.append(
            {
                "content_type": "image",
                "description": f"PDF image block from page {page_idx}.",
                "entity_info": {
                    "entity_name": f"PDF Image {idx + 1}",
                    "summary": f"Page {page_idx} image in {pdf_path.name}",
                },
                "original_item": item,
                "item_info": {"page_idx": page_idx},
            }
        )

    await stage_processor._persist_image_assets_and_nodes(
        multimodal_data_list=multimodal_data,
        file_path=str(pdf_path),
        doc_id="doc-stage-pdf",
    )

    table = ImageAssetTable(session_knowledge_dir)
    records = table.load()
    print(f"  Persisted image assets: {len(records)}")
    print(
        f"  Graph nodes: {len(stage_processor.graphcore.chunk_entity_relation_graph.nodes)} | "
        f"edges: {len(stage_processor.graphcore.chunk_entity_relation_graph.edges)}"
    )

    print("[Stage 4] Query-time image activation by similarity threshold")
    stage_query = _StageQuery(session_knowledge_dir)
    activated = await stage_query._activate_image_assets_for_query(
        query="请结合计算机系统结构回答",
        threshold=0.5,
        top_k=3,
    )
    print(f"  Activated image assets: {len(activated)}")
    if activated:
        print("  Top activated:", activated[0].get("stored_path"))

    print("All stage tests completed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default=r"C:\Users\lqx94\Documents\GitHub\doc\data\#00005\content\Chap1 计算机系统概述.pdf",
    )
    parser.add_argument(
        "--knowledge-dir",
        default=r"C:\Users\lqx94\Documents\GitHub\doc\data\#00005\knowledge",
    )
    args = parser.parse_args()
    return asyncio.run(run(Path(args.pdf), Path(args.knowledge_dir)))


if __name__ == "__main__":
    raise SystemExit(main())
