import logging
import tempfile
import unittest
from pathlib import Path

from docthinker.processor import ProcessorMixin


class _GraphStorage:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def get_all_nodes(self):
        out = []
        for node_id, node_data in self.nodes.items():
            row = dict(node_data)
            row["id"] = node_id
            out.append(row)
        return out

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)


class _GraphCore:
    def __init__(self):
        self.chunk_entity_relation_graph = _GraphStorage()


class _DummyProcessor(ProcessorMixin):
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.graphcore = _GraphCore()
        self.logger = logging.getLogger("processor-image-assets-test")
        self.embedding_func = self._embed

    async def _embed(self, texts):
        vectors = []
        for text in texts:
            t = (text or "").lower()
            if "cpu" in t:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class ProcessorImageAssetsUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_persist_image_assets_and_graph_links(self):
        with tempfile.TemporaryDirectory() as td:
            proc = _DummyProcessor(td)
            graph = proc.graphcore.chunk_entity_relation_graph
            graph.nodes["CPU"] = {"entity_type": "concept", "description": "cpu subsystem"}
            graph.nodes["Figure 1"] = {"entity_type": "image", "description": "existing image node"}

            src_img = Path(td) / "source.png"
            src_img.write_bytes(b"PNG")

            multimodal = [
                {
                    "content_type": "image",
                    "description": "This image explains CPU and memory flow.",
                    "entity_info": {
                        "entity_name": "Figure 1",
                        "summary": "CPU architecture overview",
                    },
                    "original_item": {"img_path": str(src_img), "page_idx": 1},
                    "item_info": {"page_idx": 1},
                }
            ]

            await proc._persist_image_assets_and_nodes(
                multimodal_data_list=multimodal,
                file_path="Chap1.pdf",
                doc_id="doc-abc",
            )

            table_file = Path(td) / "multimodal" / "image_assets.json"
            self.assertTrue(table_file.exists())
            self.assertTrue(any("image_asset" in str(k[0]) for k in graph.edges.keys()))
            self.assertTrue(
                any(edge.get("keywords") == "image_asset_of" for edge in graph.edges.values())
            )


if __name__ == "__main__":
    unittest.main()
