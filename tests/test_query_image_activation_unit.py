import logging
import tempfile
import unittest
from pathlib import Path

from docthinker.image_assets import ImageAssetTable
from docthinker.query import QueryMixin


class _DummyQuery(QueryMixin):
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.logger = logging.getLogger("query-image-activation-test")
        self.embedding_func = self._embed

    async def _embed(self, texts):
        vectors = []
        for text in texts:
            q = (text or "").lower()
            if "cpu" in q:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return vectors


class QueryImageActivationUnitTest(unittest.IsolatedAsyncioTestCase):
    async def test_activate_image_assets(self):
        with tempfile.TemporaryDirectory() as td:
            table = ImageAssetTable(td)
            img1 = Path(td) / "multimodal" / "images" / "img1.png"
            img2 = Path(td) / "multimodal" / "images" / "img2.png"
            img1.parent.mkdir(parents=True, exist_ok=True)
            img1.write_bytes(b"PNG")
            img2.write_bytes(b"PNG")
            table.upsert_records(
                [
                    {
                        "image_id": "img-1",
                        "stored_path": str(img1),
                        "embedding": [0.99, 0.01, 0.0],
                    },
                    {
                        "image_id": "img-2",
                        "stored_path": str(img2),
                        "embedding": [0.0, 1.0, 0.0],
                    },
                ]
            )

            q = _DummyQuery(td)
            selected = await q._activate_image_assets_for_query(
                query="CPU architecture",
                threshold=0.6,
                top_k=2,
            )
            self.assertEqual(1, len(selected))
            self.assertEqual("img-1", selected[0]["image_id"])

            block = q._build_activated_image_paths_block(selected)
            self.assertIn("Image Paths:", block)
            self.assertIn(str(img1), block)


if __name__ == "__main__":
    unittest.main()
