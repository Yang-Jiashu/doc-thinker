import tempfile
import unittest
from pathlib import Path

from docthinker.image_assets import (
    ImageAssetTable,
    build_image_asset_id,
    extract_image_items,
)


class ImageAssetsUnitTest(unittest.TestCase):
    def test_upsert_and_activation(self):
        with tempfile.TemporaryDirectory() as td:
            table = ImageAssetTable(td)
            table.upsert_records(
                [
                    {
                        "image_id": "img-1",
                        "stored_path": "a.png",
                        "embedding": [1.0, 0.0],
                    },
                    {
                        "image_id": "img-2",
                        "stored_path": "b.png",
                        "embedding": [0.2, 0.98],
                    },
                ]
            )
            rows = table.load()
            self.assertEqual(2, len(rows))

            selected = table.select_activated_by_embedding(
                query_embedding=[1.0, 0.0],
                threshold=0.6,
                top_k=2,
            )
            self.assertEqual(1, len(selected))
            self.assertEqual("img-1", selected[0]["image_id"])

    def test_copy_and_id(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.png"
            src.write_bytes(b"PNG")
            table = ImageAssetTable(td)
            image_id = build_image_asset_id(
                session_id="#00001",
                doc_id="doc-1",
                source_pdf="a.pdf",
                page_idx=1,
                source_image_path=str(src),
                index=0,
            )
            stored = table.copy_image_to_store(str(src), image_id)
            self.assertTrue(Path(stored).exists())

    def test_extract_image_items(self):
        items = extract_image_items(
            [
                {"type": "text", "text": "hello"},
                {"type": "image", "img_path": "x.png"},
                {"type": "image", "img_path": ""},
                {"type": "table", "img_path": "y.png"},
            ]
        )
        self.assertEqual(1, len(items))
        self.assertEqual("x.png", items[0]["img_path"])


if __name__ == "__main__":
    unittest.main()
