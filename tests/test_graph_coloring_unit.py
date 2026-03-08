import unittest

from docthinker.image_assets import (
    DEFAULT_NODE_COLOR,
    EXPANDED_NODE_COLOR,
    IMAGE_NODE_COLOR,
    is_image_node,
    resolve_graph_node_color,
)
from docthinker.server.routers.graph import _is_expanded_node


class GraphColoringUnitTest(unittest.TestCase):
    def test_image_node_detection(self):
        self.assertTrue(is_image_node({"entity_type": "image_asset"}))
        self.assertTrue(is_image_node({"is_image_node": 1}))
        self.assertTrue(is_image_node({"source_id": "image_asset:abc"}))
        self.assertFalse(is_image_node({"entity_type": "concept"}))

    def test_color_priority(self):
        self.assertEqual(
            IMAGE_NODE_COLOR,
            resolve_graph_node_color(
                {"entity_type": "image_asset"}, is_expanded=False
            ),
        )
        self.assertEqual(
            IMAGE_NODE_COLOR,
            resolve_graph_node_color(
                {"entity_type": "image_asset"}, is_expanded=True
            ),
        )
        self.assertEqual(
            EXPANDED_NODE_COLOR,
            resolve_graph_node_color({"entity_type": "concept"}, is_expanded=True),
        )
        self.assertEqual(
            DEFAULT_NODE_COLOR,
            resolve_graph_node_color({"entity_type": "concept"}, is_expanded=False),
        )

    def test_expanded_flag(self):
        self.assertTrue(_is_expanded_node({"is_expanded": 1}))
        self.assertTrue(_is_expanded_node({"source_id": "llm_expansion"}))
        self.assertFalse(_is_expanded_node({"is_expanded": 0}))


if __name__ == "__main__":
    unittest.main()
