"""Test cross-graph interaction - causal chain retrieval."""

import asyncio
import tempfile
from pathlib import Path
from docthinker.causal.dag import CausalDAG
from docthinker.flow.dynamics import KnowledgeFlowDynamics
from docthinker.trigraph.interaction import CrossGraphInteraction


def test_cross_graph():
    with tempfile.TemporaryDirectory() as td:
        dag = CausalDAG(Path(td) / "dag.json")
        flow = KnowledgeFlowDynamics(Path(td) / "flow.json")

        dag.add_edge("气候变化", "冰川融化", mechanism="全球变暖", strength=0.9)
        dag.add_edge("冰川融化", "海平面上升", mechanism="冰体融化注入海洋", strength=0.85)
        dag.add_edge("海平面上升", "沿海洪灾", mechanism="海水倒灌", strength=0.8)
        dag.add_edge("工业排放", "气候变化", mechanism="温室气体增加", strength=0.9)

        interaction = CrossGraphInteraction(causal_dag=dag, flow_dynamics=flow)

        # Test causal context building
        ctx = interaction.build_causal_context(["气候变化"], max_chains=5, max_depth=3)
        assert "气候变化" in ctx, "Should mention source entity"
        assert "因果推理链" in ctx, "Should have header"
        print(f"Causal context:\n{ctx}\n")

        # Test verification
        verification = interaction.verify_semantic_with_causal(
            ["气候变化", "冰川融化", "经济发展"]
        )
        assert "气候变化" in verification["with_causal_backing"]
        assert "冰川融化" in verification["with_causal_backing"]
        assert "经济发展" in verification["without_causal_backing"]
        print(f"Verification: {verification}")

        # Test flow activation after causal context
        assert flow.get_activation("气候变化") > 0.0, "Should have activation"

        print("\nAll CrossGraph tests passed!")


if __name__ == "__main__":
    test_cross_graph()
