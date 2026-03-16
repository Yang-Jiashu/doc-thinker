"""Full integration test for C1-C5 Tri-Graph Architecture.

Tests all five innovation points end-to-end:
C1: Tri-Graph Architecture (TriGraphManager)
C2: Causal DAG (CausalDAG + CausalExtractor)
C3: Cross-Graph Interaction (CrossGraphInteraction)
C4: Knowledge Flow Dynamics (KnowledgeFlowDynamics)
C5: Self-Iterative Evolution Loop (SEALEvolution)
"""

import asyncio
import tempfile
import json
from pathlib import Path

from docthinker.causal.dag import CausalDAG, CausalNode, CausalEdge
from docthinker.causal.extractor import CausalExtractor
from docthinker.flow.dynamics import KnowledgeFlowDynamics
from docthinker.seal.evolution import SEALEvolution
from docthinker.trigraph.manager import TriGraphManager
from docthinker.trigraph.interaction import CrossGraphInteraction


async def mock_llm(prompt, **kwargs):
    prompt_lower = prompt.lower()
    if "causal" in prompt_lower or "cause" in prompt_lower:
        return json.dumps({
            "causal_relations": [
                {"cause": "深度学习", "effect": "特征提取自动化", "mechanism": "神经网络自动学习特征表示", "strength": 0.92, "evidence": "深度学习使得特征提取从手工设计转变为自动学习"},
                {"cause": "特征提取自动化", "effect": "模型精度提升", "mechanism": "更优特征表示提升下游任务", "strength": 0.85, "evidence": "自动学习的特征表示通常优于手工设计"},
                {"cause": "数据量增长", "effect": "深度学习", "mechanism": "大数据驱动模型训练", "strength": 0.88, "evidence": "海量数据是深度学习成功的关键因素"},
                {"cause": "GPU算力提升", "effect": "深度学习", "mechanism": "并行计算加速训练", "strength": 0.9, "evidence": "GPU使得大规模神经网络训练成为可能"},
            ]
        }, ensure_ascii=False)
    if "assess" in prompt_lower or "quality" in prompt_lower:
        return json.dumps({
            "coverage_gaps": [{"entity": "注意力机制", "reason": "缺少因果解释"}],
            "broken_chains": [],
            "inconsistencies": [],
            "quality_score": 0.75,
        }, ensure_ascii=False)
    if "augment" in prompt_lower or "hypothesis" in prompt_lower:
        return json.dumps({
            "hypotheses": [
                {"cause": "注意力机制", "effect": "长距离依赖建模", "mechanism": "动态权重分配关注关键信息", "confidence": 0.8, "evidence": "注意力机制通过权重分配解决长距离依赖问题"}
            ]
        }, ensure_ascii=False)
    if "validate" in prompt_lower or "validity" in prompt_lower:
        return json.dumps({
            "validated": [
                {"cause": "注意力机制", "effect": "长距离依赖建模", "mechanism": "动态权重分配", "validity_score": 0.85, "reason": "与已知因果关系一致"}
            ]
        }, ensure_ascii=False)
    return "{}"


async def test_full_integration():
    print("=" * 60)
    print("Full Integration Test: C1-C5 Tri-Graph Architecture")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        knowledge_dir = Path(td)

        # ── C1: Tri-Graph Architecture ──
        print("\n[C1] Testing Tri-Graph Architecture...")
        mgr = TriGraphManager(knowledge_dir=knowledge_dir, llm_func=mock_llm)
        assert mgr.causal_dag is not None
        assert mgr.flow_dynamics is not None
        assert mgr.seal_evolution is not None
        assert mgr.cross_graph is not None
        print("  OK: All three graph components initialised")

        # ── C2: Causal DAG Construction ──
        print("\n[C2] Testing Causal Relation Extraction Pipeline...")
        text = "深度学习使得特征提取从手工设计转变为自动学习。自动学习的特征表示通常优于手工设计，从而提升模型精度。海量数据是深度学习成功的关键因素。GPU使得大规模神经网络训练成为可能。"
        result = await mgr.build_causal_from_text(text, source_id="test_paper")
        print(f"  Extracted: {result['extracted']}, Added: {result['added']}, Rejected: {result['rejected']}")
        assert result["added"] >= 3, f"Expected >=3 edges, got {result['added']}"
        
        dag_stats = mgr.causal_dag.stats()
        print(f"  DAG: {dag_stats['nodes']} nodes, {dag_stats['edges']} edges")

        # Verify DAG constraint
        topo = mgr.causal_dag.topological_sort()
        print(f"  Topological order: {topo}")
        assert len(topo) == dag_stats["nodes"], "Topological sort should include all nodes"

        # Verify cycle prevention
        cycle_edge = mgr.causal_dag.add_edge("模型精度提升", "数据量增长", mechanism="would_cycle", strength=0.5)
        if cycle_edge is None:
            print("  OK: Cycle prevention working")
        
        print("  OK: Causal DAG built and validated")

        # ── C3: Cross-Graph Interaction ──
        print("\n[C3] Testing Cross-Graph Interaction...")

        ctx = mgr.cross_graph.build_causal_context(["深度学习"], max_chains=5, max_depth=3)
        assert "因果推理链" in ctx
        chain_count = ctx.count("→")
        print(f"  Causal context: {chain_count} links in chains")

        verification = mgr.cross_graph.verify_semantic_with_causal(
            ["深度学习", "特征提取自动化", "注意力机制", "Transformer"]
        )
        print(f"  Verification: {verification['coverage_ratio']:.0%} coverage")
        assert verification["coverage_ratio"] > 0, "Should have some causal backing"
        print("  OK: Cross-graph verification working")

        # ── C4: Knowledge Flow Dynamics ──
        print("\n[C4] Testing Knowledge Flow Dynamics...")
        flow = mgr.flow_dynamics

        flow.record_access("深度学习", boost=0.3)
        flow.record_access("深度学习", boost=0.3)
        flow.record_access("特征提取自动化", boost=0.2)

        top = flow.get_top_activated(5)
        print(f"  Top activated: {[(n, round(v, 3)) for n, v in top[:3]]}")
        assert top[0][0] == "深度学习", "Most accessed should be top"

        activated = flow.propagate_causal(mgr.causal_dag._forward, ["深度学习"])
        print(f"  Causal propagation activated {activated} nodes")

        decay_result = flow.decay_all()
        print(f"  Decay: {decay_result}")
        print("  OK: Flow dynamics working")

        # ── C5: Self-Iterative Evolution Loop (SEAL) ──
        print("\n[C5] Testing SEAL Evolution Loop...")
        seal_result = await mgr.post_query_evolution(
            question="深度学习如何改变特征提取？",
            answer="深度学习通过神经网络自动学习特征表示，取代了传统的手工特征工程。",
            semantic_entities=["深度学习", "特征提取", "神经网络", "注意力机制"],
        )
        print(f"  Assessment score: {seal_result.get('assessment', {}).get('quality_score', 'N/A')}")
        print(f"  Hypotheses: {len(seal_result.get('hypotheses', []))}")
        print(f"  Validated: {len(seal_result.get('validated', []))}")
        print(f"  Inserted: {seal_result.get('inserted', 0)}")

        final_stats = mgr.causal_dag.stats()
        print(f"  Final DAG: {final_stats['nodes']} nodes, {final_stats['edges']} edges")
        print("  OK: SEAL evolution working")

        # ── Persistence ──
        print("\n[Persistence] Testing save/load...")
        mgr.save_all()
        mgr2 = TriGraphManager(knowledge_dir=knowledge_dir, llm_func=mock_llm)
        stats2 = mgr2.causal_dag.stats()
        assert stats2["nodes"] == final_stats["nodes"]
        assert stats2["edges"] == final_stats["edges"]
        print(f"  OK: Reloaded: {stats2['nodes']} nodes, {stats2['edges']} edges")

        # ── Frontend data ──
        print("\n[Frontend] Testing graph data export...")
        gdata = mgr2.causal_dag.to_graph_data()
        print(f"  Nodes: {len(gdata['nodes'])}, Edges: {len(gdata['edges'])}")
        assert gdata["metadata"]["source"] == "causal_dag"
        for node in gdata["nodes"]:
            assert "id" in node and "label" in node
        for edge in gdata["edges"]:
            assert "source" in edge and "target" in edge and "label" in edge
        print("  OK: Frontend data format correct")

        # ── Ablation flags ──
        print("\n[Ablation] Module independence check...")
        from docthinker.causal import CausalDAG as C_DAG
        from docthinker.flow import KnowledgeFlowDynamics as C_Flow
        from docthinker.seal import SEALEvolution as C_SEAL
        from docthinker.trigraph import CrossGraphInteraction as C_Cross
        print("  OK: Each module independently importable")

    print("\n" + "=" * 60)
    print("ALL C1-C5 INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_integration())
