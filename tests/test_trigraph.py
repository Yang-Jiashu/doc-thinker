"""Integration test for TriGraphManager."""

import asyncio
import tempfile
from pathlib import Path
from docthinker.trigraph.manager import TriGraphManager


async def mock_llm(prompt, **kwargs):
    """Mock LLM that returns structured JSON for causal extraction."""
    if "cause-effect" in prompt.lower() or "causal" in prompt.lower():
        return '''{
            "causal_relations": [
                {"cause": "气候变化", "effect": "海平面上升", "mechanism": "全球变暖导致冰川融化", "strength": 0.9, "evidence": "全球变暖导致冰川融化引起海平面上升"},
                {"cause": "海平面上升", "effect": "沿海城市洪灾", "mechanism": "海水倒灌", "strength": 0.8, "evidence": "海平面上升引起沿海城市洪灾"}
            ]
        }'''
    if "assess" in prompt.lower() or "quality" in prompt.lower():
        return '{"coverage_gaps": [], "broken_chains": [], "inconsistencies": [], "quality_score": 0.8}'
    return '{"result": "ok"}'


async def test_trigraph():
    with tempfile.TemporaryDirectory() as td:
        mgr = TriGraphManager(knowledge_dir=Path(td), llm_func=mock_llm)

        # Test causal extraction
        result = await mgr.build_causal_from_text(
            "全球变暖导致冰川融化引起海平面上升。海平面上升引起沿海城市洪灾。",
            source_id="test_doc",
        )
        assert result["added"] >= 1, f"Should add causal edges, got: {result}"
        print(f"  Causal extraction: {result}")

        # Test DAG state
        stats = mgr.causal_dag.stats()
        assert stats["nodes"] >= 2
        assert stats["edges"] >= 1
        print(f"  DAG stats: {stats}")

        # Test deep mode query context
        ctx = await mgr.deep_mode_query_context(
            question="气候变化对城市有什么影响？",
            episodic_concepts=["气候", "海平面"],
        )
        print(f"  Query context: causal={bool(ctx['causal_context'])}, activated={ctx['flow_activated']}")

        # Test graph data for frontend
        gdata = mgr.causal_dag.to_graph_data()
        assert len(gdata["nodes"]) >= 2
        assert len(gdata["edges"]) >= 1
        print(f"  Graph data: {len(gdata['nodes'])} nodes, {len(gdata['edges'])} edges")

        # Test persistence
        mgr.save_all()
        mgr2 = TriGraphManager(knowledge_dir=Path(td), llm_func=mock_llm)
        stats2 = mgr2.causal_dag.stats()
        assert stats2["nodes"] == stats["nodes"]
        assert stats2["edges"] == stats["edges"]
        print(f"  Persistence: OK (reloaded {stats2['nodes']} nodes, {stats2['edges']} edges)")

        # Test SEAL evolution
        seal_result = await mgr.post_query_evolution(
            question="气候变化的影响",
            answer="气候变化导致海平面上升和洪灾",
        )
        print(f"  SEAL evolution: inserted={seal_result.get('inserted', 0)}")

        print("\nAll TriGraph integration tests passed!")


if __name__ == "__main__":
    asyncio.run(test_trigraph())
