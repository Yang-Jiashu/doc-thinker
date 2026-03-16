"""Unit tests for CausalDAG."""

import tempfile
from pathlib import Path
from docthinker.causal.dag import CausalDAG


def test_causal_dag():
    with tempfile.TemporaryDirectory() as td:
        dag = CausalDAG(Path(td) / "test_dag.json")
        dag.add_node("A", "event")
        dag.add_node("B", "event")
        dag.add_node("C", "event")

        # Test edge addition
        e1 = dag.add_edge("A", "B", mechanism="causes", strength=0.9)
        assert e1 is not None, "Edge A->B should be added"

        e2 = dag.add_edge("B", "C", mechanism="leads_to", strength=0.7)
        assert e2 is not None, "Edge B->C should be added"

        # Test cycle prevention
        e3 = dag.add_edge("C", "A", mechanism="would_cycle", strength=0.5)
        assert e3 is None, "Edge C->A should be rejected (cycle)"

        # Test self-loop prevention
        e4 = dag.add_edge("A", "A", mechanism="self", strength=0.5)
        assert e4 is None, "Self-loop should be rejected"

        # Test traversal
        ancestors = dag.get_ancestors("C")
        assert "B" in ancestors, "B should be ancestor of C"
        assert "A" in ancestors, "A should be ancestor of C"

        descendants = dag.get_descendants("A")
        assert "B" in descendants, "B should be descendant of A"
        assert "C" in descendants, "C should be descendant of A"

        # Test causal chains
        chains = dag.get_causal_chains(["A"])
        assert len(chains) > 0, "Should find causal chains"

        # Test topological sort
        topo = dag.topological_sort()
        assert topo.index("A") < topo.index("B"), "A before B"
        assert topo.index("B") < topo.index("C"), "B before C"

        # Test persistence
        dag.save()
        dag2 = CausalDAG(Path(td) / "test_dag.json")
        stats = dag2.stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 2

        # Test graph data export
        gdata = dag2.to_graph_data()
        assert len(gdata["nodes"]) == 3
        assert len(gdata["edges"]) == 2

        print("All CausalDAG tests passed!")


if __name__ == "__main__":
    test_causal_dag()
