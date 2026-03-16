"""Unit tests for KnowledgeFlowDynamics."""

import tempfile
from pathlib import Path
from docthinker.flow.dynamics import KnowledgeFlowDynamics
from docthinker.causal.dag import CausalDAG


def test_flow_dynamics():
    with tempfile.TemporaryDirectory() as td:
        flow = KnowledgeFlowDynamics(Path(td) / "flow.json")

        # Test access recording
        val = flow.record_access("entity_A")
        assert val > 0.5, "Activation should be boosted"

        val2 = flow.record_access("entity_A")
        assert val2 > val, "Second access should boost further"

        # Test batch access
        flow.record_batch_access(["entity_B", "entity_C"])
        assert flow.get_activation("entity_B") > 0.0
        assert flow.get_activation("entity_C") > 0.0

        # Test top activated
        top = flow.get_top_activated(5)
        assert len(top) > 0
        assert top[0][0] == "entity_A"  # Most accessed

        # Test decay (should not prune immediately with short time)
        result = flow.decay_all()
        assert result["decayed"] >= 0

        # Test causal propagation
        dag = CausalDAG(Path(td) / "dag.json")
        dag.add_edge("entity_A", "entity_D", strength=0.8)
        dag.add_edge("entity_D", "entity_E", strength=0.6)

        activated = flow.propagate_causal(dag._forward, ["entity_A"])
        assert activated >= 1, "Should activate at least entity_D"

        assert flow.get_activation("entity_D") > 0.0

        # Test persistence
        flow.save()
        flow2 = KnowledgeFlowDynamics(Path(td) / "flow.json")
        assert flow2.get_activation("entity_A") > 0.0

        print("All FlowDynamics tests passed!")


if __name__ == "__main__":
    test_flow_dynamics()
