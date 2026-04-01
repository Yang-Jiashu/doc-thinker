"""Subgraph selection strategies for the KG self-study loop.

Each strategy returns a list of entity dicts + relation dicts forming the
subgraph region to study.  Weights are tuned for HotpotQA-style benchmarks
(80% bridge, 20% comparison).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

_log = logging.getLogger("docthinker.kg_self_study.selector")

# Default strategy allocation (HotpotQA-optimised)
DEFAULT_STRATEGY_WEIGHTS = {
    "bridge_entity": 0.40,
    "two_hop_completion": 0.20,
    "comparison_alignment": 0.15,
    "weak_component": 0.10,
    "hub_enrichment": 0.10,
    "evidence_chain": 0.05,
}


class SubgraphSelector:
    """Pick KG regions for each self-study round."""

    def __init__(
        self,
        strategy_weights: Optional[Dict[str, float]] = None,
        max_entities_per_round: int = 40,
    ):
        self.weights = strategy_weights or DEFAULT_STRATEGY_WEIGHTS
        self.max_entities = max_entities_per_round
        self._studied_entities: Set[str] = set()

    def select(
        self,
        all_nodes: List[Dict[str, Any]],
        all_edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a subgraph selection with strategy metadata.

        Returns dict with keys: entities, relations, strategy, metadata.
        """
        strategy = self._pick_strategy()
        _log.info("[selector] round strategy: %s", strategy)

        selector_fn = {
            "bridge_entity": self._select_bridge,
            "two_hop_completion": self._select_two_hop,
            "comparison_alignment": self._select_comparison,
            "weak_component": self._select_weak_component,
            "hub_enrichment": self._select_hub,
            "evidence_chain": self._select_evidence,
        }.get(strategy, self._select_bridge)

        result = selector_fn(all_nodes, all_edges)
        result["strategy"] = strategy
        return result

    # -- strategy dispatch ---------------------------------------------------

    def _pick_strategy(self) -> str:
        strategies = list(self.weights.keys())
        weights = [self.weights[s] for s in strategies]
        return random.choices(strategies, weights=weights, k=1)[0]

    # -- Strategy 1: Bridge Entity (40%) ------------------------------------

    def _select_bridge(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        """Find entities appearing in >=2 different source_ids."""
        entity_sources: Dict[str, Set[str]] = defaultdict(set)
        node_map: Dict[str, Dict] = {}
        for n in nodes:
            name = n.get("id") or n.get("entity_id") or ""
            sources = str(n.get("source_id", "")).split("<SEP>")
            entity_sources[name] = {s.strip() for s in sources if s.strip()}
            node_map[name] = n

        bridge_entities = [
            name for name, srcs in entity_sources.items()
            if len(srcs) >= 2 and name not in self._studied_entities
        ]

        if not bridge_entities:
            bridge_entities = [
                name for name, srcs in entity_sources.items()
                if len(srcs) >= 2
            ]

        if not bridge_entities:
            return self._fallback_random(nodes, edges)

        random.shuffle(bridge_entities)
        selected_names = set(bridge_entities[: self.max_entities // 2])

        neighbors = self._get_neighbors(selected_names, edges)
        selected_names.update(list(neighbors)[: self.max_entities // 2])

        return self._build_result(selected_names, node_map, edges,
                                  metadata={"bridge_count": len(bridge_entities)})

    # -- Strategy 2: Comparison Alignment (15%) -----------------------------

    def _select_comparison(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        """Find same-type entity pairs without direct edges."""
        type_groups: Dict[str, List[str]] = defaultdict(list)
        node_map: Dict[str, Dict] = {}
        for n in nodes:
            name = n.get("id") or n.get("entity_id") or ""
            etype = n.get("entity_type", "unknown")
            type_groups[etype].append(name)
            node_map[name] = n

        edge_set = set()
        for e in edges:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            edge_set.add((s, t))
            edge_set.add((t, s))

        pairs = []
        for etype, members in type_groups.items():
            if len(members) < 2:
                continue
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if (a, b) not in edge_set:
                        pairs.append((a, b, etype))

        if not pairs:
            return self._fallback_random(nodes, edges)

        random.shuffle(pairs)
        selected_names: Set[str] = set()
        for a, b, _ in pairs[: self.max_entities // 2]:
            selected_names.update({a, b})

        return self._build_result(selected_names, node_map, edges,
                                  metadata={"comparison_pairs": len(pairs)})

    # -- Strategy 3: Two-Hop Completion (20%) --------------------------------

    def _select_two_hop(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        """Find A->B->C where A and C have no direct edge."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        node_map: Dict[str, Dict] = {}
        for n in nodes:
            name = n.get("id") or n.get("entity_id") or ""
            node_map[name] = n
        for e in edges:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            adj[s].add(t)
            adj[t].add(s)

        gaps: List[Tuple[str, str, str]] = []
        for b, b_neighbors in adj.items():
            b_list = list(b_neighbors)
            for i, a in enumerate(b_list):
                for c in b_list[i + 1:]:
                    if c not in adj.get(a, set()):
                        gaps.append((a, b, c))

        if not gaps:
            return self._fallback_random(nodes, edges)

        random.shuffle(gaps)
        selected_names: Set[str] = set()
        for a, b, c in gaps[: self.max_entities // 3]:
            selected_names.update({a, b, c})

        return self._build_result(selected_names, node_map, edges,
                                  metadata={"two_hop_gaps": len(gaps)})

    # -- Strategy 4: Hub Enrichment (10%) ------------------------------------

    def _select_hub(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        """Select highest-degree entities for detail enrichment."""
        degree: Dict[str, int] = defaultdict(int)
        node_map: Dict[str, Dict] = {}
        for n in nodes:
            name = n.get("id") or n.get("entity_id") or ""
            node_map[name] = n
        for e in edges:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            degree[s] += 1
            degree[t] += 1

        ranked = sorted(degree.items(), key=lambda x: -x[1])
        top_hubs = [name for name, _ in ranked[: self.max_entities // 3]]

        selected_names = set(top_hubs)
        neighbors = self._get_neighbors(selected_names, edges)
        selected_names.update(list(neighbors)[: self.max_entities * 2 // 3])

        return self._build_result(selected_names, node_map, edges,
                                  metadata={"hub_degrees": dict(ranked[:10])})

    # -- Strategy 5: Weak Component Bridging (10%) ---------------------------

    def _select_weak_component(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        """Find weakly connected or isolated clusters."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        node_map: Dict[str, Dict] = {}
        all_names: Set[str] = set()
        for n in nodes:
            name = n.get("id") or n.get("entity_id") or ""
            node_map[name] = n
            all_names.add(name)
        for e in edges:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            adj[s].add(t)
            adj[t].add(s)

        visited: Set[str] = set()
        components: List[Set[str]] = []
        for name in all_names:
            if name in visited:
                continue
            component: Set[str] = set()
            stack = [name]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adj.get(current, set()) - visited)
            components.append(component)

        if len(components) < 2:
            return self._fallback_random(nodes, edges)

        components.sort(key=len)
        selected_names: Set[str] = set()
        small_components = [c for c in components if len(c) <= 5]
        main_component = max(components, key=len)

        for comp in small_components[: 3]:
            selected_names.update(comp)

        main_sample = random.sample(
            list(main_component),
            min(self.max_entities // 2, len(main_component)),
        )
        selected_names.update(main_sample)

        return self._build_result(selected_names, node_map, edges,
                                  metadata={"component_count": len(components)})

    # -- Strategy 6: Evidence Chain (5%) -------------------------------------

    def _select_evidence(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        """Select edges with weak evidence (single source, low weight)."""
        node_map = {
            (n.get("id") or n.get("entity_id") or ""): n for n in nodes
        }
        weak = []
        for e in edges:
            source_id = str(e.get("source_id", ""))
            weight = float(e.get("weight", 0.5))
            sources = [s for s in source_id.split("<SEP>") if s.strip()]
            if len(sources) <= 1 or weight < 0.4:
                weak.append(e)

        if not weak:
            return self._fallback_random(nodes, edges)

        random.shuffle(weak)
        selected_names: Set[str] = set()
        for e in weak[: self.max_entities // 2]:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            selected_names.update({s, t})

        return self._build_result(selected_names, node_map, edges,
                                  metadata={"weak_edge_count": len(weak)})

    # -- helpers -------------------------------------------------------------

    def _get_neighbors(
        self, entity_names: Set[str], edges: List[Dict],
    ) -> Set[str]:
        neighbors: Set[str] = set()
        for e in edges:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            if s in entity_names:
                neighbors.add(t)
            if t in entity_names:
                neighbors.add(s)
        return neighbors - entity_names

    def _build_result(
        self,
        selected_names: Set[str],
        node_map: Dict[str, Dict],
        all_edges: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        self._studied_entities.update(selected_names)
        entities = [node_map[n] for n in selected_names if n in node_map]
        relations = []
        for e in all_edges:
            s = str(e.get("source") or e.get("src_id", ""))
            t = str(e.get("target") or e.get("tgt_id", ""))
            if s in selected_names and t in selected_names:
                relations.append(e)
        return {
            "entities": entities,
            "relations": relations,
            "metadata": metadata or {},
        }

    def _fallback_random(
        self, nodes: List[Dict], edges: List[Dict],
    ) -> Dict[str, Any]:
        _log.info("[selector] falling back to random selection")
        node_map = {
            (n.get("id") or n.get("entity_id") or ""): n for n in nodes
        }
        sample = random.sample(nodes, min(self.max_entities, len(nodes)))
        names = {n.get("id") or n.get("entity_id") or "" for n in sample}
        return self._build_result(names, node_map, edges,
                                  metadata={"fallback": True})

    def mark_studied(self, entity_names: Set[str]) -> None:
        self._studied_entities.update(entity_names)

    def reset_studied(self) -> None:
        self._studied_entities.clear()
