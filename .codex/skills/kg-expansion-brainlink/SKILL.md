---
name: kg-expansion-brainlink
description: Build and evolve session-scoped expanded knowledge graph nodes, including candidate generation, lifecycle management, promotion, and root-edge linking.
---

# KG Expansion Brainlink

Use this skill when implementing or modifying expanded-node logic for the knowledge graph.

## Scope

- Candidate generation from current KG context
- Expanded node lifecycle persistence
- Promotion from expanded node to normal node
- Root-node edge linking and answer-entity merging

## Entry Points

- `docthinker/server/routers/graph.py`
- `docthinker/kg_expansion/expander.py`
- `docthinker/kg_expansion/manager.py`
- `docthinker/server/routers/query.py`

## Workflow

1. Keep generation and lifecycle separated.
2. Write candidates to `expanded_nodes.json` through `ExpandedNodeManager`.
3. Never promote on generation; promote only after answer-time usage validation.
4. Promote by updating graph node properties and linking to root entities.
5. Persist graph changes with `index_done_callback`.

## Guardrails

- Keep session scope strict; never cross-write other sessions.
- Treat `source_id=llm_expansion` as expansion provenance.
- Add edges with explicit `keywords`/`description` for auditability.

