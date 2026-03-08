---
name: query-forced-expanded-routing
description: Inject matched expanded nodes into retrieval instructions, combine with session memory and KG evidence, and keep query routing extensible.
---

# Query Forced Expanded Routing

Use this skill when changing query orchestration that must include expanded nodes.

## Scope

- Expanded node matching before query execution
- Merging retrieval instructions
- Thinking-mode and non-thinking-mode query path consistency
- Post-answer usage feedback for expanded nodes

## Entry Points

- `docthinker/server/schemas.py`
- `docthinker/server/routers/query.py`

## Workflow

1. Add request knobs on `QueryRequest` instead of hardcoding behavior.
2. Resolve expanded-node matches with `ExpandedNodeManager`.
3. Merge user retrieval instructions with expansion instructions.
4. Pass merged instructions to `session_rag.aquery(..., user_prompt=...)`.
5. Do promotion and graph-merge in background tasks.

## Guardrails

- Keep query latency path short; heavy enrichment must be background.
- Return debug-friendly metadata (`expanded_matches`, instruction applied).
- Preserve existing fallback behavior.

