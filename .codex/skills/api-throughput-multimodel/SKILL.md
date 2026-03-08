---
name: api-throughput-multimodel
description: Improve API throughput using async I/O, model pooling, and configurable GraphCore concurrency without changing business semantics.
---

# API Throughput Multimodel

Use this skill for performance work on backend query throughput.

## Scope

- Multi-model round-robin for LLM calls
- Async rerank networking
- Reusable HTTP sessions
- GraphCore concurrency tuning via config

## Entry Points

- `docthinker/providers/settings.py`
- `docthinker/server/app.py`
- `docthinker/utils.py`
- `docthinker/auto_thinking/vlm_client.py`

## Workflow

1. Expose all concurrency knobs in settings.
2. Route LLM requests through a bounded-concurrency model router.
3. Ensure network calls are async and session-reused.
4. Pass tuned async values into GraphCore kwargs.
5. Add clean shutdown for long-lived clients.

## Guardrails

- Do not block event loop with sync HTTP in query path.
- Keep defaults conservative but production-ready.
- Keep behavior backward-compatible when only one model is configured.

