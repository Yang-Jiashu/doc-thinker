<div align="center">

<img src="docs/assets/banner.png" alt="DocThinker Banner" width="820" />

# DocThinker

**Self-Evolving Knowledge Graphs · Tiered Memory · Structured Reasoning**

*Language captures the results of cognition, while cognition itself encompasses perception, experience, and reasoning.*

[![Paper](https://img.shields.io/badge/arXiv-2603.05551-b31b1b.svg)](https://arxiv.org/pdf/2603.05551)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Live-orange)](http://localhost:5000)
[![LightRAG](https://img.shields.io/badge/LightRAG-Based-8B5CF6)](https://github.com/HKUDS/LightRAG)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Integration-E74C3C)](https://github.com/letta-ai/letta)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-UI-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-KG-4C72B0)](https://networkx.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector-3B5998?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![D3.js](https://img.shields.io/badge/D3.js-Visualization-F9A03C?logo=d3dotjs&logoColor=white)](https://d3js.org/)
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Clustering-27AE60)](https://hdbscan.readthedocs.io/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NER-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)

[English](README.md) | [中文](README.zh-CN.md)

[Quick Start](#-quick-start) · [Key Contributions](#-key-contributions) · [Pipeline](#-overview) · [Use Cases](#-use-cases) · [API Reference](#api-reference)

</div>

## 📖 Overview

**DocThinker** is a document-driven RAG system that constructs self-evolving knowledge graphs from uploaded documents. Unlike conventional retrieve-then-respond pipelines, DocThinker treats knowledge as a **dynamic graph**:

- **Growth** — Goes beyond explicit knowledge in ingested documents, autonomously reasoning and extending implicit associations and related knowledge not directly stated;
- **Evolution** — Splits retrieval, extraction, and answering into three collaborative Agents, using reinforcement learning to drive strategy iteration and co-optimization;
- **Reasoning** — Leverages SPARQL-style Chain-of-Thought (CoT) for multi-hop variable binding reasoning over structured triples;
- **Memory** — Integrates [OpenClaw / Letta](https://github.com/letta-ai/letta) tiered memory architecture (hot / warm / cold), enabling unbounded conversation length.

<!-- TODO: Replace with demo video -->
<!-- https://github.com/user-attachments/assets/YOUR_VIDEO_ID -->

<div align="center">
<img src="docs/assets/pipeline.png" alt="DocThinker Pipeline" width="820" />
<p><b>Figure 1.</b> DocThinker end-to-end pipeline — from document input to knowledge graph construction, tiered memory management, hybrid retrieval & reasoning, and output with feedback back to the graph.</p>
</div>

### ✨ Highlights

- **Two-path KG self-expansion** — Path A: HDBSCAN clusters entity embeddings and expands by cluster themes; Path B: takes the top-50 highest-degree original nodes and expands across 6 dimensions (hierarchy, causation, analogy, contrast, temporal, application), all candidates validated by LLM (score < 0.6 rejected) and semantically deduplicated
- **Self-evolving knowledge graph** — Expanded nodes enter the graph as candidates; after being adopted in user query answers, they gradually promote to formal nodes (use_count ≥ 2, score ≥ 1.2), while unadopted nodes decay; meanwhile, background sliding-window scanning automatically infers 6 types of latent edges to complete the graph
- **Multi-Agent co-evolution** — Splits retrieval, extraction, and answering into three collaborative Agents, driven by reinforcement learning through reward signals (answer quality, retrieval hit rate) for end-to-end co-optimization
- **Tiered conversation memory (Claw)** — OpenClaw / Letta-inspired three-layer memory architecture: hot layer (recent 6 turns), warm layer (LLM-compressed MEMORY.md), cold layer (historical conversation vector index with Top-k similarity retrieval), enabling unbounded conversation length
- **SPARQL Chain-of-Thought reasoning** — Guides LLM via prompts to decompose complex queries into SPARQL-style triple-pattern chains (`?variable, relation, ?variable/entity`), progressively binding variables and reasoning within KG context



---

## 🧬 Key Contributions

### 1. 🔀 Two-Path KG Self-Expansion

Expansion operates in two complementary passes:

| Path | Strategy | Grounding |
|------|----------|-----------|
| **A — Cluster-based** | HDBSCAN clusters entity embeddings → LLM generates cluster summaries → expands new entities grounded in cluster themes | Density structure |
| **B — Top-N multi-angle** | Top-50 highest-degree nodes expanded across 6 cognitive dimensions (hierarchy, causation, analogy, contrast, temporal, application) | Graph topology |

All candidates pass through **LLM self-validation** (factuality, non-redundancy, edge validity, specificity scoring) and **semantic deduplication** before admission.

### 2. 🔄 Self-Evolving Knowledge Graph

Newly expanded nodes do not immediately become authoritative knowledge — they enter the graph as `candidates`. Only when users repeatedly adopt a node in actual conversations do its usage count and score accumulate; once thresholds are met, the node is promoted to a formal part of the graph. Nodes that remain unused gradually decay and are phased out.

Additionally, single-pass extraction inevitably misses implicit cross-paragraph relationships. The system runs a background sliding-window scan over existing entities, having the LLM infer missing edges across six categories — hierarchical, causal, contrastive, temporal, application, and collaborative. After deduplication and validation, discovered edges are written to the graph and rendered as dashed red lines to distinguish them from original edges.

### 3. 🤖 Multi-Agent Co-Evolution

<div align="center">
<img src="docs/assets/multi_agent_evolution.png" alt="Multi-Agent Co-Evolution Architecture" width="820" />
<p><sub><b>Figure 2.</b> DocThinker multi-Agent co-evolution architecture — Retrieval, Extraction, and Answering Agents collaborate around the self-evolving knowledge graph and tiered memory, continuously optimizing through reinforcement learning feedback loops.</sub></p>
</div>

DocThinker splits the traditional RAG monolithic pipeline into three specialized Agents:

| Agent | Responsibility | Optimization Target |
|-------|---------------|-------------------|
| **Retrieval Agent** | Receives queries, retrieves relevant entities and document fragments from KG and vector store | Retrieval hit rate ↑ |
| **Extraction Agent** | Extracts entities and relations from retrieved results, constructs structured knowledge and writes to the graph | Extraction coverage ↑ |
| **Answering Agent** | Generates final answers based on SPARQL CoT reasoning, triggers node promotion/decay feedback | Answer quality ↑ |

The three Agents are modeled as a **Sequential Markov Decision Process (Sequential MDP)**, where each Agent's output serves as the next Agent's state input. Reinforcement learning aligns **local rewards** (each Agent's own metrics) with **global rewards** (end-to-end answer quality) through ranking consistency, driving joint strategy iteration and avoiding local optima at the expense of overall performance.

### 4. 🗃️ Tiered Conversation Memory (Claw)

Inspired by the [OpenClaw / MemGPT / Letta](https://github.com/letta-ai/letta) architecture, Claw implements a **three-layer memory hierarchy**:

| Layer | Temperature | Mechanism | Injection |
|-------|------------|-----------|-----------|
| **Hot** — Working Memory | Immediate | Recent *N* conversation turns | Always prepended |
| **Warm** — Core Memory | Session | LLM-compressed `MEMORY.md` (user preferences, facts, instructions) | Always prepended |
| **Cold** — Semantic Archive | Long-term | Older turns chunked, embedded, and vector-indexed | Top-*k* retrieval per query |

After each Q&A turn, older conversations are automatically archived to the cold layer, and the warm layer is periodically re-compressed by the LLM — enabling **unbounded conversation length** without context window overflow.

### 5. 🧠 SPARQL Chain-of-Thought (CoT) Reasoning

<div align="center">
<img src="docs/assets/sparql_cot.png" alt="SPARQL CoT Reasoning" width="680" />
<p><sub><b>Figure 3.</b> SPARQL Chain-of-Thought reasoning pipeline — queries are decomposed into triple-pattern chains with variable binding against KG context.</sub></p>
</div>

Complex queries are internally decomposed into **SPARQL-style triple-pattern chains** before answer generation. The LLM binds variables against KG context via shared-variable chaining, constructs a variable binding table, then synthesizes the final answer. This replaces unstructured "find relevant info" with **systematic graph traversal reasoning**.

---

## 💡 Use Cases

<table>
<tr>
<td width="50%" valign="top">

> *"Upload a novel and explore its knowledge graph"*

<img src="docs/assets/usecase_kg.gif" width="100%"/>

</td>
<td width="50%" valign="top">

> *"Deep-mode conversation with SPARQL CoT reasoning and tiered memory"*

<img src="docs/assets/usecase_chat.gif" width="100%"/>

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
git clone https://github.com/Yang-Jiashu/doc-thinker.git && cd doc-thinker
conda create -n docthinker python=3.11 -y && conda activate docthinker
pip install -r requirements.txt && pip install -e .
cp env.example .env   # ← fill in API keys (OpenAI / DashScope / SiliconFlow)
```

**Launch:**

```bash
# Terminal 1 — Backend (FastAPI)
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend (Flask UI)
python run_ui.py
```

Open `http://localhost:5000` — upload a PDF, ask questions, and explore the evolving knowledge graph.

---

## ⚡ Query Modes

| Mode | Strategy | Latency | Depth |
|------|----------|---------|-------|
| **Fast** | Vector similarity | ~1 s | Shallow |
| **Standard** | Hybrid KG + vector + reranking | ~3 s | Medium |
| **Deep** | SPARQL CoT + spreading activation + episodic memory + expansion matching + post-query feedback | ~8 s | Full |

<details>
<summary><b>Deep Mode Pipeline (7 steps)</b></summary>

1. Retrieve analogous episodes from episodic memory via spreading activation.
2. Match expanded candidate nodes against the query (token-overlap + embedding).
3. Inject matched expansions as forced retrieval instructions.
4. Decompose query into SPARQL CoT triple-pattern chain.
5. Hybrid KG + vector retrieval with spreading activation.
6. LLM generates answer with full context and variable binding.
7. Post-query feedback: validate expanded nodes, store episode, co-activate links, update Claw memory layers.

</details>

## 📄 PDF Processing

| Mode | Engine | Best for |
|------|--------|----------|
| `auto` (default) | VLM (short) / MinerU (long) | General use |
| `vlm` | Cloud VLM (Qwen-VL) | Image-heavy documents |
| `mineru` | MinerU layout engine | Long documents with complex tables |

<details>
<summary><b>📡 API Reference</b></summary>

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Sessions | `/sessions` | GET / POST | List / create sessions |
| | `/sessions/{id}/history` | GET | Chat history |
| | `/sessions/{id}/files` | GET | Ingested files |
| Ingest | `/ingest` | POST | Upload PDF / TXT |
| | `/ingest/stream` | POST | Stream raw text |
| Query | `/query/stream` | POST | SSE streaming query |
| | `/query` | POST | Non-streaming query |
| KG | `/knowledge-graph/data` | GET | Nodes + edges for visualization |
| | `/knowledge-graph/expand` | POST | Trigger 2-path expansion |
| | `/knowledge-graph/stats` | GET | KG statistics |
| Memory | `/memory/stats` | GET | Episode + Claw memory stats |
| | `/memory/consolidate` | POST | Run episodic consolidation |
| Settings | `/settings` | GET / POST | Runtime config |

</details>

<details>
<summary><b>📂 Project Structure</b></summary>

| Directory | Description |
|-----------|-------------|
| `docthinker/` | Core: PDF parsing, KG construction, query routing, 2-path expansion (`kg_expansion/`), auto-thinking (`auto_thinking/`), HyperGraphRAG (`hypergraph/`), server (`server/`), UI (`ui/`). |
| `graphcore/` | Graph RAG engine: KG storage (NetworkX / FAISS / Qdrant / PG), SPARQL CoT prompting, entity extraction, reranking. |
| `neuro_memory/` | Episodic memory: spreading activation, episode store, analogical retrieval, consolidation. |
| `claw/` | Tiered memory: hot (working), warm (core / MEMORY.md), cold (semantic archive). |
| `config/` | `settings.yaml` — PDF, memory, retrieval, cognition parameters. |

</details>

---

## 📝 Citation

If you find DocThinker useful in your research, please cite:

```bibtex
@article{docthinker2026,
  title={DocThinker: Self-Evolving Knowledge Graphs with Tiered Memory and Structured Reasoning for Document Understanding},
  author={Yang, Jiashu},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

## 🤝 Contributing

PRs and issues welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📜 License

[MIT](LICENSE)
