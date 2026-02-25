# 从文档到各种 RAG 的代码与文档总览

本文档按「文档 → 解析 → 入库 → 检索/问答」与「各类 RAG 职责」两条线，把仓库里的文档和代码串起来，便于通读和排查。

---

## 一、文档与设计（先看什么）

| 文档 | 路径 | 内容概要 |
|------|------|----------|
| 主 README | `README.md` | 仓库定位、环境安装、数据目录约定 |
| 项目结构 | `docs/PROJECT_STRUCTURE.md` | UI/后端入口、模板目录、知识图谱入口、启动顺序 |
| 知识图谱优化 | `docs/KG_OPTIMIZATIONS.md` | 主 KG 与记忆图的 UI/构建/变更/合并/查询/存储：已做与可做优化清单 |
| 类脑记忆设计 | `neuro_memory/DESIGN.md` | 扩散激活、巩固重放、结构映射与类比、显著性、图式、区分；算法模块与数据结构 |
| 类脑记忆验证 | `neuro_memory/VERIFY.md` | 本地脚本验证、完整服务验证、自检清单 |
| （内部模块） | （不公开） | 线性图与 NER 模块不在开源仓库暴露 |

---

## 二、配置与全局状态（入口级）

| 文件 | 作用 |
|------|------|
| `核心库/config.py` | 主配置类：工作目录、解析器、多模态开关、auto-thinking、图构建模式（llm/linear）、spacy 等 |
| `核心库/api_config.py` | `APIConfig` / `APIRoutes`：API 前缀、CORS、各类 API 开关与路由名 |
| `核心库/server/state.py` | `AppState`：单例 state，挂载 settings、session_manager、rag_instance、cognitive_processor、ingestion_service、orchestrator、**memory_engine** |
| `.env` / `env.example` | 环境变量：WORKING_DIR、解析/LLM/Embedding/Rerank、可选 TIKTOKEN_CACHE_DIR 等 |

---

## 三、文档解析与多模态处理（文档 → 结构化内容）

- **解析层**  
  - `核心库/parser.py`：MineruParser、DoclingParser，产出带 `content_list` 的解析结果（文本/图/表/公式块）。  
  - 输出目录由 `config.parser_output_dir` 决定，脚本常用 `data/mineru_output/` 或 `output/`。

- **多模态处理**  
  - `核心库/processor.py`、`核心库/modalprocessors.py`：ProcessorMixin、Image/Table/Equation/Generic 处理器、ContextExtractor。  
  - 为图片/表格/公式生成描述或结构化文本，供后续入库与检索。

- **主流水线入口**  
  - 核心库主入口模块：QueryMixin + ProcessorMixin + BatchMixin，解析与入库入口。  
  - 解析 → 多模态处理 → 插入 AutoThink 图引擎（及可选 HyperGraph 同步），维护 `knowledge_graph`、`hyper_chunk_sink` 等。

---

## 四、入库与存储（结构化内容 → 索引与图）

- **会话与入库服务**  
  - `核心库/session_manager.py`：会话创建、会话级 RAG 实例、消息历史、文件列表。  
  - `核心库/services/ingestion_service.py`：`IngestionService`，对全局 RAG 和会话 RAG 做 `ingest_text` / `ingest_folder` / `ingest_files`，内部调 `图引擎.ainsert` 或 `insert`。

- **后端入库 API**  
  - `核心库/server/routers/ingest.py`：  
    - 文件上传、MinerU `content_list` 的加载与合并、`_process_text_for_ingest`（CognitiveProcessor 认知分析）。  
    - 写入全局/会话 RAG；若有 `state.memory_engine`，则用认知结果调用 `memory_engine.add_observation`（即写入类脑记忆）。

- **主 RAG 存储（AutoThink 图引擎）**  
  - 由核心库主入口模块初始化 AutoThink 图引擎，使用 `config.working_dir` 下 KV/向量/图存储。  
  - 实体与关系：`核心库/knowledge_graph.py`（KnowledgeGraph），与 AutoThink 图引擎 的 chunk_entity_relation_graph 协同；消歧、去重、MinHash 等见 `docs/KG_OPTIMIZATIONS.md`。

- **AutoThink 超图（可选）**  
  - `核心库/hypergraph/hypergraphrag.py`：chunking、实体/关系抽取、图与向量存储。  
  - `核心库/hypergraph/operate.py`：chunking_by_token_size、extract_entities、kg_query 等。  
  - 与主 RAG 的同步由 **Auto-Thinking 编排器** 的 `hyper_chunk_sink`、`sync_mode` 等控制（见下）。

- **类脑记忆（neuro_memory）**  
  - `neuro_memory/engine.py`：`MemoryEngine`，`add_observation`（写 Episode + 即时联想）、`consolidate`（巩固）、`retrieve_analogies`（类比检索）。  
  - `neuro_memory/models.py`：Episode、EdgeType、MemoryEdge。  
  - `neuro_memory/graph_store.py`：MemoryGraphStore。  
  - `neuro_memory/episode_store.py`、`spreading_activation.py`、`consolidation.py`、`analogical_retrieval.py`：存储、扩散激活、巩固、类比检索。  
  - 在服务中由 `核心库/server/app.py` 的 lifespan 初始化并挂到 `state.memory_engine`；写入来自 ingest 与 query 的认知结果（见上 ingest、下 query）。

---

## 五、查询与问答（请求 → 回答）

- **查询 API 入口**  
  - `核心库/server/routers/query.py`：`POST /query`。  
  - 流程概要：  
    1. 身份/打招呼类短问 → 直接 LLM 回复。  
    2. 有 session 且可做快答时 → `_try_fast_qa`（单文件、小文档 + VLM 看图/短文本）。  
    3. 启用 thinking 且有 `state.orchestrator` → 走 **Auto-Thinking**；若存在 `state.memory_engine`，先 `retrieve_analogies`，将「类比记忆」拼入 context_prefix，再交给编排器。  
    4. 否则 → 直接调 `state.rag_instance.query`（或等价）。  
  - 每轮对话结束后 `_ingest_chat_turn`：认知分析 → `add_cognitive_memory`、`memory_engine.add_observation`、`ingestion_service.ingest_text`。

- **主 RAG 查询逻辑**  
  - `核心库/query.py`：QueryMixin，文本/多模态查询、缓存、Prompt、检索与生成。  
  - 内部用 AutoThink 图引擎 的检索与生成接口；可选 rerank、多步检索等。

- **Auto-Thinking 编排器（复杂问题）**  
  - `核心库/auto_thinking/orchestrator.py`：`HybridRAGOrchestrator`。  
  - 职责：问题复杂度分类、子问题拆解、在 **Doc Thinker 主入口** 与 **AutoThink 超图** 之间路由、同步 chunk（`hyper_chunk_sink`、eager/lazy）、子答案聚合。  
  - 依赖：`classifier`、`decomposer`、`vlm_client`、`rag`、`hyper`；若启用 hyper，则 `rag.hyper_chunk_sink` 指向 `_collect_hyper_chunks`。

- **认知处理（为记忆与入库提供摘要/实体/关系）**  
  - `核心库/cognitive/processor.py`：CognitiveProcessor。  
  - 对一段文本做理解与关联，返回 CognitiveInsight（summary、key_points、concepts、entities、relations、inferred_relations 等），供 `add_cognitive_memory` 和 `memory_engine.add_observation` 使用。

---

## 六、图谱与记忆相关 API（前端/可视化用）

- **主知识图谱**  
  - `核心库/server/routers/graph.py`：  
    - `GET /knowledge-graph/data`：按 session 或全局取 AutoThink 图引擎 的 chunk_entity_relation_graph，返回节点/边。  
    - `GET /knowledge-graph/stats`、`POST /knowledge-graph/entity`、`POST /knowledge-graph/relationship`、`PUT /knowledge-graph/entity/{name}`、`DELETE /knowledge-graph/relationship`：统计与增删改。

- **类脑记忆图谱**  
  - 同文件：  
    - `GET /memory/stats`：episodes 数、边数。  
    - `GET /memory/graph-data`：记忆图节点/边（供前端与「知识图谱」页「记忆图谱」数据源共用）。  
    - `POST /memory/consolidate`：触发一次巩固。

---

## 七、RAG 种类与职责对照

| 模块/类 | 角色 | 主要入口/调用点 |
|---------|------|----------------|
| **AutoThink 图引擎**（通过 Doc Thinker 主入口） | 主检索与生成；chunk/实体/关系图 | `state.rag_instance`，ingest 时 ainsert/insert，query 时 query；graph 路由读其 chunk_entity_relation_graph |
| **AutoThink 超图** | 复杂问题子检索、图构建（chunk+实体+关系） | `state.orchestrator.hyper`；编排器路由与同步；config 中 `graph_construction_mode` 可配 llm/linear |
| **neuro_memory.MemoryEngine** | 类脑联想与记忆：Episode、扩散激活、巩固、类比检索 | `state.memory_engine`；ingest 与 query 后写 observation；query 前可选 retrieve_analogies；graph 路由读/巩固 |
| **CognitiveProcessor** | 文本理解与关联，产出 CognitiveInsight | `state.cognitive_processor`；ingest 与 _ingest_chat_turn 中 process；结果喂给 add_cognitive_memory 与 memory_engine |
| **HybridRAGOrchestrator** | 复杂问句拆解、RAG/Hyper 路由、子答案聚合 | `state.orchestrator`；query 路由在 enable_thinking 时使用 |
| **KnowledgeGraph**（核心库） | 实体/关系管理（与 AutoThink 图引擎 图协同） | `state.rag_instance.knowledge_graph`；graph 路由的 stats/entity/relationship |

---

## 八、数据流简图（文档到回答）

```
文档/对话
  → 解析(parser) / 多模态(processor)
  → Doc Thinker 主入口.process_document_complete / ingest_text
  → AutoThink 图引擎.insert / ainsert  (+ 可选 HyperGraph 同步)
  → CognitiveProcessor.process → add_cognitive_memory + memory_engine.add_observation

用户问题
  → POST /query
  → [身份/快答] 或 [Orchestrator：memory_engine.retrieve_analogies + 子问题 → RAG/Hyper 检索 → 聚合]
  → 或 直接 rag_instance.query
  → _ingest_chat_turn（写认知记忆 + 记忆引擎 + ingest_text）
```

---

以上覆盖「从文档拆解到各种 RAG 代码」的阅读路径；要查具体行为时，可按上表定位到对应文件与路由再细看实现。
