<div align="center">

<img src="docs/assets/banner.png" alt="DocThinker Banner" width="820" />

# DocThinker

**自进化知识图谱 · 长短期记忆 · 结构化推理**

*语言记录了认知过程的结果，而认知过程包含感知，经验，推理的过程。*

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

[快速开始](#-快速开始) · [核心贡献](#-核心贡献) · [系统管线](#-项目简介) · [使用场景](#-使用场景) · [API 参考](#api-参考)

</div>

## 📖 项目简介

**DocThinker** 是一个文档驱动的 RAG 系统，从上传文档中构建自进化的知识图谱。与传统"检索-LLM 回复"路线不同，DocThinker 将知识视为**动态的图谱**，实现类脑记忆：

- **生长** — 不局限于摄入文档中的显式知识，能自主推理并延伸出文档未直接表述的隐含关联以及相关知识；
- **进化** — 将检索、抽取、回答拆分为三个协作 Agent，通过强化学习驱动各 Agent 的策略迭代与协同优化；
- **推理** — 借助 SPARQL 风格的链式思维（CoT），在结构化三元组上进行多跳变量绑定推理；
- **记忆** — 集成 [OpenClaw / Letta](https://github.com/letta-ai/letta) 分层记忆架构（热/温/冷），实现无限对话长度。

<!-- TODO: 替换为演示视频 -->
<!-- https://github.com/user-attachments/assets/YOUR_VIDEO_ID -->

<div align="center">
<img src="docs/assets/pipeline.png" alt="DocThinker Pipeline" width="820" />
<p><b>图 1.</b> DocThinker 端到端管线 — 从文档输入到知识图谱构建、分层记忆管理、混合检索推理，最终输出并反馈回图谱。</p>
</div>

### ✨ 核心亮点

- **双路径 KG 自扩展** — 路径 A：HDBSCAN 聚类实体 embedding 并按聚类主题扩展；路径 B：取度数最高的 50 个原始节点，从层级、因果、类比、对立、时序、应用六个维度扩展，所有候选经 LLM 验证（score < 0.6 拒绝）和语义去重
- **自进化知识图谱** — 扩展节点以候选状态入图，经用户查询采纳后逐步晋升为正式节点（use_count ≥ 2, score ≥ 1.2），未采纳则衰减；同时后台以滑动窗口扫描实体，自动推断层级、因果、对比等六类潜在边并补全图谱
- **多 Agent 协同进化** — 将检索、抽取、回答拆分为三个协作 Agent，以强化学习驱动各 Agent 的迭代，通过奖励信号（回答质量、检索命中率）实现端到端的协同优化
- **分层对话记忆（Claw）** — 受 OpenClaw / Letta 启发的三层记忆架构：热层（最近 6 轮对话）、温层（LLM 压缩的 MEMORY.md）、冷层（历史对话向量索引，按相似度 Top-k 检索），实现无限对话长度
- **SPARQL 链式思维推理** — 通过提示词引导 LLM 将复杂查询分解为 SPARQL 风格的三元组模式链（`?变量, 关系, ?变量/实体`），在 KG 上下文中逐步绑定变量并推理



---

## 🧬 核心贡献

### 1. 🔀 双路径 KG 自扩展

扩展以两条互补路径执行：

| 路径 | 策略 | 锚定基础 |
|------|------|---------|
| **A — 聚类驱动** | HDBSCAN 聚类实体 embedding → LLM 生成聚类摘要 → 基于摘要主题扩展新实体 | 密度结构 |
| **B — Top-N 多角度** | 取连接度最高的 50 个节点，从 6 个认知维度（层级、因果、类比、对立、时序、应用）扩展 | 图拓扑 |

所有候选经过 **LLM 自验证**（事实性、非冗余性、边有效性、具体性评分）和**语义去重**后方可入图。

### 2. 🔄 自进化知识图谱

扩展的新节点不会直接成为正式知识——它们先以 `candidate` 身份进入图谱。只有当用户在实际对话中反复用到某个节点时，该节点的使用计数和评分才会累积，满足条件后晋升为正式节点；长期无人问津的节点则逐渐衰减淘汰。

此外，单次抽取难免遗漏跨段落的隐含关系。系统在后台用滑动窗口遍历已有实体，让 LLM 从层级、因果、对比、时序、应用、协作六个角度补充缺失的边，经去重校验后写入图谱，在可视化中以虚线与原始边区分。

### 3. 🤖 多 Agent 协同进化

<div align="center">
<img src="docs/assets/multi_agent_evolution.png" alt="多 Agent 协同进化架构" width="820" />
<p><sub><b>图 2.</b> DocThinker 多 Agent 协同进化架构 — 检索、抽取、回答三个 Agent 围绕自进化知识图谱与分层记忆协作，通过强化学习反馈闭环持续优化。</sub></p>
</div>

DocThinker 将传统 RAG 的单一管线拆分为三个专职 Agent：

| Agent | 职责 | 优化目标 |
|-------|------|---------|
| **Retrieval Agent** | 接收查询，从知识图谱与向量存储中检索相关实体和文档片段 | 检索命中率 ↑ |
| **Extraction Agent** | 从检索结果中抽取实体、关系，构建结构化知识并写入图谱 | 抽取覆盖率 ↑ |
| **Answering Agent** | 基于 SPARQL CoT 推理生成最终回答，并触发节点晋升/衰减反馈 | 回答质量 ↑ |

三个 Agent 以**序贯马尔可夫决策过程（Sequential MDP）**建模，前一个 Agent 的输出即为下一个 Agent 的状态输入。强化学习通过**局部奖励**（各 Agent 自身指标）与**全局奖励**（端到端回答质量）的排名一致性对齐，驱动各 Agent 策略的联合迭代，避免局部最优而整体退化。

### 4. 🗃️ 分层对话记忆（Claw）

受 [OpenClaw / MemGPT / Letta](https://github.com/letta-ai/letta) 架构启发，Claw 实现了**三层记忆层级**：

| 层级 | 温度 | 机制 | 注入方式 |
|------|------|------|---------|
| **热层** — 工作记忆 | 即时 | 最近 *N* 轮对话 | 始终注入 |
| **温层** — 核心记忆 | 会话级 | LLM 压缩的 `MEMORY.md`（用户偏好、事实、指令） | 始终注入 |
| **冷层** — 语义档案 | 长期 | 旧对话分块、嵌入、向量索引 | 按查询 Top-*k* 检索 |

每轮问答后，旧对话自动归档至冷层，温层由 LLM 定期重新压缩 — 实现**无限对话长度**而不溢出上下文窗口。

### 5. 🧠 SPARQL 链式思维（CoT）推理

<div align="center">
<img src="docs/assets/sparql_cot.png" alt="SPARQL CoT 推理" width="680" />
<p><sub><b>图 3.</b> SPARQL 链式思维推理管线 — 查询被分解为三元组模式链，在 KG 上下文中进行变量绑定。</sub></p>
</div>

复杂查询在回答前被内部分解为 **SPARQL 风格的三元组模式链**。LLM 通过共享变量链在 KG 上下文中绑定变量，构建变量绑定表，然后合成最终回答。这将非结构化的"查找相关信息"替换为**系统性图遍历推理**。



---

## 💡 使用场景

<table>
<tr>
<td width="50%" valign="top">

> *"上传小说，探索自动构建的知识图谱"*

<img src="docs/assets/usecase_kg.gif" width="100%"/>

</td>
<td width="50%" valign="top">

> *"深度模式对话 — SPARQL CoT 推理 + 分层记忆"*

<img src="docs/assets/usecase_chat.gif" width="100%"/>

</td>
</tr>
</table>

---

## 🚀 快速开始

```bash
git clone https://github.com/Yang-Jiashu/doc-thinker.git && cd doc-thinker
conda create -n docthinker python=3.11 -y && conda activate docthinker
pip install -r requirements.txt && pip install -e .
cp env.example .env   # ← 填入 API Key（OpenAI / DashScope / SiliconFlow）
```

**启动：**

```bash
# 终端 1 — 后端（FastAPI）
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# 终端 2 — 前端（Flask UI）
python run_ui.py
```

打开 `http://localhost:5000` — 上传 PDF，提问，探索自进化的知识图谱。

---

## ⚡ 查询模式

| 模式 | 策略 | 延迟 | 深度 |
|------|------|------|------|
| **快速** | 向量相似度 | ~1 s | 浅 |
| **标准** | 混合 KG + 向量 + 重排序 | ~3 s | 中 |
| **深度** | SPARQL CoT + 扩散激活 + 情节记忆 + 扩展匹配 + 查询后反馈 | ~8 s | 完整 |

<details>
<summary><b>深度模式管线（7 步）</b></summary>

1. 通过扩散激活从情节记忆中检索类比情节。
2. 将扩展候选节点与查询匹配（词重叠 + 嵌入相似度）。
3. 将命中的扩展节点作为强制检索指令注入。
4. 将查询分解为 SPARQL CoT 三元组模式链。
5. 混合 KG + 向量检索并启用扩散激活。
6. LLM 使用完整上下文和变量绑定生成回答。
7. 查询后反馈：验证扩展节点、存储情节、共激活链接、更新 Claw 记忆层。

</details>

## 📄 PDF 处理

| 模式 | 引擎 | 适用场景 |
|------|------|---------|
| `auto`（默认） | VLM（短文档）/ MinerU（长文档） | 通用 |
| `vlm` | 云端 VLM（Qwen-VL） | 图片密集文档 |
| `mineru` | MinerU 布局引擎 | 含复杂表格的长文档 |

<details>
<summary><b>📡 API 参考</b></summary>

| 类别 | 端点 | 方法 | 说明 |
|------|------|------|------|
| 会话 | `/sessions` | GET / POST | 列出 / 创建会话 |
| | `/sessions/{id}/history` | GET | 聊天历史 |
| | `/sessions/{id}/files` | GET | 已上传文件 |
| 上传 | `/ingest` | POST | 上传 PDF / TXT |
| | `/ingest/stream` | POST | 流式文本上传 |
| 查询 | `/query/stream` | POST | SSE 流式查询 |
| | `/query` | POST | 非流式查询 |
| KG | `/knowledge-graph/data` | GET | 可视化节点/边 |
| | `/knowledge-graph/expand` | POST | 触发双路径扩展 |
| | `/knowledge-graph/stats` | GET | KG 统计 |
| 记忆 | `/memory/stats` | GET | 情节 + Claw 记忆统计 |
| | `/memory/consolidate` | POST | 触发情节固化 |
| 设置 | `/settings` | GET / POST | 运行时配置 |

</details>

<details>
<summary><b>📂 项目结构</b></summary>

| 目录 | 说明 |
|------|------|
| `docthinker/` | 核心：PDF 解析、KG 构建、查询路由、双路径扩展（`kg_expansion/`）、自动思考（`auto_thinking/`）、HyperGraphRAG（`hypergraph/`）、服务端（`server/`）、UI（`ui/`）。 |
| `graphcore/` | 图 RAG 引擎：KG 存储（NetworkX / FAISS / Qdrant / PG）、SPARQL CoT 提示词、实体抽取、重排序。 |
| `neuro_memory/` | 情节记忆：扩散激活、情节存储、类比检索、记忆固化。 |
| `claw/` | 分层记忆：热层（工作记忆）、温层（核心 / MEMORY.md）、冷层（语义档案）。 |
| `config/` | `settings.yaml` — PDF、记忆、检索、认知参数。 |

</details>

---

## 📝 引用

如果 DocThinker 对您的研究有帮助，请引用：

```bibtex
@article{docthinker2026,
  title={DocThinker: Self-Evolving Knowledge Graphs with Tiered Memory and Structured Reasoning for Document Understanding},
  author={Yang, Jiashu},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

## 🤝 贡献

欢迎 PR 和 Issue！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📜 协议

[MIT](LICENSE)
