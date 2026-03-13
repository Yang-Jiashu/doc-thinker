<div align="center">

<img src="logo.png" alt="DocThinker Logo" width="220" />

# DocThinker

**An Intelligent Personal Knowledge Assistant with Brain-Like Memory Architecture**
**具有类人脑记忆架构的智能个人知识助手**

*Beyond traditional retrieval — think and remember like a human.*
*超越传统检索，像人类一样思考与记忆*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-success)]()

[English](#-overview) | [中文](#-项目简介)

</div>

---

<details open>
<summary><h2>English</h2></summary>

## 📖 Overview

**DocThinker** is a next-generation intelligent agent system designed to break through the limitations of traditional RAG (Retrieval-Augmented Generation). Instead of simply searching for similar text chunks, it builds a **structured, brain-like memory system**.

In this system, knowledge is stored as interconnected Episodes, Concepts, and Entities. Like the human brain, it can understand complex documents, build knowledge graphs, and proactively associate related information.

## ✨ Key Features

### 1. KG-Based Memory Architecture
DocThinker treats the Knowledge Graph (KG) as memory itself, not just an external database.
- **Episode Nodes**: Every interaction (document reading, conversation, event) becomes a node in the graph.
- **Entity & Concept**: Automatically extracted and linked to related episodes.
- **Unified Storage**: Structured relationships and vector embeddings fused within the same graph.

### 2. Auto-Association Mechanism
The system doesn't just passively wait for queries — it thinks proactively.
- **On-Insert Association**: When new knowledge is written, the system automatically discovers and connects to existing related memories.
- **Spreading Activation**: The retrieval process simulates human cognition by activating concept nodes and spreading outward to find latent connections.
- **Spontaneous Recall**: Proactively surfaces related memories based on context without explicit search.

### 3. Multimodal Perception
Powered by **MinerU** and **Docling**, DocThinker deeply understands documents.
- **Deep Parsing**: Precise recognition of PDF layouts, tables, formulas, and images.
- **Hierarchical Structure Preservation**: Fully retains the document's logical structure (Chapter → Section → Paragraph), beyond mere chunking.

## 🏗 Architecture

The system mirrors the human cognitive process: **Perception → Cognition → Memory → Application**.

```mermaid
graph TD
    User[User / Document] --> Perception
    subgraph Perception [Perception Layer]
        Doc[Document Parser]
        Chat[Dialog Understanding]
    end
    
    Perception --> Cognition
    subgraph Cognition [Cognition Layer]
        Intent[Intent Recognition]
        Reason[Reasoning & Planning]
    end
    
    Cognition <--> Memory
    subgraph Memory [Memory Core]
        WM[Working Memory]
        KG[KG Memory]
        Vector[Vector Memory]
        
        KG -- Spreading Activation --> WM
        Vector -- Semantic Retrieval --> WM
    end
    
    Memory --> Application
    subgraph Application [Application Layer]
        QA[QA System]
        Vis[Memory Visualization]
        Agent[Agent Actions]
    end
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker

# 2. Create and activate a Conda environment
conda create -n docthinker python=3.11 -y
conda activate docthinker

# 3. Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Configuration
Copy the example config and fill in your API key (supports OpenAI, DashScope/Qwen, SiliconFlow, etc.):

```bash
cp env.example .env
# Edit .env and fill in your API key
```

### Run

**Start interactive chat:**
```bash
python main.py
```

**Start Web UI (recommended):**
```bash
# Terminal 1: Start FastAPI backend
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Flask UI
python run_ui.py
```

**Start API server only:**
```bash
python main.py --server
```

## 🔍 Query Modes

DocThinker offers three query modes for different use cases:

| Mode | Retrieval Strategy | Use Case | Description |
|------|-------------------|----------|-------------|
| ⚡ **Fast** | Direct vector matching (Naive) | Simple fact queries, quick Q&A | Skips spreading activation and deep reasoning; directly matches text chunks and nodes for fastest response. |
| ⚖️ **Standard** | Hybrid retrieval | Everyday use (default) | Combines vector retrieval with KG structured queries, balancing speed and answer quality. |
| 🧠 **Deep** | Hybrid + Spreading Activation (Mix + Thinking) | Complex analysis, cross-document reasoning | Enables **spreading activation** along KG edges to uncover latent connections, plus multi-step reasoning for deeper but slower answers. |

## 📄 PDF Processing

PDF parsing mode is configured via `config/settings.yaml`:

```yaml
pdf:
  parse_mode: "auto"     # "auto" | "vlm" | "mineru"
  page_threshold: 15     # Only effective in auto mode
```

| Mode | Description |
|------|-------------|
| `auto` (default) | Auto-routing: pages ≤ `page_threshold` use cloud VLM recognition; pages > `page_threshold` use MinerU framework for OCR extraction. |
| `vlm` | Force cloud VLM (Vision Language Model) for all PDFs. Best for short documents or when image content understanding is needed. |
| `mineru` | Force MinerU framework for all PDFs. Best for long documents with precise structure, table, and layout extraction. |

> Override YAML config via environment variables `PDF_PARSE_MODE` and `PDF_VLM_PAGE_THRESHOLD`.

## 🌐 Knowledge Graph Expansion

On the KG visualization page, click the **"Expand"** button in the top-right corner to trigger **LLM associative expansion**:

- **Function**: The system invokes the LLM to reason about core entities from multiple perspectives (e.g., hypernyms/hyponyms, causal relations, application scenarios), generating new candidate nodes and relations that are automatically written into the KG.
- **Effect**: Expanded nodes appear in **yellow**, distinguishing them from original nodes (blue/green/pink). These nodes supplement knowledge logically related but not explicitly mentioned in the documents.
- **Usage**: In deep mode queries, expanded nodes participate in spreading activation and matching, helping the system understand broader context and improving cross-domain, cross-concept reasoning.

## 📂 Project Structure

| Directory | Description |
|-----------|-------------|
| `docthinker/` | **Core library**: document parsing (`parser`, `pdf_pipeline/`), KG construction (`processor`), query engine (`query`), cognitive processing (`cognitive/`), KG expansion (`kg_expansion/`), auto chain-of-thought (`auto_thinking/`), FastAPI backend (`server/`), Flask UI (`ui/`). |
| `graphcore/` | **Graph RAG engine**: CoreGraph-based KG storage, vector retrieval, LLM binding, entity/relation extraction. Used by docthinker as the underlying graph engine. |
| `neuro_memory/` | **Brain-like memory engine**: spreading activation, episode store, graph store, integrated with docthinker server. |
| `linearrag_module/` | **LinearRAG module** (optional): NER-based graph-free RAG strategy, retained as a switchable alternative interface. |
| `config/` | **Configuration**: `settings.yaml` centrally manages hyperparameters for PDF processing, memory system, retrieval, cognition, etc. |
| `scripts/` | **Utility scripts**: connectivity tests, PDF pipeline tests, config checks, and other helper scripts. |
| `tests/` | **Unit tests**: automated test cases for each module. |
| `docs/` | **Documentation**: system flow descriptions, storage migration guides, security checks, etc. |
| `data/` | **Runtime data**: session data, knowledge graphs, vector stores, multimodal image assets (auto-generated at runtime, not version-controlled). |

## 🤝 Contributing

Pull requests and issues are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

</details>

---

<details open>
<summary><h2>中文</h2></summary>

## 📖 项目简介

**DocThinker** 是下一代智能体系统，旨在突破传统 RAG（检索增强生成）的局限。它不再仅仅是搜索相似的文本片段，而是构建了一个**结构化、类人脑的记忆系统**。

在这个系统中，知识被存储为相互连接的情节（Episodes）、概念（Concepts）和实体（Entities）。它能够像人类大脑一样，理解复杂的文档，建立知识图谱，并主动联想相关信息。

---

## ✨ 核心特性

### 1. 🧠 基于 KG 的记忆架构
DocThinker 将知识图谱（KG）视为记忆本身，而非仅仅是外挂数据库。
- **情节节点 (Episode Nodes)**：每一次交互（文档阅读、对话、事件）都作为图中的一个节点。
- **实体与概念 (Entity & Concept)**：自动提取并链接到相关情节。
- **统一存储**：结构化关系与向量嵌入（Embedding）融合在同一个图谱中。

### 2. 🔗 自动联想机制 (Auto-Association)
系统不仅被动等待查询，更具备主动思考能力。
- **写入时联想 (On-Insert Association)**：新知识写入时，自动发现并连接到已有的相关记忆。
- **扩散激活 (Spreading Activation)**：检索过程模拟人类思维，通过激活概念节点并向四周扩散，寻找潜在关联。
- **自发回忆 (Spontaneous Recall)**：根据上下文主动浮现相关记忆，无需显式搜索。

### 3. 📄 多模态感知 (Multimodal Perception)
基于 **MinerU** 和 **Docling** 强大的解析能力，DocThinker 能够深度理解文档。
- **深度解析**：精准识别 PDF 布局、表格、公式和图片。
- **层级结构保持**：完整保留文档的逻辑结构（章 -> 节 -> 段落），不仅仅是切片。

---

## 🏗 架构设计

系统模仿人类认知过程：**感知 (Perception) → 认知 (Cognition) → 记忆 (Memory) → 应用 (Application)**。

```mermaid
graph TD
    User[用户 / 文档] --> Perception
    subgraph Perception [感知层 Perception]
        Doc[文档解析器]
        Chat[对话理解]
    end
    
    Perception --> Cognition
    subgraph Cognition [认知层 Cognition]
        Intent[意图识别]
        Reason[推理与规划]
    end
    
    Cognition <--> Memory
    subgraph Memory [记忆核心 Memory Core]
        WM[工作记忆]
        KG[知识图谱记忆]
        Vector[向量记忆]
        
        KG -- 扩散激活 --> WM
        Vector -- 语义检索 --> WM
    end
    
    Memory --> Application
    subgraph Application [应用层 Application]
        QA[问答系统]
        Vis[记忆可视化]
        Agent[智能体行动]
    end
```

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker

# 2. 创建并激活 Conda 虚拟环境
conda create -n docthinker python=3.11 -y
conda activate docthinker

# 3. 安装依赖
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 配置
复制示例配置文件并填入你的 API Key（支持 OpenAI, DashScope/千问, SiliconFlow 等）：

```bash
cp env.example .env
# 编辑 .env 文件填入 API Key
```

### 运行

**启动交互式对话：**
```bash
python main.py
```

**启动 Web UI（推荐）：**
```bash
# 终端 1：启动 FastAPI 后端
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# 终端 2：启动 Flask UI
python run_ui.py
```

**仅启动 API 服务：**
```bash
python main.py --server
```

---

## 🔍 查询模式说明

DocThinker 提供三种查询模式，适用于不同场景：

| 模式 | 检索策略 | 适用场景 | 说明 |
|------|---------|---------|------|
| ⚡ **快速模式** | 直接向量匹配（Naive） | 简单事实查询、快速问答 | 跳过扩散激活与深度推理，直接匹配文本块和节点，响应最快。 |
| ⚖️ **标准模式** | 混合检索（Hybrid） | 日常使用（默认） | 结合向量检索与知识图谱结构化查询，平衡速度与答案质量。 |
| 🧠 **深度模式** | 混合检索 + 扩散激活（Mix + Thinking） | 复杂分析、跨文档推理 | 在标准模式基础上启用**扩散激活**（Spreading Activation），沿知识图谱边向外扩散，发掘隐含关联；同时启用多步推理，答案更深入但耗时较长。 |

---

## 📄 PDF 处理配置

PDF 解析模式通过 `config/settings.yaml` 配置：

```yaml
pdf:
  parse_mode: "auto"     # "auto" | "vlm" | "mineru"
  page_threshold: 15     # 仅 auto 模式生效
```

| 模式 | 说明 |
|------|------|
| `auto`（默认） | 自动路由：页数 ≤ `page_threshold` 使用云端 VLM 识别，页数 > `page_threshold` 使用 MinerU 框架进行 OCR 提取。 |
| `vlm` | 强制使用云端 VLM（视觉语言模型）处理所有 PDF，适合短文档或需要图片内容理解的场景。 |
| `mineru` | 强制使用 MinerU 框架处理所有 PDF，适合长文档，可精准提取文档结构、表格和布局信息。 |

> 也可通过环境变量 `PDF_PARSE_MODE` 和 `PDF_VLM_PAGE_THRESHOLD` 覆盖 YAML 配置。

---

## 🌐 知识图谱扩展

在知识图谱可视化页面，点击右上角的 **「扩展」** 按钮可以触发 **LLM 联想扩展**：

- **功能**：系统调用大语言模型，从多个角度（如上下位概念、因果关系、应用场景等）对图谱中已有的核心实体进行联想推理，生成新的候选节点和关系，并自动写入知识图谱。
- **效果**：扩展后的新节点以 **黄色** 显示，与原有节点（蓝/绿/粉色）区分。这些扩展节点补充了文档中未显式提及但逻辑上相关的知识。
- **用途**：在深度模式查询时，扩展节点也会参与扩散激活和匹配，有助于系统理解更广泛的上下文，提升跨领域、跨概念的推理能力。

---

## 📂 项目结构

| 目录 | 说明 |
|------|------|
| `docthinker/` | **核心主库**：文档解析（`parser`、`pdf_pipeline/`）、知识图谱构建（`processor`）、查询引擎（`query`）、认知处理（`cognitive/`）、KG 扩展（`kg_expansion/`）、自动思维链（`auto_thinking/`）、FastAPI 后端（`server/`）、Flask UI（`ui/`）。 |
| `graphcore/` | **图 RAG 引擎**：基于 CoreGraph 的知识图谱存储、向量检索、LLM 绑定、实体/关系提取。被 docthinker 作为底层图引擎调用。 |
| `neuro_memory/` | **类脑记忆引擎**：扩散激活（Spreading Activation）、情节存储（Episode Store）、图存储（Graph Store），与 docthinker 服务端集成。 |
| `linearrag_module/` | **LinearRAG 模块**（可选）：基于 NER 的无关系图 RAG 策略，作为可切换的备选接口保留。 |
| `config/` | **配置文件**：`settings.yaml` 集中管理 PDF 处理、记忆系统、检索、认知等超参数。 |
| `scripts/` | **工具脚本**：连通性测试、PDF 管线测试、配置检查等辅助脚本。 |
| `tests/` | **单元测试**：各模块的自动化测试用例。 |
| `docs/` | **项目文档**：系统流程说明、存储迁移指南、安全检查等。 |
| `data/` | **运行时数据**：会话数据、知识图谱、向量库、多模态图片资产（运行时自动生成，不纳入版本控制）。 |

---

## 🤝 贡献指南

欢迎提交 Pull Request 或 Issue！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 开源协议

本项目采用 [MIT 协议](LICENSE) 开源。

</details>
