# 项目文件夹说明与整理建议

## 一、核心代码（必留）

| 文件夹 | 用途 | 说明 |
|--------|------|------|
| **核心库目录** | AutoThink 核心库与后端 | 解析、入库、查询、知识图谱、自动思考、超图、cognitive、UI 等，被 server 与脚本直接使用。（目录名见仓库根下。） |
| **neuro_memory/** | 类脑记忆引擎 | 与 AutoThink 核心对接，写入/巩固/类比检索；server 在 lifespan 中初始化并挂到 state.memory_engine。 |
| **scripts/** | 入库与脚本 | 如 verify_neuro_memory、ingest_mineru_academic 等，README 中说明的入口。 |
| **docs/** | 项目文档 | PROJECT_STRUCTURE、CODE_AND_DOCS_OVERVIEW、KG_OPTIMIZATIONS、FOLDERS 等。 |
| **tests/** | 单元/集成测试 | 正式测试用例，与根目录零散的 test_*.py 不同。 |

---

## 二、运行时与数据（按需保留）

| 文件夹 | 用途 | 建议 |
|--------|------|------|
| **rag_storage_api/** | 会话与 RAG 存储 | `session_manager` 默认 `base_storage_path`，会话、索引、向量/图存储在此。**保留**，可加入 .gitignore 若不想提交运行时数据。 |
| **data/** | 解析用数据 | README 约定 data/raw、data/mineru_output 等。按需保留，通常 .gitignore 已忽略或只提交样例。 |
| **output/** | 解析/生成输出 | 解析结果、部分脚本输出。可 .gitignore，按需保留本地。 |
| **neuro_memory_verify_data/** | 记忆验证脚本输出 | `scripts/verify_neuro_memory.py` 写入；可删除后重新跑脚本生成，建议加入 .gitignore。 |

---

## 三、依赖与第三方（二选一或按需）

| 文件夹 | 用途 | 建议 |
|--------|------|------|
| **图存储目录（可选）** | 图/向量存储实现 | 项目可通过 pip 依赖使用；若需本地修改可保留仓库内对应目录并 pip 安装为可编辑。 |
| （无公开子模块） | NER/线性图（可选） | 如需 NER/线性图实现，请在私有模块中对接，不在开源仓库暴露。 |

---

## 四、空目录与缓存（可清理）

| 文件夹 | 用途 | 建议 |
|--------|------|------|
| **templates/** | 已废弃 | 原为旧版 UI 模板，已删空；模板现统一在核心库的 `ui/templates/`。可删除此空目录。 |
| **.pytest_cache/** | pytest 缓存 | 已应在 .gitignore，可删除。 |
| **.ruff_cache/** | ruff 缓存 | 同上。 |
| **__pycache__/** | Python 字节码 | 同上。 |
| **test_llm_logic_cache/** | 某测试缓存 | 可删除或加入 .gitignore。 |
| **snapshots/** | 不明 | 若为版本控制/备份用，可保留或 .gitignore；若未使用可删。 |

---

## 五、其他

| 文件夹 | 用途 | 建议 |
|--------|------|------|
| **.trae/** | 文档/计划存档 | 非代码，个人或团队文档；可保留或移出项目。 |
| **run_ui.py**（文件） | UI 启动入口 | 在项目根，必留。 |

---

## 六、根目录零散 test_*.py

根目录存在多份 `test_*.py`（如 test_tokenization、test_dual_mode、test_knowledge_graph 等），与 `tests/` 下用例重叠或为早期单测。建议：

- 若仍需要：迁到 `tests/` 并统一用 pytest 运行。
- 若不再用：删除或移到 `scripts/legacy_tests/` 归档。

---

## 七、整理操作建议（执行前请确认）

1. **可安全删除的空目录**  
   - `templates/`（已空）

2. **建议加入 .gitignore（若尚未忽略）**  
   - `neuro_memory_verify_data/`  
   - `rag_storage_api/`（若不想提交运行时存储）  
   - `test_llm_logic_cache/`  
   - `snapshots/`（若为本地备份）

3. **按需二选一**  
   - 若只用 pip 依赖，可删除本地图存储实现目录。

4. **保留不动**  
   - 核心库、neuro_memory、scripts、docs、tests、run_ui.py。

如需我按上述建议直接执行删除/修改 .gitignore，可说明要执行哪几项。
