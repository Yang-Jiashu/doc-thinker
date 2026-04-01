# KG 自习循环设计文档（Test-Time Scaling on KG）

## 1. 定位

在文档 ingest（写入）和用户 query（读取）之间，插入**后台自习循环**：
LLM 对已有 KG **自主出题 → 检索答题 → 归纳演绎 → 写回新知识 + 经验**。

目标不是预先生成答案，而是**提升 KG 的密度与质量**——更多边、更紧关联、更高信息密度——让同样的检索机制捞出更好的上下文。

核心比喻：**上课记笔记 → 考前刷题 → 真正考试**。

## 2. 模块位置

```
docthinker/
├── kg_self_study/          ← 新模块
│   ├── __init__.py
│   ├── prompts.py          # P1-P6 prompt 模板
│   ├── subgraph_selector.py # 6 种选区策略
│   ├── question_generator.py# 出题（P2）
│   ├── experience_manager.py# 经验存储/检索/精炼
│   └── orchestrator.py     # 自习循环主控
```

## 3. 流水线

```
P1 子图分析 → P2 出题 → P3 答题+推理 → P4 知识凝练 → P5 经验提取 → P6 经验精炼
```

## 4. 选区策略（针对 HotpotQA 优化）

| 策略 | 权重 | 说明 |
|------|------|------|
| 桥接实体优先 | 40% | 跨 ≥2 source_id 的实体，补多跳路径 |
| 2-hop 路径补全 | 20% | A→B→C 存在但 A→C 无直接边 |
| 同类属性对齐 | 15% | entity_type 相同但无直接边的对 |
| 弱连通打通 | 10% | 不同连通分量间 embedding 最近的实体对 |
| 枢纽实体深化 | 10% | 高度数实体的细节补充 |
| 证据链强化 | 5% | source_id 单一或 weight 低的边 |

## 5. 经验系统（仿 ReMe/HINDSIGHT/MEMO）

经验 = 自习过程中产生的**方法论级元知识**，存为 KG 中 `entity_type="experience"` 的特殊节点。

5 个维度：检索策略经验、推理模式经验、失败模式经验、结构模式经验、元经验。

经验生命周期：获取 → 重用 → 精炼 → 废弃（仿 ReMe 的 Acquisition → Reuse → Refinement）。

## 6. 写回规则

- 新边 confidence < 0.5 不写入
- 新边需 ≥ 2 条依据三元组
- 汇总节点需归纳 ≥ 3 个已有实体
- 所有写回带 `source="self_study"` 标记
- 接入现有 candidate → active → promoted 生命周期

## 7. 预算控制

- `max_rounds_per_session`: 5
- `max_tokens_per_session`: 50000
- `questions_per_round`: 3
- 连续 2 轮无新产出 → 早停
- 自习用模型可独立配置（如 qwen-turbo 降本）

## 8. 触发方式

- ingest 后自动触发
- 定时触发（可配置间隔）
- 手动触发（API / 管理界面）
- 空闲触发（无用户请求时）
