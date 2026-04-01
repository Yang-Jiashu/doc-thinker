"""Prompt templates for the KG self-study loop (P1 – P6)."""

# ---------------------------------------------------------------------------
# P1 — Subgraph Analysis
# ---------------------------------------------------------------------------
SUBGRAPH_ANALYSIS_PROMPT = """\
你是一个知识图谱分析专家。下面是一个知识子图的结构信息，请分析并输出 JSON。

【子图实体】
{entities_json}

【子图关系】
{relations_json}

请分析并输出严格 JSON：
{{
  "bridge_candidates": [
    {{"entity": "...", "sources": ["doc1", "doc2"], "current_degree": 0}}
  ],
  "same_type_pairs": [
    {{"entity_a": "...", "entity_b": "...", "type": "...", "shared_neighbors": 0}}
  ],
  "two_hop_gaps": [
    {{"a": "...", "bridge": "...", "c": "...", "r1": "...", "r2": "..."}}
  ],
  "hub_entities": [
    {{"entity": "...", "degree": 0, "description_length": 0}}
  ],
  "potential_contradictions": [
    {{"entity": "...", "desc_from_source_1": "...", "desc_from_source_2": "..."}}
  ],
  "isolated_clusters": [
    {{"entities": ["..."], "connection_to_main": "weak"}}
  ],
  "weak_edges": [
    {{"source": "...", "target": "...", "weight": 0.0, "single_source": true}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# P2-A — Bridge multi-hop questions (Strategy 1, weight 40%)
# ---------------------------------------------------------------------------
BRIDGE_QUESTION_PROMPT = """\
你是一个多跳问题生成器。基于以下跨文档桥接实体及其邻居，\
生成需要 2-hop 推理才能回答的问题。

【桥接实体】
{bridge_entity}: {description}
  来源文档 1 的邻居: {neighbors_from_doc1}
  来源文档 2 的邻居: {neighbors_from_doc2}

要求：
1. 每个问题必须跨越至少 2 个实体才能回答
2. 问题应模拟以下模式：
   - "与 [实体A的某属性] 相关的 [实体B] 具有什么特征？"
   - "[实体A] 所属的 [类别] 中，还有哪个实体具有 [某属性]？"
   - "[实体A] 和 [实体C] 通过什么中间环节产生联系？"
3. 同时生成该问题的「理想推理路径」
4. 标注当前 KG 中该路径是否完整

输出严格 JSON 数组：
[
  {{
    "question": "...",
    "hop_count": 2,
    "ideal_path": ["实体A --R1--> 桥接实体 --R2--> 实体C"],
    "path_complete": true,
    "missing_edges": [],
    "question_type": "bridge"
  }}
]

生成 {n_questions} 个问题。
"""

# ---------------------------------------------------------------------------
# P2-B — Two-hop inference questions (Strategy 3, weight 20%)
# ---------------------------------------------------------------------------
TWO_HOP_INFERENCE_PROMPT = """\
你是一个推理路径测试器。以下是知识图谱中存在的 2-hop 路径（A→B→C），\
但 A 和 C 之间没有直接边。请生成测试这些路径是否可推导出直接关系的问题。

【2-hop 路径列表】
{two_hop_paths_json}

要求：
1. 对每条路径生成一个问题，直接问 A 和 C 的关系（不提及 B）
2. 同时生成一个「逆向验证」问题
3. 标注推导类型

输出严格 JSON 数组：
[
  {{
    "question": "...",
    "reverse_question": "...",
    "inference_type": "transitive",
    "path": "A --R1--> B --R2--> C",
    "expected_new_relation": "A --R3--> C"
  }}
]
"""

# ---------------------------------------------------------------------------
# P2-C — Comparison attribute alignment (Strategy 2, weight 15%)
# ---------------------------------------------------------------------------
COMPARISON_QUESTION_PROMPT = """\
你是一个实体比较专家。以下是类型相同但缺少直接比较关系的实体对。\
请生成比较类问题，并识别需要对齐的属性。

【同类实体对】
{same_type_pairs_json}

要求：
1. 生成「A 和 B 在 [某维度] 上有什么不同？」类型的问题
2. 识别 A 有但 B 缺少的属性边（反之亦然）
3. 标注哪些属性需要补全才能支持比较

输出严格 JSON 数组：
[
  {{
    "question": "...",
    "comparison_dimension": "时间",
    "a_has_attribute": true,
    "b_has_attribute": false,
    "missing_for": "B 缺少成立时间的属性边",
    "question_type": "comparison"
  }}
]
"""

# ---------------------------------------------------------------------------
# P2-D — Contradiction detection & memory update
# ---------------------------------------------------------------------------
CONTRADICTION_QUESTION_PROMPT = """\
你是一个知识一致性检查器。以下实体在不同来源中有不同的描述，\
请检测矛盾、过时信息，并生成验证问题。

【存疑实体】
{contradiction_candidates_json}

要求：
1. 对每个实体生成以下三类问题：
   a) 事实一致性：「关于 X，来源 A 说 P，来源 B 说 Q，哪个正确？」
   b) 时效性：「关于 X 的描述是否可能已过时？」
   c) 选择性遗忘：「如果 Q 是最新的正确信息，是否应该废弃 P？」
2. 标注矛盾类型

输出严格 JSON 数组：
[
  {{
    "entity": "X",
    "question": "...",
    "conflict_type": "factual",
    "source_a": {{"id": "...", "description": "...", "time": "..."}},
    "source_b": {{"id": "...", "description": "...", "time": "..."}},
    "suggested_action": "keep_newer"
  }}
]
"""

# ---------------------------------------------------------------------------
# P2-E — Edge validation
# ---------------------------------------------------------------------------
EDGE_VALIDATION_PROMPT = """\
你是一个关系验证专家。以下是知识图谱中证据较弱的边。\
请生成验证这些边是否成立的问题。

【待验证边】
{weak_edges_json}

要求：
1. 对每条边生成一个「正向验证」问题和一个「反面测试」问题
2. 标注验证结果应如何处理

输出严格 JSON 数组：
[
  {{
    "edge": "A --R--> B",
    "positive_question": "...",
    "negative_question": "...",
    "current_evidence_strength": "weak",
    "is_discovered": false
  }}
]
"""

# ---------------------------------------------------------------------------
# P2-F — Weak component bridging (Strategy 5)
# ---------------------------------------------------------------------------
COMPONENT_BRIDGING_PROMPT = """\
你是一个知识图谱连通性分析师。以下是与主图连接薄弱的孤立实体组。\
请尝试发现它们与主图之间可能存在的隐含关系。

【孤立实体组】
{isolated_clusters_json}

【主图中语义最近的实体】
{nearest_main_entities_json}

要求：
1. 对每组生成「连接假设」问题
2. 判断：是真正的孤立（无关领域），还是缺失了连接边？

输出严格 JSON 数组：
[
  {{
    "isolated_entity": "X",
    "nearest_main_entity": "Y",
    "semantic_similarity": 0.85,
    "hypothesis_question": "...",
    "likely_relation_type": "hierarchical"
  }}
]
"""

# ---------------------------------------------------------------------------
# P3 — Answer & Reasoning (unified)
# ---------------------------------------------------------------------------
ANSWER_AND_REASON_PROMPT = """\
你是一个知识图谱推理引擎。请基于以下检索到的子图信息回答问题，\
并详细记录推理路径。

【问题】
{question}
【问题类型】{question_type}

【检索到的实体】
{retrieved_entities}

【检索到的关系】
{retrieved_relations}

【检索到的文本块】
{retrieved_chunks}

请输出严格 JSON：
{{
  "answer": "基于证据的回答",
  "confidence": 0.0,
  "reasoning_chain": [
    {{"step": 1, "action": "...", "evidence_triple_id": "...", "conclusion": "..."}}
  ],
  "path_used": "A --R1--> B --R2--> C",
  "missing_info": [
    {{"description": "...", "severity": "high"}}
  ],
  "contradictions_found": [
    {{"entity": "...", "conflict": "...", "recommendation": "..."}}
  ],
  "answerable": true
}}
"""

# ---------------------------------------------------------------------------
# P4 — Knowledge Synthesis & Writeback
# ---------------------------------------------------------------------------
KNOWLEDGE_SYNTHESIS_PROMPT = """\
你是一个知识图谱进化引擎。基于前面的出题-答题过程，\
决定哪些新知识应该写回知识图谱。

【本轮自习记录】
{all_qa_records_json}

请按以下类别输出需要执行的 KG 操作，严格 JSON：
{{
  "new_edges": [
    {{
      "source": "...",
      "target": "...",
      "relation": "...",
      "keywords": "...",
      "inference_type": "transitive",
      "evidence_chain": ["triple_id_1", "triple_id_2"],
      "confidence": 0.0
    }}
  ],
  "edge_updates": [
    {{
      "source": "...", "target": "...",
      "action": "increase_weight",
      "reason": "...",
      "new_weight": 0.0,
      "additional_source_id": "..."
    }}
  ],
  "entity_updates": [
    {{
      "entity": "...",
      "action": "enrich_description",
      "new_content": "...",
      "evidence": ["triple_id"]
    }}
  ],
  "contradiction_flags": [
    {{
      "entity": "...",
      "conflicting_sources": ["s1", "s2"],
      "conflict_type": "factual",
      "suggested_resolution": "keep_newer",
      "edges_affected": []
    }}
  ],
  "deprecation_candidates": [
    {{
      "entity_or_edge": "...",
      "reason": "...",
      "newer_replacement": "...",
      "action": "deprecate"
    }}
  ],
  "summary_nodes": [
    {{
      "new_entity": "...",
      "entity_type": "concept",
      "description": "...",
      "member_entities": ["e1", "e2", "e3"],
      "relation_to_members": "包含",
      "evidence": []
    }}
  ]
}}

写回规则：
1. new_edges: confidence < 0.5 的不输出；evidence_chain 为空的不输出
2. contradiction_flags: 不自动解决矛盾，只标记
3. deprecation_candidates: 必须有明确证据才能建议废弃
4. summary_nodes: 至少归纳 3 个已有实体才允许创建汇总节点
5. 所有操作都必须附带 evidence
"""

# ---------------------------------------------------------------------------
# P5 — Experience Extraction
# ---------------------------------------------------------------------------
EXPERIENCE_EXTRACTION_PROMPT = """\
你是一个知识图谱的「学习教练」。刚才完成了一轮自习，\
请从整轮过程中提炼可复用的经验。

【本轮自习完整记录】
{full_study_session_json}

请从以下 5 个维度提取经验，输出严格 JSON：
{{
  "retrieval_experiences": [
    {{
      "experience_id": "exp_ret_001",
      "pattern": "什么场景下",
      "effective_strategy": "什么检索策略有效",
      "ineffective_strategy": "什么策略无效",
      "evidence": {{
        "question_ids": [],
        "success_rate_with": 0.0,
        "success_rate_without": 0.0
      }},
      "applicable_to": [],
      "confidence": 0.0
    }}
  ],
  "reasoning_experiences": [
    {{
      "experience_id": "exp_rea_001",
      "pattern": "什么推理模式",
      "reliability": "high",
      "pattern_negative": "什么推理模式不可靠",
      "reliability_negative": "low",
      "reason": "为什么",
      "evidence": {{
        "correct_inferences": [],
        "incorrect_inferences": []
      }},
      "applicable_to": [],
      "confidence": 0.0
    }}
  ],
  "failure_experiences": [
    {{
      "experience_id": "exp_fail_001",
      "failure_pattern": "什么场景容易出错",
      "root_cause": "根本原因",
      "suggested_fix": "建议修复方式",
      "affected_question_types": [],
      "frequency": "本轮出现频次描述",
      "priority": "high",
      "confidence": 0.0
    }}
  ],
  "structural_experiences": [
    {{
      "experience_id": "exp_str_001",
      "observation": "图结构层面的观察",
      "implication": "对后续操作的含义",
      "threshold": {{"metric": "...", "critical_value": 0}},
      "evidence": {{}},
      "applicable_to": [],
      "confidence": 0.0
    }}
  ],
  "meta_experiences": [
    {{
      "experience_id": "exp_meta_001",
      "observation": "关于自习过程本身的发现",
      "implication": "对后续自习策略的调整建议",
      "current_allocation": {{}},
      "suggested_allocation": {{}},
      "confidence": 0.0
    }}
  ]
}}

提取规则：
1. 每条经验必须有 evidence
2. confidence < 0.5 的不输出
3. 经验应是可泛化的模式，不是针对单个实体的特例
4. 区分「可靠模式」和「不可靠模式」，两者都要记录
5. failure_experiences 必须附带 root_cause 和 suggested_fix
"""

# ---------------------------------------------------------------------------
# P6 — Experience Refinement
# ---------------------------------------------------------------------------
EXPERIENCE_REFINEMENT_PROMPT = """\
你是一个经验精炼引擎。以下经验已经被使用了多次，\
请根据使用记录判断是否需要更新。

【待精炼经验】
{experience_with_usage_stats}

请判断并输出严格 JSON：
{{
  "action": "keep",
  "refined_experience": null,
  "merge_with": null,
  "merged_result": null,
  "reason": "保持不变的原因 / 精炼 / 合并 / 废弃的原因"
}}

action 取值：keep | refine | merge | deprecate

精炼规则：
1. times_retrieved >= 5 且 times_useful / times_retrieved < 0.2 → 考虑 deprecate
2. 两条经验的 applicable_to 高度重叠 → 考虑 merge
3. 新证据与旧经验矛盾 → 用新证据 refine
4. confidence 随使用反馈动态调整
"""
