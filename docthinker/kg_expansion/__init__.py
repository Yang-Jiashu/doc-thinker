# docthinker/kg_expansion/__init__.py
"""知识图谱 LLM 扩展：分层送入 + 多角度 prompting，生成联想节点。"""

from .expander import KGExpander
from .manager import ExpandedNodeManager, extract_entities_from_text

__all__ = ["KGExpander", "ExpandedNodeManager", "extract_entities_from_text"]
