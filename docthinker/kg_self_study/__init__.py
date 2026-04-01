"""KG Self-Study Loop: test-time scaling on knowledge graphs.

Background process that lets the LLM autonomously quiz, reason over,
and densify the KG — improving retrieval quality before user queries arrive.
"""

from .orchestrator import SelfStudyOrchestrator
from .experience_manager import ExperienceManager

__all__ = [
    "SelfStudyOrchestrator",
    "ExperienceManager",
]
