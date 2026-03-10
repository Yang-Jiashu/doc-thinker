"""Thin re-export — actual implementation lives in linearrag_module/."""
from linearrag_module.ner_extractor import NERExtractor, SKIP_LABELS  # noqa: F401

__all__ = ["NERExtractor", "SKIP_LABELS"]
