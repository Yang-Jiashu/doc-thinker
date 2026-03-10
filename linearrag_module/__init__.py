"""LinearRAG module — NER-based entity extraction as a switchable alternative.

To enable:  set ``ENTITY_EXTRACTION_MODE=ner`` in ``.env``
To disable: set ``ENTITY_EXTRACTION_MODE=llm`` (default)

This module is kept separate so it can be evolved independently and
integrated more deeply in the future.
"""

from .ner_extractor import NERExtractor

__all__ = ["NERExtractor"]
