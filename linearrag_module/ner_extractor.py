"""NER-based entity extraction — zero LLM cost alternative to operate.extract_entities.

Adapted from LinearRAG's SpacyNER.  Produces ``custom_kg`` dicts that
GraphCore can ingest via ``ainsert_custom_kg``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SKIP_LABELS = {"ORDINAL", "CARDINAL"}


def _md5(text: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


class NERExtractor:
    """Spacy-backed NER that works for both Chinese and English."""

    _model_cache: dict[str, Any] = {}

    def __init__(
        self,
        spacy_model: str = "zh_core_web_sm",
        fallback_model: str | None = "en_core_web_sm",
    ):
        self.spacy_model_name = spacy_model
        self.fallback_model_name = fallback_model
        self._nlp = self._load_model(spacy_model, fallback_model)

    @classmethod
    def _load_model(cls, model_name: str, fallback: str | None = None):
        if model_name in cls._model_cache:
            return cls._model_cache[model_name]
        try:
            import spacy
            nlp = spacy.load(model_name)
            cls._model_cache[model_name] = nlp
            logger.info(f"Loaded spacy model: {model_name}")
            return nlp
        except Exception as exc:
            logger.warning(f"Cannot load spacy model '{model_name}': {exc}")
            if fallback and fallback != model_name:
                return cls._load_model(fallback, None)
            raise RuntimeError(
                f"No usable spacy model. Install one via: python -m spacy download {model_name}"
            ) from exc

    def extract_from_chunks(
        self,
        chunks: Dict[str, Dict[str, Any]],
        file_path: str = "unknown",
    ) -> Dict[str, Any]:
        """Extract entities from GraphCore-style chunk dict.

        Returns a ``custom_kg`` dict ready for ``ainsert_custom_kg``.
        """
        chunk_list: List[Dict[str, Any]] = []
        entities_list: List[Dict[str, str]] = []
        relationships_list: List[Dict[str, Any]] = []
        seen_entities: Dict[str, str] = {}

        chunk_ids = list(chunks.keys())
        texts = [chunks[cid]["content"] for cid in chunk_ids]

        batch_size = max(1, len(texts) // 4)
        docs = list(self._nlp.pipe(texts, batch_size=batch_size))

        for idx, doc in enumerate(docs):
            cid = chunk_ids[idx]
            chunk_content = chunks[cid]["content"]
            chunk_order = chunks[cid].get("chunk_order_index", idx)

            chunk_list.append({
                "content": chunk_content,
                "source_id": cid,
                "file_path": file_path,
                "chunk_order_index": chunk_order,
            })

            sentence_entities: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

            for ent in doc.ents:
                if ent.label_ in SKIP_LABELS:
                    continue
                ent_text = ent.text.strip()
                if not ent_text or len(ent_text) < 2:
                    continue
                ent_label = ent.label_
                sent_text = ent.sent.text.strip() if ent.sent else ""

                if ent_text not in seen_entities:
                    seen_entities[ent_text] = ent_label
                    entities_list.append({
                        "entity_name": ent_text,
                        "entity_type": ent_label.lower(),
                        "description": f"[{ent_label}] {ent_text} (from: {sent_text[:120]})",
                        "source_id": cid,
                        "file_path": file_path,
                    })
                if sent_text:
                    sentence_entities[sent_text].append((ent_text, ent_label))

            for sent_text, ent_pairs in sentence_entities.items():
                names = [e[0] for e in ent_pairs]
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        relationships_list.append({
                            "src_id": names[i],
                            "tgt_id": names[j],
                            "description": f"Co-occur in: {sent_text[:200]}",
                            "keywords": "co-occurrence",
                            "source_id": cid,
                            "file_path": file_path,
                            "weight": 1.0,
                        })

        logger.info(
            f"NER extraction: {len(entities_list)} entities, "
            f"{len(relationships_list)} relations from {len(chunk_list)} chunks"
        )

        return {
            "chunks": chunk_list,
            "entities": entities_list,
            "relationships": relationships_list,
        }

    def extract_from_text(
        self,
        text: str,
        file_path: str = "unknown",
        chunk_token_size: int = 1200,
    ) -> Dict[str, Any]:
        """Convenience: chunk raw text by sentences, then extract.

        Uses simple sentence-boundary chunking to avoid depending on
        GraphCore's tokenizer at this level.
        """
        doc = self._nlp(text)
        sents = list(doc.sents)

        chunks: Dict[str, Dict[str, Any]] = {}
        current_text = ""
        chunk_idx = 0

        for sent in sents:
            candidate = (current_text + " " + sent.text).strip() if current_text else sent.text
            if len(candidate) > chunk_token_size * 3:
                if current_text:
                    cid = _md5(current_text, prefix="chunk-")
                    chunks[cid] = {
                        "content": current_text,
                        "chunk_order_index": chunk_idx,
                    }
                    chunk_idx += 1
                current_text = sent.text
            else:
                current_text = candidate

        if current_text.strip():
            cid = _md5(current_text, prefix="chunk-")
            chunks[cid] = {
                "content": current_text,
                "chunk_order_index": chunk_idx,
            }

        return self.extract_from_chunks(chunks, file_path=file_path)

    def extract_query_entities(self, question: str) -> Set[str]:
        """Extract entity names from a user query (for retrieval seeding)."""
        doc = self._nlp(question)
        return {
            ent.text.strip().lower()
            for ent in doc.ents
            if ent.label_ not in SKIP_LABELS and len(ent.text.strip()) >= 2
        }
