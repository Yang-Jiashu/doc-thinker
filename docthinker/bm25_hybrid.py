"""BM25 helper utilities shared by GraphCore and HyperGraphRAG."""
#对实体，关系，chunk三类构造BM25索引，以及BM25检索方式
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


@dataclass
class BM25Entry:
    doc_id: str
    text: str
    payload: Dict[str, Any]


class _BM25Index:
    def __init__(self, entries: Sequence[BM25Entry]) -> None:
        self.entries = list(entries)
        self._tokenized = [_tokenize(entry.text) for entry in self.entries]
        self._model: Optional[BM25Okapi]
        if self.entries:
            self._model = BM25Okapi(self._tokenized)
        else:
            self._model = None

    def search(self, query: str, top_k: int) -> List[Tuple[BM25Entry, float]]:
        if not self._model or not query.strip():
            return []
        scores = self._model.get_scores(_tokenize(query))
        scored_entries: List[Tuple[BM25Entry, float]] = [
            (entry, float(score)) for entry, score in zip(self.entries, scores)
        ]
        scored_entries.sort(key=lambda item: item[1], reverse=True)
        return scored_entries[:top_k]


class BM25HybridRetriever:
    """Caches BM25 indexes for chunk/entity/relation corpora."""

    def __init__(self) -> None:
        self._entity_indexes: Dict[int, _BM25Index] = {}
        self._relation_indexes: Dict[int, _BM25Index] = {}
        self._chunk_indexes: Dict[int, _BM25Index] = {}

    def search_entities(
        self,
        knowledge_graph_inst: Any,
        query: str,
        top_k: int,
    ) -> List[Tuple[BM25Entry, float]]:
        index = self._entity_indexes.get(id(knowledge_graph_inst))
        if index is None:
            entries = self._build_entity_entries(knowledge_graph_inst)
            index = _BM25Index(entries)
            self._entity_indexes[id(knowledge_graph_inst)] = index
        return index.search(query, top_k)

    def search_relations(
        self,
        knowledge_graph_inst: Any,
        query: str,
        top_k: int,
    ) -> List[Tuple[BM25Entry, float]]:
        index = self._relation_indexes.get(id(knowledge_graph_inst))
        if index is None:
            entries = self._build_relation_entries(knowledge_graph_inst)
            index = _BM25Index(entries)
            self._relation_indexes[id(knowledge_graph_inst)] = index
        return index.search(query, top_k)

    def search_chunks(
        self,
        chunk_storage: Any,
        query: str,
        top_k: int,
        *,
        doc_filter: Optional[Iterable[str]] = None,
    ) -> List[Tuple[BM25Entry, float]]:
        index = self._chunk_indexes.get(id(chunk_storage))
        if index is None:
            entries = self._build_chunk_entries(chunk_storage)
            index = _BM25Index(entries)
            self._chunk_indexes[id(chunk_storage)] = index
        results = index.search(query, top_k * 3 if doc_filter else top_k)
        if doc_filter:
            allow = {doc.lower() for doc in doc_filter}
            filtered: List[Tuple[BM25Entry, float]] = []
            for entry, score in results:
                doc_id = str(entry.payload.get("full_doc_id") or "").lower()
                if doc_id in allow:
                    filtered.append((entry, score))
                if len(filtered) >= top_k:
                    break
            return filtered
        return results[:top_k]

    @staticmethod
    def _build_entity_entries(storage: Any) -> List[BM25Entry]:
        graph = getattr(storage, "_graph", None)
        if graph is None:
            return []
        entries: List[BM25Entry] = []
        for node_name, data in graph.nodes(data=True):
            description = data.get("description") or ""
            entity_type = data.get("entity_type") or ""
            text = f"{node_name} {entity_type} {description}"
            payload = dict(data)
            payload["entity_name"] = node_name
            entries.append(BM25Entry(doc_id=node_name, text=text, payload=payload))
        return entries

    @staticmethod
    def _build_relation_entries(storage: Any) -> List[BM25Entry]:
        graph = getattr(storage, "_graph", None)
        if graph is None:
            return []
        entries: List[BM25Entry] = []
        for source, target, data in graph.edges(data=True):
            desc = data.get("description") or ""
            hyperedge = data.get("hyperedge_name") or f"{source}-{target}"
            text = f"{hyperedge} {source} {target} {desc}"
            payload = dict(data)
            payload.setdefault("src_id", source)
            payload.setdefault("tgt_id", target)
            payload.setdefault("hyperedge", hyperedge)
            entries.append(
                BM25Entry(doc_id=f"{source}->{target}", text=text, payload=payload)
            )
        return entries

    @staticmethod
    def _build_chunk_entries(storage: Any) -> List[BM25Entry]:
        raw_map = getattr(storage, "_data", None)
        if raw_map is None:
            return []
        # Storage might expose multiprocessing dict like SyncManager; convert to dict
        if hasattr(raw_map, "items"):
            items = list(raw_map.items())
        else:
            items = []
        entries: List[BM25Entry] = []
        for chunk_id, data in items:
            if not data:
                continue
            content = data.get("content") or ""
            payload = dict(data)
            payload.setdefault("chunk_id", chunk_id)
            entries.append(BM25Entry(doc_id=chunk_id, text=content, payload=payload))
        return entries


bm25_retriever = BM25HybridRetriever()


def normalize_bm25_scores(
    scored_items: Sequence[Tuple[BM25Entry, float]]
) -> Dict[str, float]:
    if not scored_items:
        return {}
    scores = [score for _, score in scored_items]
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(max_score, min_score):
        return {item[0].doc_id: 1.0 for item in scored_items}
    return {
        entry.doc_id: (score - min_score) / (max_score - min_score)
        for entry, score in scored_items
    }


__all__ = ["bm25_retriever", "normalize_bm25_scores", "BM25Entry"]
