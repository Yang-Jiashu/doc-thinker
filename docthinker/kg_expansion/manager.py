from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip()).lower()


def _tokenize(text: str) -> List[str]:
    lowered = str(text or "").lower()
    return [t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", lowered) if t]


def extract_entities_from_text(text: str, max_entities: int = 12) -> List[str]:
    """
    Lightweight entity extraction fallback for expansion promotion.
    This is intentionally simple and deterministic to keep latency low.
    """
    source = str(text or "")
    if not source:
        return []

    entities: List[str] = []
    seen = set()

    for m in re.finditer(r"[\u4e00-\u9fff]{2,10}", source):
        name = m.group(0).strip()
        if name and name not in seen:
            seen.add(name)
            entities.append(name)
            if len(entities) >= max_entities:
                return entities

    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9_\-]{2,32}\b", source):
        name = m.group(0).strip()
        if len(name) < 3:
            continue
        if name.lower() in {"the", "and", "for", "with", "from", "this", "that"}:
            continue
        if name not in seen:
            seen.add(name)
            entities.append(name)
            if len(entities) >= max_entities:
                return entities

    return entities


class ExpandedNodeManager:
    """
    Session-scoped lifecycle manager for LLM expanded nodes.
    Data is persisted to a JSON file in each session knowledge directory.
    """

    def __init__(
        self,
        storage_path: Path,
        *,
        promote_score_threshold: float = 1.2,
        promote_use_threshold: int = 2,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.promote_score_threshold = float(promote_score_threshold)
        self.promote_use_threshold = int(promote_use_threshold)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._lock = RLock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self.storage_path.exists():
                try:
                    payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        items = payload.get("nodes") or []
                    elif isinstance(payload, list):
                        items = payload
                    else:
                        items = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        entity = str(item.get("entity") or "").strip()
                        if not entity:
                            continue
                        key = _normalize_name(entity)
                        self._records[key] = self._normalize_record(item)
                except Exception:
                    self._records = {}
            self._loaded = True

    def _normalize_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now_iso()
        entity = str(item.get("entity") or "").strip()
        roots = item.get("root_ids") or []
        entities = item.get("attached_entities") or []
        return {
            "entity": entity,
            "status": str(item.get("status") or "candidate"),
            "reason": str(item.get("reason") or ""),
            "angle": str(item.get("angle") or ""),
            "source": str(item.get("source") or "llm_expansion"),
            "root_ids": [str(x).strip() for x in roots if str(x).strip()],
            "hit_count": int(item.get("hit_count") or 0),
            "use_count": int(item.get("use_count") or 0),
            "promotion_score": float(item.get("promotion_score") or 0.0),
            "last_hit_at": item.get("last_hit_at"),
            "last_used_at": item.get("last_used_at"),
            "created_at": item.get("created_at") or now,
            "updated_at": item.get("updated_at") or now,
            "attached_entities": [str(x).strip() for x in entities if str(x).strip()],
        }

    def _persist(self) -> None:
        with self._lock:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": _utc_now_iso(),
                "nodes": sorted(self._records.values(), key=lambda x: x.get("entity", "")),
            }
            self.storage_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def upsert_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        default_root_ids: Optional[Iterable[str]] = None,
        source: str = "llm_expansion",
    ) -> Dict[str, int]:
        self._ensure_loaded()
        added = 0
        updated = 0
        roots = [str(x).strip() for x in (default_root_ids or []) if str(x).strip()]
        now = _utc_now_iso()

        with self._lock:
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity") or "").strip()
                if not entity:
                    continue
                key = _normalize_name(entity)
                existing = self._records.get(key)
                if existing is None:
                    self._records[key] = self._normalize_record(
                        {
                            "entity": entity,
                            "status": "candidate",
                            "reason": item.get("reason", ""),
                            "angle": item.get("angle", ""),
                            "source": source,
                            "root_ids": list(roots),
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    added += 1
                else:
                    merged_roots = list(dict.fromkeys([*(existing.get("root_ids") or []), *roots]))
                    existing["reason"] = str(item.get("reason") or existing.get("reason") or "")
                    existing["angle"] = str(item.get("angle") or existing.get("angle") or "")
                    existing["root_ids"] = merged_roots
                    existing["updated_at"] = now
                    updated += 1
            self._persist()

        return {"added": added, "updated": updated}

    def list_nodes(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            items = list(self._records.values())
        if status:
            items = [x for x in items if str(x.get("status") or "") == status]
        items.sort(
            key=lambda x: (
                -float(x.get("promotion_score") or 0.0),
                -int(x.get("use_count") or 0),
                x.get("entity") or "",
            )
        )
        return items[: max(1, int(limit))]

    def get(self, entity: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        key = _normalize_name(entity)
        with self._lock:
            item = self._records.get(key)
            return dict(item) if item else None

    def match_nodes(
        self,
        query: str,
        *,
        top_k: int = 2,
        memory_terms: Optional[Sequence[str]] = None,
        min_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        q = str(query or "").strip()
        if not q:
            return []

        q_tokens = set(_tokenize(q))
        mem_tokens = set(_tokenize(" ".join(memory_terms or [])))
        matches: List[Dict[str, Any]] = []

        with self._lock:
            items = list(self._records.values())

        for item in items:
            status = str(item.get("status") or "")
            if status == "deprecated":
                continue

            entity = str(item.get("entity") or "")
            reason = str(item.get("reason") or "")
            corpus = f"{entity} {reason}".strip()
            corpus_lower = corpus.lower()
            score = 0.0

            if entity and entity.lower() in q.lower():
                score += 0.6
            if corpus and q.lower() in corpus_lower:
                score += 0.2

            c_tokens = set(_tokenize(corpus))
            if q_tokens and c_tokens:
                inter = len(q_tokens & c_tokens)
                union = len(q_tokens | c_tokens) or 1
                score += 0.5 * (inter / union)

            if mem_tokens and c_tokens:
                inter_mem = len(mem_tokens & c_tokens)
                if inter_mem:
                    score += min(0.2, 0.05 * inter_mem)

            if score >= min_score:
                enriched = dict(item)
                enriched["score"] = round(float(score), 4)
                matches.append(enriched)

        matches.sort(
            key=lambda x: (
                -float(x.get("score") or 0.0),
                -float(x.get("promotion_score") or 0.0),
                x.get("entity") or "",
            )
        )
        return matches[: max(1, int(top_k))]

    def mark_hits(self, entities: Sequence[str]) -> None:
        self._ensure_loaded()
        now = _utc_now_iso()
        with self._lock:
            for entity in entities:
                key = _normalize_name(entity)
                item = self._records.get(key)
                if not item:
                    continue
                item["hit_count"] = int(item.get("hit_count") or 0) + 1
                item["last_hit_at"] = now
                item["updated_at"] = now
            self._persist()

    def record_response_usage(
        self,
        *,
        answer: str,
        matches: Sequence[Dict[str, Any]],
        attached_entities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        answer_text = str(answer or "").lower()
        now = _utc_now_iso()
        used: List[str] = []
        promoted: List[str] = []
        attached = [str(x).strip() for x in (attached_entities or []) if str(x).strip()]

        with self._lock:
            for matched in matches:
                entity = str(matched.get("entity") or "").strip()
                if not entity:
                    continue
                key = _normalize_name(entity)
                item = self._records.get(key)
                if not item:
                    continue
                if entity.lower() in answer_text:
                    used.append(entity)
                    item["use_count"] = int(item.get("use_count") or 0) + 1
                    item["last_used_at"] = now
                    item["promotion_score"] = float(item.get("promotion_score") or 0.0) + max(
                        0.4, float(matched.get("score") or 0.4)
                    )
                    item["status"] = self._next_status(item)
                    if item["status"] == "promoted":
                        promoted.append(entity)
                    for ent in attached:
                        if ent not in item["attached_entities"]:
                            item["attached_entities"].append(ent)
                else:
                    item["promotion_score"] = max(
                        0.0, float(item.get("promotion_score") or 0.0) - 0.05
                    )
                item["updated_at"] = now
            self._persist()

        return {"used": used, "promoted": promoted}

    def _next_status(self, item: Dict[str, Any]) -> str:
        score = float(item.get("promotion_score") or 0.0)
        uses = int(item.get("use_count") or 0)
        current = str(item.get("status") or "candidate")

        if current == "promoted":
            return "promoted"
        if uses >= self.promote_use_threshold and score >= self.promote_score_threshold:
            return "promoted"
        if score >= 0.6:
            return "active"
        return "candidate"

    def build_forced_instruction(
        self,
        matches: Sequence[Dict[str, Any]],
        *,
        limit: int = 2,
    ) -> str:
        selected = list(matches)[: max(1, int(limit))]
        if not selected:
            return ""
        lines: List[str] = []
        lines.append(
            "请在回答时优先核对以下扩展节点（至少匹配其中1个，最多2个），"
            "并说明它们与问题的关系。"
        )
        for item in selected:
            entity = str(item.get("entity") or "").strip()
            reason = str(item.get("reason") or "").strip()
            angle = str(item.get("angle") or "").strip()
            line = f"- 节点: {entity}"
            if angle:
                line += f" | 角度: {angle}"
            if reason:
                line += f" | 提示: {reason[:220]}"
            lines.append(line)
        lines.append("若节点被实际采用，请将其与当前结论建立明确逻辑关系。")
        return "\n".join(lines)
