"""Experience storage, retrieval, and refinement for the KG self-study loop.

Experiences are meta-knowledge about *how* to reason over the KG, stored as
special ``entity_type="experience"`` nodes.  Inspired by ReMe's
Acquisition → Reuse → Refinement cycle and HINDSIGHT's four-layer memory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set

from .prompts import EXPERIENCE_EXTRACTION_PROMPT, EXPERIENCE_REFINEMENT_PROMPT

_log = logging.getLogger("docthinker.kg_self_study.experience")

EXPERIENCE_CATEGORIES = (
    "retrieval_experiences",
    "reasoning_experiences",
    "failure_experiences",
    "structural_experiences",
    "meta_experiences",
)


def _safe_json_parse(raw: str) -> Any:
    text = raw.strip()
    for sc, ec in [("{", "}"), ("[", "]")]:
        s = text.find(sc)
        e = text.rfind(ec)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s: e + 1])
            except json.JSONDecodeError:
                continue
    return None


class ExperienceManager:
    """Manages the experience pool (procedural memory layer on KG)."""

    def __init__(self, store_path: Optional[str] = None):
        self._lock = RLock()
        self._experiences: Dict[str, Dict[str, Any]] = {}
        self._store_path = Path(store_path) if store_path else None
        if self._store_path and self._store_path.exists():
            self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._experiences = data if isinstance(data, dict) else {}
            _log.info("[experience] loaded %d experiences", len(self._experiences))
        except Exception as exc:
            _log.warning("[experience] load failed: %s", exc)

    def _save(self) -> None:
        if not self._store_path:
            return
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps(self._experiences, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            _log.warning("[experience] save failed: %s", exc)

    # -- P5: extraction ------------------------------------------------------

    async def extract_experiences(
        self,
        session_record: Dict[str, Any],
        llm_func: Callable,
    ) -> Dict[str, Any]:
        """Run P5 prompt to extract experiences from a study session."""
        prompt = EXPERIENCE_EXTRACTION_PROMPT.format(
            full_study_session_json=json.dumps(
                session_record, ensure_ascii=False, indent=2,
            )[:12000],
        )

        try:
            raw = await llm_func(prompt)
            result = _safe_json_parse(raw)
            if not isinstance(result, dict):
                _log.warning("[experience] P5 parse failed")
                return {}
        except Exception as exc:
            _log.error("[experience] P5 LLM call failed: %s", exc)
            return {}

        added = 0
        for category in EXPERIENCE_CATEGORIES:
            for exp in result.get(category, []):
                exp_id = exp.get("experience_id", "")
                if not exp_id:
                    continue
                confidence = float(exp.get("confidence", 0))
                if confidence < 0.5:
                    continue
                exp["category"] = category
                exp["created_at"] = int(time.time())
                exp["times_retrieved"] = 0
                exp["times_useful"] = 0
                exp["version"] = 1
                exp["status"] = "active"
                with self._lock:
                    self._experiences[exp_id] = exp
                added += 1

        _log.info("[experience] extracted %d experiences", added)
        self._save()
        return result

    # -- retrieval (for injection at query time) -----------------------------

    def retrieve_relevant(
        self,
        question_type: str = "",
        entity_types: Optional[List[str]] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to a question type or entity types."""
        entity_types = entity_types or []
        candidates: List[Dict[str, Any]] = []

        with self._lock:
            for exp in self._experiences.values():
                if exp.get("status") == "deprecated":
                    continue
                applicable = exp.get("applicable_to", [])
                score = 0.0
                if question_type and question_type in applicable:
                    score += 1.0
                for et in entity_types:
                    if et in applicable:
                        score += 0.5
                confidence = float(exp.get("confidence", 0.5))
                score += confidence * 0.3
                if score > 0:
                    candidates.append({**exp, "_relevance_score": score})

        candidates.sort(key=lambda x: -x["_relevance_score"])
        results = candidates[:max_results]

        with self._lock:
            for r in results:
                exp_id = r.get("experience_id", "")
                if exp_id in self._experiences:
                    self._experiences[exp_id]["times_retrieved"] = (
                        self._experiences[exp_id].get("times_retrieved", 0) + 1
                    )
            self._save()

        return results

    def record_usefulness(self, experience_id: str, useful: bool) -> None:
        """Record whether a retrieved experience was actually useful."""
        with self._lock:
            exp = self._experiences.get(experience_id)
            if not exp:
                return
            if useful:
                exp["times_useful"] = exp.get("times_useful", 0) + 1
                exp["confidence"] = min(1.0, exp.get("confidence", 0.5) + 0.05)
            else:
                exp["confidence"] = max(0.0, exp.get("confidence", 0.5) - 0.03)
            self._save()

    # -- P6: refinement ------------------------------------------------------

    async def refine_experiences(
        self,
        llm_func: Callable,
        min_retrievals: int = 5,
    ) -> int:
        """Run P6 refinement on experiences that have been used enough."""
        to_refine: List[Dict[str, Any]] = []
        with self._lock:
            for exp in self._experiences.values():
                if exp.get("status") == "deprecated":
                    continue
                if exp.get("times_retrieved", 0) >= min_retrievals:
                    to_refine.append(exp.copy())

        if not to_refine:
            return 0

        refined_count = 0
        for exp in to_refine:
            prompt = EXPERIENCE_REFINEMENT_PROMPT.format(
                experience_with_usage_stats=json.dumps(
                    exp, ensure_ascii=False, indent=2,
                ),
            )

            try:
                raw = await llm_func(prompt)
                result = _safe_json_parse(raw)
                if not isinstance(result, dict):
                    continue
            except Exception as exc:
                _log.warning("[experience] P6 call failed: %s", exc)
                continue

            action = result.get("action", "keep")
            exp_id = exp.get("experience_id", "")

            with self._lock:
                if action == "deprecate":
                    if exp_id in self._experiences:
                        self._experiences[exp_id]["status"] = "deprecated"
                        refined_count += 1
                elif action == "refine" and result.get("refined_experience"):
                    refined = result["refined_experience"]
                    if exp_id in self._experiences:
                        for k, v in refined.items():
                            self._experiences[exp_id][k] = v
                        self._experiences[exp_id]["version"] = (
                            self._experiences[exp_id].get("version", 1) + 1
                        )
                        refined_count += 1
                elif action == "merge" and result.get("merge_with"):
                    merge_target = result["merge_with"]
                    if exp_id in self._experiences:
                        self._experiences[exp_id]["status"] = "deprecated"
                    if merge_target in self._experiences and result.get("merged_result"):
                        for k, v in result["merged_result"].items():
                            self._experiences[merge_target][k] = v
                        self._experiences[merge_target]["version"] = (
                            self._experiences[merge_target].get("version", 1) + 1
                        )
                    refined_count += 1

        self._save()
        _log.info("[experience] refined %d experiences", refined_count)
        return refined_count

    # -- KG node conversion --------------------------------------------------

    def to_kg_nodes(self) -> List[Dict[str, Any]]:
        """Convert active experiences to KG-compatible entity dicts.

        These can be upserted into the entity VDB so they are retrievable
        alongside regular entities at query time.
        """
        nodes = []
        with self._lock:
            for exp in self._experiences.values():
                if exp.get("status") == "deprecated":
                    continue
                exp_id = exp.get("experience_id", "")
                category = exp.get("category", "general")
                desc_parts = []
                for key in ("pattern", "effective_strategy",
                            "observation", "failure_pattern",
                            "implication", "suggested_fix"):
                    val = exp.get(key)
                    if val:
                        desc_parts.append(f"{key}: {val}")
                description = " | ".join(desc_parts) or exp_id

                nodes.append({
                    "entity_id": exp_id,
                    "entity_type": "experience",
                    "description": description,
                    "source_id": "self_study",
                    "metadata": {
                        "category": category,
                        "applicable_to": exp.get("applicable_to", []),
                        "confidence": exp.get("confidence", 0.5),
                        "version": exp.get("version", 1),
                    },
                })
        return nodes

    # -- stats ---------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._experiences)
            active = sum(1 for e in self._experiences.values()
                         if e.get("status") != "deprecated")
            deprecated = total - active
        return {"total": total, "active": active, "deprecated": deprecated}
