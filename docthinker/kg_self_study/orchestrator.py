"""Self-study orchestrator: runs the full P1→P6 pipeline.

Coordinates subgraph selection, question generation, KG-based answering,
knowledge synthesis, and experience extraction in a budgeted loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .experience_manager import ExperienceManager
from .prompts import ANSWER_AND_REASON_PROMPT, KNOWLEDGE_SYNTHESIS_PROMPT
from .question_generator import QuestionGenerator
from .subgraph_selector import SubgraphSelector

_log = logging.getLogger("docthinker.kg_self_study")


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


@dataclass
class SelfStudyConfig:
    max_rounds: int = 5
    max_tokens: int = 50000
    questions_per_round: int = 3
    min_new_knowledge_to_continue: int = 1
    strategy_weights: Optional[Dict[str, float]] = None
    max_entities_per_round: int = 40
    experience_store_path: Optional[str] = None
    writeback_confidence_threshold: float = 0.5


@dataclass
class StudyRoundResult:
    round_idx: int
    strategy: str
    questions_asked: int = 0
    questions_answered: int = 0
    new_edges_proposed: int = 0
    entity_updates_proposed: int = 0
    contradictions_found: int = 0
    experiences_extracted: int = 0
    synthesis_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StudySessionResult:
    rounds: List[StudyRoundResult] = field(default_factory=list)
    total_new_edges: int = 0
    total_entity_updates: int = 0
    total_contradictions: int = 0
    total_experiences: int = 0
    elapsed_seconds: float = 0.0
    stopped_reason: str = ""


class SelfStudyOrchestrator:
    """Main controller for the KG self-study loop."""

    def __init__(
        self,
        llm_func: Callable,
        kg_query_func: Callable,
        kg_write_func: Callable,
        kg_read_nodes_func: Callable,
        kg_read_edges_func: Callable,
        config: Optional[SelfStudyConfig] = None,
    ):
        """
        Args:
            llm_func: async (prompt: str) -> str
            kg_query_func: async (question: str) -> dict with
                           retrieved_entities, retrieved_relations, retrieved_chunks
            kg_write_func: async (operations: dict) -> None
                           accepts the P4 synthesis output format
            kg_read_nodes_func: async () -> list[dict]  (all KG nodes)
            kg_read_edges_func: async () -> list[dict]  (all KG edges)
        """
        self.llm_func = llm_func
        self.kg_query_func = kg_query_func
        self.kg_write_func = kg_write_func
        self.kg_read_nodes = kg_read_nodes_func
        self.kg_read_edges = kg_read_edges_func
        self.config = config or SelfStudyConfig()

        self.selector = SubgraphSelector(
            strategy_weights=self.config.strategy_weights,
            max_entities_per_round=self.config.max_entities_per_round,
        )
        self.question_gen = QuestionGenerator(
            llm_func=llm_func,
            questions_per_strategy=self.config.questions_per_round,
        )
        self.experience_mgr = ExperienceManager(
            store_path=self.config.experience_store_path,
        )

    async def run_session(self) -> StudySessionResult:
        """Execute a full self-study session (multiple rounds)."""
        t0 = time.time()
        session = StudySessionResult()
        consecutive_empty = 0

        _log.info("[self_study] session starting (max_rounds=%d)", self.config.max_rounds)

        all_nodes = await self.kg_read_nodes()
        all_edges = await self.kg_read_edges()

        if len(all_nodes) < 3:
            session.stopped_reason = "too_few_entities"
            _log.info("[self_study] only %d entities, skipping", len(all_nodes))
            return session

        for round_idx in range(self.config.max_rounds):
            _log.info("[self_study] round %d/%d", round_idx + 1, self.config.max_rounds)

            result = await self._run_round(round_idx, all_nodes, all_edges)
            session.rounds.append(result)
            session.total_new_edges += result.new_edges_proposed
            session.total_entity_updates += result.entity_updates_proposed
            session.total_contradictions += result.contradictions_found
            session.total_experiences += result.experiences_extracted

            if result.new_edges_proposed + result.entity_updates_proposed == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            if consecutive_empty >= 2:
                session.stopped_reason = "no_new_knowledge"
                _log.info("[self_study] 2 empty rounds, early stopping")
                break

            all_nodes = await self.kg_read_nodes()
            all_edges = await self.kg_read_edges()

        if not session.stopped_reason:
            session.stopped_reason = "max_rounds_reached"

        session.elapsed_seconds = time.time() - t0
        _log.info(
            "[self_study] session done in %.1fs: %d edges, %d updates, "
            "%d contradictions, %d experiences",
            session.elapsed_seconds,
            session.total_new_edges,
            session.total_entity_updates,
            session.total_contradictions,
            session.total_experiences,
        )
        return session

    async def _run_round(
        self,
        round_idx: int,
        all_nodes: List[Dict],
        all_edges: List[Dict],
    ) -> StudyRoundResult:
        # Step 1: Select subgraph
        selection = self.selector.select(all_nodes, all_edges)
        entities = selection["entities"]
        relations = selection["relations"]
        strategy = selection["strategy"]

        result = StudyRoundResult(round_idx=round_idx, strategy=strategy)

        if not entities:
            return result

        # Step 2: P1 — Analyze subgraph
        analysis = await self.question_gen.analyze_subgraph(entities, relations)

        # Step 3: P2 — Generate questions
        questions = await self.question_gen.generate_questions(
            strategy, analysis, entities, relations,
        )
        result.questions_asked = len(questions)

        if not questions:
            return result

        # Step 4: P3 — Answer each question using KG retrieval
        qa_records = []
        for q in questions:
            question_text = q.get("question", "")
            if not question_text:
                continue
            qa_record = await self._answer_question(q)
            qa_records.append(qa_record)
            if qa_record.get("answerable"):
                result.questions_answered += 1

        if not qa_records:
            return result

        # Step 5: P4 — Knowledge synthesis
        synthesis = await self._synthesize_knowledge(qa_records)
        result.synthesis_result = synthesis
        result.new_edges_proposed = len(synthesis.get("new_edges", []))
        result.entity_updates_proposed = len(synthesis.get("entity_updates", []))
        result.contradictions_found = len(synthesis.get("contradiction_flags", []))

        # Write back to KG
        if synthesis:
            try:
                await self.kg_write_func(synthesis)
            except Exception as exc:
                _log.error("[self_study] writeback failed: %s", exc)

        # Step 6: P5 — Experience extraction
        session_record = {
            "round": round_idx,
            "strategy": strategy,
            "questions": questions,
            "qa_records": qa_records,
            "synthesis": synthesis,
            "subgraph_stats": {
                "entity_count": len(entities),
                "relation_count": len(relations),
            },
        }
        exp_result = await self.experience_mgr.extract_experiences(
            session_record, self.llm_func,
        )
        for cat in ("retrieval_experiences", "reasoning_experiences",
                     "failure_experiences", "structural_experiences",
                     "meta_experiences"):
            result.experiences_extracted += len(exp_result.get(cat, []))

        return result

    async def _answer_question(self, question: Dict) -> Dict[str, Any]:
        """P3: Use KG retrieval to answer a self-study question."""
        question_text = question.get("question", "")
        question_type = question.get("question_type", "general")

        try:
            retrieval = await self.kg_query_func(question_text)
        except Exception as exc:
            _log.warning("[self_study] KG query failed: %s", exc)
            retrieval = {}

        prompt = ANSWER_AND_REASON_PROMPT.format(
            question=question_text,
            question_type=question_type,
            retrieved_entities=json.dumps(
                retrieval.get("entities", [])[:20], ensure_ascii=False,
            ),
            retrieved_relations=json.dumps(
                retrieval.get("relations", [])[:20], ensure_ascii=False,
            ),
            retrieved_chunks=json.dumps(
                retrieval.get("chunks", [])[:5], ensure_ascii=False,
            ),
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            if isinstance(parsed, dict):
                parsed["original_question"] = question
                return parsed
        except Exception as exc:
            _log.warning("[self_study] P3 LLM call failed: %s", exc)

        return {
            "answer": "",
            "confidence": 0.0,
            "answerable": False,
            "original_question": question,
        }

    async def _synthesize_knowledge(
        self, qa_records: List[Dict],
    ) -> Dict[str, Any]:
        """P4: Synthesize new knowledge from Q&A records."""
        prompt = KNOWLEDGE_SYNTHESIS_PROMPT.format(
            all_qa_records_json=json.dumps(
                qa_records, ensure_ascii=False, indent=2,
            )[:15000],
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            if isinstance(parsed, dict):
                self._filter_by_confidence(parsed)
                return parsed
        except Exception as exc:
            _log.error("[self_study] P4 LLM call failed: %s", exc)

        return {}

    def _filter_by_confidence(self, synthesis: Dict) -> None:
        """Remove low-confidence items from synthesis output."""
        threshold = self.config.writeback_confidence_threshold
        if "new_edges" in synthesis:
            synthesis["new_edges"] = [
                e for e in synthesis["new_edges"]
                if float(e.get("confidence", 0)) >= threshold
                and len(e.get("evidence_chain", [])) >= 1
            ]
        if "summary_nodes" in synthesis:
            synthesis["summary_nodes"] = [
                s for s in synthesis["summary_nodes"]
                if len(s.get("member_entities", [])) >= 3
            ]

    async def run_refinement(self) -> int:
        """Run P6 experience refinement (call periodically)."""
        return await self.experience_mgr.refine_experiences(self.llm_func)
