from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import json
import logging

_log = logging.getLogger("docthinker.cognitive")

class PotentialLink(BaseModel):
    kind: str = "entity"
    entity_id: str
    name: str
    entity_type: str
    confidence: float = 0.0
    matched_by: str

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str = ""
    confidence: float = 0.0
    aliases: List[str] = []
    attributes: Dict[str, Any] = {}

class ExtractedRelation(BaseModel):
    source: str
    target: str
    relation: str
    description: str = ""
    confidence: float = 0.0
    evidence: str = ""

class CognitiveInsight(BaseModel):
    summary: str
    key_points: List[str] = []
    concepts: List[str]
    potential_links: List[PotentialLink]
    reasoning: str
    action_items: List[str] = []
    entities: List[ExtractedEntity] = []
    relations: List[ExtractedRelation] = []
    inferred_relations: List[ExtractedRelation] = []
    hypotheses: List[str] = []

class CognitiveProcessor:
    def __init__(self, llm_func, embedding_func, knowledge_graph=None):
        """
        Initialize the Cognitive Processor.
        
        Args:
            llm_func: Async function to call LLM
            embedding_func: Function to get embeddings
            knowledge_graph: Optional reference to the global knowledge graph for association
        """
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self.kg = knowledge_graph

    async def process(self, content: str, source_type: str = "text") -> CognitiveInsight:
        """
        Process an incoming information signal through the cognitive pipeline.
        
        Steps:
        1. Understand: Extract gist and concepts.
        2. Associate: Check against existing knowledge (if available).
        3. Reason: Formulate insights and next steps.
        """
        _log.info("Thinking about incoming %s...", source_type)
        
        # Step 1 & 3: Understand and Reason (Combined for efficiency)
        insight = await self._analyze_content(content, source_type)
        
        if self.kg:
            try:
                insight.potential_links = await self._associate(insight)
            except Exception as e:
                _log.warning("Cognitive associate failed: %s", e)
            
        return insight

    async def _associate(self, insight: CognitiveInsight) -> List[PotentialLink]:
        if not hasattr(self.kg, "search_entities"):
            return []

        seen: set[str] = set()
        links: List[PotentialLink] = []
        for concept in (insight.concepts or []):
            concept_text = (concept or "").strip()
            if not concept_text:
                continue
            entities = self.kg.search_entities(concept_text, limit=10)
            for entity in entities:
                if entity.id in seen:
                    continue
                seen.add(entity.id)
                links.append(
                    PotentialLink(
                        entity_id=entity.id,
                        name=entity.name,
                        entity_type=getattr(entity, "type", "unknown"),
                        confidence=float(getattr(entity, "confidence", 0.0) or 0.0),
                        matched_by=concept_text,
                    )
                )
        return links

    def _parse_json(self, response: str) -> Dict[str, Any]:
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3]
        elif json_str.startswith("```"):
            json_str = json_str[3:-3]
        return json.loads(json_str)

    async def _extract_understanding(self, content: str, source_type: str) -> Dict[str, Any]:
        prompt = f"""
You are a cognitive analyst. Distill the input into a compact understanding.

Input ({source_type}):
{content[:5000]}...

Return JSON:
{{
  "summary": "...",
  "key_points": ["..."],
  "concepts": ["..."],
  "reasoning": "...",
  "action_items": ["..."]
}}
"""
        response = await self.llm_func(prompt)
        return self._parse_json(response)

    async def _extract_entities(self, content: str, concepts: List[str]) -> List[ExtractedEntity]:
        prompt = f"""
Extract entities with type, description, confidence, aliases, attributes.
Concept hints: {", ".join(concepts[:20])}

Input:
{content[:5000]}...

Return JSON:
{{
  "entities": [
    {{"name": "...", "entity_type": "...", "description": "...", "confidence": 0.0, "aliases": [], "attributes": {{}}}}
  ]
}}
"""
        response = await self.llm_func(prompt)
        data = self._parse_json(response)
        return [ExtractedEntity(**item) for item in data.get("entities", []) if isinstance(item, dict)]

    async def _extract_relations(
        self, content: str, entities: List[ExtractedEntity]
    ) -> List[ExtractedRelation]:
        entity_names = [e.name for e in entities]
        prompt = f"""
Extract explicit relations from the input using provided entities.
Entities: {", ".join(entity_names[:30])}

Input:
{content[:5000]}...

Return JSON:
{{
  "relations": [
    {{"source": "...", "target": "...", "relation": "...", "description": "...", "confidence": 0.0, "evidence": "..."}}
  ]
}}
"""
        response = await self.llm_func(prompt)
        data = self._parse_json(response)
        return [ExtractedRelation(**item) for item in data.get("relations", []) if isinstance(item, dict)]

    async def _infer_relations(
        self,
        summary: str,
        concepts: List[str],
        entities: List[ExtractedEntity],
        relations: List[ExtractedRelation],
    ) -> tuple[List[ExtractedRelation], List[str]]:
        entity_names = [e.name for e in entities]
        rel_pairs = [f"{r.source}->{r.relation}->{r.target}" for r in relations]
        prompt = f"""
Given the understanding, infer plausible but not explicit relations. Also list hypotheses.

Summary: {summary}
Concepts: {", ".join(concepts[:20])}
Entities: {", ".join(entity_names[:30])}
Explicit Relations: {", ".join(rel_pairs[:30])}

Return JSON:
{{
  "inferred_relations": [
    {{"source": "...", "target": "...", "relation": "...", "description": "...", "confidence": 0.0, "evidence": "..."}}
  ],
  "hypotheses": ["..."]
}}
"""
        response = await self.llm_func(prompt)
        data = self._parse_json(response)
        inferred = [
            ExtractedRelation(**item)
            for item in data.get("inferred_relations", [])
            if isinstance(item, dict)
        ]
        hypotheses = data.get("hypotheses", [])
        return inferred, hypotheses

    async def _analyze_content(self, content: str, source_type: str) -> CognitiveInsight:
        import time as _time
        import logging
        _log = logging.getLogger("docthinker.cognitive")
        _t0 = _time.perf_counter()

        try:
            _t1 = _time.perf_counter()
            understanding = await self._extract_understanding(content, source_type)
            _log.info("[cognitive T+%.2fs] _extract_understanding done (%.2fs)",
                      _time.perf_counter() - _t0, _time.perf_counter() - _t1)
            summary = understanding.get("summary", "")
            key_points = understanding.get("key_points", [])
            concepts = understanding.get("concepts", [])
            reasoning = understanding.get("reasoning", "")
            action_items = understanding.get("action_items", [])

            # Run entity and relation extraction in parallel
            _t2 = _time.perf_counter()
            entities_task = asyncio.create_task(self._extract_entities(content, concepts))
            relations_task = asyncio.create_task(self._extract_relations(content, []))
            entities, relations_raw = await asyncio.gather(entities_task, relations_task)
            _log.info("[cognitive T+%.2fs] parallel extract done (%.2fs) | %d entities, %d relations",
                      _time.perf_counter() - _t0, _time.perf_counter() - _t2,
                      len(entities), len(relations_raw))

            _t4 = _time.perf_counter()
            inferred_relations, hypotheses = await self._infer_relations(
                summary, concepts, entities, relations_raw
            )
            _log.info("[cognitive T+%.2fs] _infer_relations done (%.2fs) | %d inferred",
                      _time.perf_counter() - _t0, _time.perf_counter() - _t4, len(inferred_relations))
            _log.info("[cognitive T+%.2fs] TOTAL (4 LLM calls)", _time.perf_counter() - _t0)

            return CognitiveInsight(
                summary=summary,
                key_points=key_points,
                concepts=concepts,
                potential_links=[],
                reasoning=reasoning,
                action_items=action_items,
                entities=entities,
                relations=relations_raw,
                inferred_relations=inferred_relations,
                hypotheses=hypotheses,
            )
        except Exception as e:
            _log.warning("Cognitive analysis failed: %s", e)
            return CognitiveInsight(
                summary="Analysis failed",
                key_points=[],
                concepts=[],
                potential_links=[],
                reasoning=f"Error: {str(e)}",
                action_items=[],
                entities=[],
                relations=[],
                inferred_relations=[],
                hypotheses=[],
            )
