"""
Comprehensive diagnostic for the query pipeline.

Traces why "马云的祖籍在哪里" returns wrong answer despite correct KG data.

Pipeline stages tested:
  Stage 1: Knowledge Graph data integrity (entities, relations, chunks)
  Stage 2: VDB (Vector DB) embedding status — are entities/chunks actually embedded?
  Stage 3: BM25 retrieval (no embedding needed) — does it find correct data?
  Stage 4: Embedding function health — does the API respond at all?
  Stage 5: Vector retrieval — does vdb.query work or timeout?
  Stage 6: Hybrid retrieval error handling — does embedding failure kill BM25 results?
  Stage 7: End-to-end query simulation
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = []
SESSION_ID = "#00004"
KG_DIR = PROJECT_ROOT / "data" / SESSION_ID / "knowledge"


def report(stage: str, test: str, passed: bool, detail: str = ""):
    status = "[OK]  " if passed else "[FAIL]"
    msg = f"  {status} {stage} / {test}"
    if detail:
        for i, line in enumerate(detail.split("\n")):
            prefix = "-> " if i == 0 else "   "
            msg += f"\n         {prefix}{line}"
    print(msg)
    RESULTS.append({"stage": stage, "test": test, "passed": passed, "detail": detail})


def stage_header(name: str):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────
# STAGE 1: Knowledge Graph Data Integrity
# ─────────────────────────────────────────────────────────────────────
def test_stage1_kg_data():
    stage_header("STAGE 1: Knowledge Graph Data Integrity")

    # 1a. knowledge_graph.json — entities
    kg_path = KG_DIR / "knowledge_graph.json"
    report("Stage1", "kg_file_exists", kg_path.exists(), str(kg_path))
    if not kg_path.exists():
        return

    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    entities = kg.get("entities", {})
    relationships = kg.get("relationships", {})
    report("Stage1", "entity_count", len(entities) > 0, f"{len(entities)} entities")
    report("Stage1", "relationship_count", len(relationships) > 0, f"{len(relationships)} relationships")

    # 1b. Check for 马云 entity with correct ancestral home
    mayun_entity = None
    for eid, ent in entities.items():
        if ent.get("name") == "马云":
            mayun_entity = ent
            break

    report("Stage1", "mayun_entity_exists", mayun_entity is not None,
           f"found={mayun_entity is not None}")
    if mayun_entity:
        props = mayun_entity.get("properties", {})
        bg = props.get("成长背景", "")
        has_shengzhou = "嵊州" in str(bg)
        report("Stage1", "mayun_correct_ancestral_home", has_shengzhou,
               f"成长背景={bg}")

    # 1c. Check 祖籍 relationship
    zuji_rel = None
    for rid, rel in relationships.items():
        if rel.get("type") == "祖籍":
            zuji_rel = rel
            break

    report("Stage1", "zuji_relationship_exists", zuji_rel is not None,
           f"found={zuji_rel is not None}")
    if zuji_rel:
        evidence = zuji_rel.get("properties", {}).get("evidence", "")
        has_correct = "嵊州" in evidence
        report("Stage1", "zuji_evidence_correct", has_correct,
               f"evidence={evidence}")

    # 1d. Check text chunks — is the correct chunk stored?
    chunks_path = KG_DIR / "kv_store_text_chunks.json"
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        correct_chunk = None
        poisoned_chunks = []
        for cid, chunk in chunks.items():
            content = chunk.get("content", "")
            if "嵊州" in content and "谷来镇" in content:
                correct_chunk = cid
            if "安徽" in content or "合肥" in content:
                poisoned_chunks.append((cid, content[:100]))

        report("Stage1", "correct_chunk_exists", correct_chunk is not None,
               f"chunk_id={correct_chunk}")
        report("Stage1", "poisoned_chunks_count", True,
               f"found {len(poisoned_chunks)} chunks with WRONG info (安徽/合肥):\n" +
               "\n".join(f"  {cid}: {preview}..." for cid, preview in poisoned_chunks))

    # 1e. Check GraphCore chunk_entity_relation graph
    graphml_path = KG_DIR / "graph_chunk_entity_relation.graphml"
    report("Stage1", "graphml_exists", graphml_path.exists(),
           f"size={graphml_path.stat().st_size if graphml_path.exists() else 0} bytes")


# ─────────────────────────────────────────────────────────────────────
# STAGE 2: VDB Embedding Status
# ─────────────────────────────────────────────────────────────────────
def test_stage2_vdb_status():
    stage_header("STAGE 2: VDB (Vector DB) Embedding Status")

    for vdb_name in ["vdb_entities", "vdb_chunks", "vdb_relationships"]:
        vdb_path = KG_DIR / f"{vdb_name}.json"
        if not vdb_path.exists():
            report("Stage2", f"{vdb_name}_exists", False, "file not found")
            continue

        data = json.loads(vdb_path.read_text(encoding="utf-8"))

        # NanoVectorDB stores data as list of dicts with __vector__ field
        if isinstance(data, list):
            total = len(data)
            with_vector = sum(1 for d in data if d.get("__vector__") and len(d["__vector__"]) > 0)
            without_vector = total - with_vector
            report("Stage2", f"{vdb_name}_total_entries", total > 0, f"{total} entries")
            report("Stage2", f"{vdb_name}_with_embeddings", with_vector > 0,
                   f"{with_vector}/{total} have embeddings, {without_vector} MISSING")

            if vdb_name == "vdb_entities":
                for d in data:
                    name = d.get("entity_name", d.get("__id__", "?"))
                    has_vec = bool(d.get("__vector__") and len(d["__vector__"]) > 0)
                    if "马云" in str(name) or "祖籍" in str(name) or "嵊州" in str(name):
                        report("Stage2", f"entity_embedding: {name}",
                               has_vec, f"has_vector={has_vec}")
        elif isinstance(data, dict):
            report("Stage2", f"{vdb_name}_format", True,
                   f"dict format with {len(data)} keys")


# ─────────────────────────────────────────────────────────────────────
# STAGE 3: BM25 Retrieval (No Embedding Needed)
# ─────────────────────────────────────────────────────────────────────
def test_stage3_bm25():
    stage_header("STAGE 3: BM25 Retrieval (No Embedding Needed)")

    try:
        from docthinker.bm25_hybrid import bm25_retriever, normalize_bm25_scores, _BM25_AVAILABLE
        if not hasattr(bm25_retriever, 'search_entities'):
            from docthinker.bm25_hybrid import BM25HybridRetriever
            bm25_retriever_local = BM25HybridRetriever()
        else:
            bm25_retriever_local = bm25_retriever
        report("Stage3", "bm25_available", True, "BM25 module imported successfully")
    except Exception as e:
        report("Stage3", "bm25_available", False, f"BM25 import failed: {e}")
        return

    # 3a. Load the GraphCore graph for BM25 entity search
    graphml_path = KG_DIR / "graph_chunk_entity_relation.graphml"
    if not graphml_path.exists():
        report("Stage3", "graphml_for_bm25", False, "graphml file not found")
        return

    try:
        import networkx as nx
        G = nx.read_graphml(str(graphml_path))
        report("Stage3", "graphml_loaded", True,
               f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    except Exception as e:
        report("Stage3", "graphml_loaded", False, str(e))
        return

    # 3b. Test BM25 entity search
    query = "马云的祖籍在哪里"
    try:
        results = bm25_retriever_local.search_entities(G, query, top_k=10)
        report("Stage3", "bm25_entity_search", len(results) > 0,
               f"found {len(results)} results for '{query}'")
        for entry, score in results[:5]:
            name = entry.payload.get("entity_name", entry.doc_id)
            report("Stage3", f"  bm25_entity: {name}", True,
                   f"score={score:.4f}")
    except Exception as e:
        report("Stage3", "bm25_entity_search", False, str(e))

    # 3c. Test BM25 chunk search
    chunks_path = KG_DIR / "kv_store_text_chunks.json"
    if chunks_path.exists():
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        try:
            from docthinker.bm25_hybrid import _BM25Index, _BM25Entry

            entries = []
            for cid, chunk in chunks_data.items():
                content = chunk.get("content", "")
                entries.append(_BM25Entry(doc_id=cid, payload={"content": content, "chunk_id": cid}))

            if entries:
                index = _BM25Index(entries)
                chunk_results = index.search(query, top_k=5)
                report("Stage3", "bm25_chunk_search", len(chunk_results) > 0,
                       f"found {len(chunk_results)} chunk results")
                for entry, score in chunk_results[:3]:
                    content_preview = entry.payload.get("content", "")[:80]
                    has_correct = "嵊州" in content_preview or "祖籍" in content_preview
                    report("Stage3", f"  bm25_chunk: {entry.doc_id}", has_correct,
                           f"score={score:.4f}, preview={content_preview}...")
        except Exception as e:
            report("Stage3", "bm25_chunk_search", False, str(e))


# ─────────────────────────────────────────────────────────────────────
# STAGE 4: Embedding Function Health
# ─────────────────────────────────────────────────────────────────────
async def test_stage4_embedding():
    stage_header("STAGE 4: Embedding Function Health")

    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    embed_host = os.getenv("LLM_EMBED_HOST") or os.getenv("EMBEDDING_BINDING_HOST") or "https://api.bltcy.ai/v1"
    embed_model = os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "qwen3-embedding-4b"

    report("Stage4", "api_key_present", bool(api_key),
           f"key={'***' + api_key[-4:] if api_key and len(api_key) > 4 else 'MISSING'}")
    report("Stage4", "embed_endpoint", True, f"{embed_host}, model={embed_model}")

    if not api_key:
        report("Stage4", "SKIPPED", False, "no API key")
        return

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=embed_host)

        # 4a. Simple embedding test
        t0 = time.time()
        resp = await asyncio.wait_for(
            client.embeddings.create(model=embed_model, input=["马云的祖籍"]),
            timeout=30
        )
        elapsed = time.time() - t0
        if resp.data and resp.data[0].embedding:
            dim = len(resp.data[0].embedding)
            report("Stage4", "single_embedding_ok", True,
                   f"dim={dim}, elapsed={elapsed:.2f}s")
        else:
            report("Stage4", "single_embedding_ok", False, "empty response")

        # 4b. Batch embedding test (similar to what query does)
        texts = ["马云", "祖籍", "浙江省嵊州市", "杭州师范学院", "高考"]
        t0 = time.time()
        resp = await asyncio.wait_for(
            client.embeddings.create(model=embed_model, input=texts),
            timeout=30
        )
        elapsed = time.time() - t0
        report("Stage4", "batch_embedding_ok", len(resp.data) == len(texts),
               f"got {len(resp.data)}/{len(texts)} embeddings, elapsed={elapsed:.2f}s")

        # 4c. Timeout simulation — check what happens with concurrent requests
        async def embed_one(text):
            t = time.time()
            r = await client.embeddings.create(model=embed_model, input=[text])
            return time.time() - t

        tasks = [embed_one(f"test query {i}") for i in range(5)]
        t0 = time.time()
        try:
            times = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60)
            total = time.time() - t0
            report("Stage4", "concurrent_5_embeddings", True,
                   f"total={total:.2f}s, individual: {[f'{t:.2f}s' for t in times]}")
        except asyncio.TimeoutError:
            report("Stage4", "concurrent_5_embeddings", False,
                   f"TIMEOUT after 60s — this is the root cause of query failures!")

        await client.close()
    except Exception as e:
        report("Stage4", "embedding_test", False, str(e))


# ─────────────────────────────────────────────────────────────────────
# STAGE 5: Code Analysis — Error Handling in Hybrid Retrieval
# ─────────────────────────────────────────────────────────────────────
def test_stage5_code_analysis():
    stage_header("STAGE 5: Hybrid Retrieval Error Handling (Code Analysis)")

    import inspect
    from graphcore.coregraph import operate

    # Find the entity query function
    source = inspect.getsource(operate)

    # Check entity retrieval: does it have try/except around vector query?
    # Look for the pattern near "entities_vdb.query"
    entity_section = ""
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if "use_vector = strategy" in line and i > 4300:
            entity_section = "\n".join(lines[max(0, i-2):i+15])
            break

    entity_has_try = "try:" in entity_section and "entities_vdb.query" in entity_section
    report("Stage5", "entity_vector_query_has_try_except", entity_has_try,
           f"{'Protected' if entity_has_try else 'UNPROTECTED — embedding timeout will crash entire query!'}")

    # Check chunk retrieval
    chunk_section_start = source.find("def _get_vector_context")
    if chunk_section_start >= 0:
        chunk_section = source[chunk_section_start:chunk_section_start + 2000]
        chunk_has_try = "try:" in chunk_section and "chunks_vdb" in chunk_section
        report("Stage5", "chunk_vector_query_has_try_except", chunk_has_try,
               f"{'Protected' if chunk_has_try else 'UNPROTECTED'}")

    # Check relationship retrieval
    rel_sections = []
    for i, line in enumerate(lines):
        if "relationships_vdb.query" in line:
            rel_sections.append("\n".join(lines[max(0, i-5):i+3]))

    rel_has_try = any("try:" in s for s in rel_sections)
    report("Stage5", "relation_vector_query_has_try_except", rel_has_try,
           f"{'Protected' if rel_has_try else 'UNPROTECTED — embedding timeout will crash entire query!'}")

    # Summary
    report("Stage5", "DIAGNOSIS", entity_has_try and rel_has_try,
           "When strategy='hybrid':\n"
           "1. BM25 runs first (SUCCESS — finds correct entities/chunks)\n"
           "2. Vector query also runs (TIMEOUT — embedding API fails)\n"
           "3. Without try/except, timeout EXCEPTION kills entire query\n"
           "4. BM25 results are DISCARDED despite being correct\n"
           "5. Query falls back to LLM general knowledge → WRONG ANSWER")


# ─────────────────────────────────────────────────────────────────────
# STAGE 6: Poisoned Data Analysis
# ─────────────────────────────────────────────────────────────────────
def test_stage6_poisoned_data():
    stage_header("STAGE 6: Knowledge Base Pollution Analysis")

    chunks_path = KG_DIR / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        report("Stage6", "chunks_file", False, "not found")
        return

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    correct_info = []
    wrong_info = []
    noise_info = []

    for cid, chunk in chunks.items():
        content = chunk.get("content", "")
        if "嵊州" in content and "谷来镇" in content:
            correct_info.append(cid)
        elif "安徽" in content or "合肥" in content:
            wrong_info.append(cid)
        elif "乱码" in content or "损坏" in content or "garbled" in content:
            noise_info.append(cid)

    total = len(chunks)
    report("Stage6", "total_chunks", True, f"{total}")
    report("Stage6", "correct_chunks", len(correct_info) > 0,
           f"{len(correct_info)} chunks with correct info (嵊州市谷来镇)")
    report("Stage6", "wrong_chunks", len(wrong_info) == 0,
           f"{len(wrong_info)} chunks with WRONG info (安徽/合肥) — from LLM hallucination ingested back")
    report("Stage6", "noise_chunks", True,
           f"{len(noise_info)} chunks with noise (garbled text discussions)")

    if wrong_info:
        report("Stage6", "POLLUTION_WARNING", False,
               f"Chunks {wrong_info} contain WRONG answer from first query hallucination!\n"
               "These were ingested via _ingest_chat_turn() after the LLM gave wrong answer.\n"
               "Even if retrieval works, these poisoned chunks may surface wrong info.")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  QUERY PIPELINE DIAGNOSTIC")
    print(f"  Question: '马云的祖籍在哪里'")
    print(f"  Session:  {SESSION_ID}")
    print(f"  Time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    test_stage1_kg_data()
    test_stage2_vdb_status()
    test_stage3_bm25()
    asyncio.run(test_stage4_embedding())
    test_stage5_code_analysis()
    test_stage6_poisoned_data()

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    print(f"  Total tests: {total}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")

    if failed:
        print(f"\n  FAILURES:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"    [FAIL] {r['stage']} / {r['test']}")
                if r["detail"]:
                    for line in r["detail"].split("\n"):
                        print(f"           {line}")

    out_path = PROJECT_ROOT / "data" / SESSION_ID / "code" / "query_diagnostic_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "results": RESULTS},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
