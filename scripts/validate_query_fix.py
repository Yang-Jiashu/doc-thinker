"""
Validation: confirm all fixes are in place.
1. rank_bm25 installed and BM25 retriever works with Chinese
2. Entity/relation vector queries protected by try/except
3. Poisoned chunk removed from knowledge base
4. BM25 tokenizer handles CJK characters
"""

import inspect
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KG_DIR = PROJECT_ROOT / "data" / "#00004" / "knowledge"
RESULTS = []


def report(test: str, passed: bool, detail: str = ""):
    status = "[OK]  " if passed else "[FAIL]"
    print(f"  {status} {test}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")
    RESULTS.append({"test": test, "passed": passed})


def main():
    print("=" * 60)
    print("  QUERY FIX VALIDATION")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Fix 1: BM25 installed
    print("\n  --- Fix 1: BM25 Availability ---")
    try:
        import rank_bm25
        report("rank_bm25 installed", True)
    except ImportError:
        report("rank_bm25 installed", False)

    try:
        from docthinker.bm25_hybrid import bm25_retriever, BM25Entry, _BM25Index
        report("BM25 module imported", True)
    except Exception as e:
        report("BM25 module imported", False, str(e))

    # Fix 4: Tokenizer handles CJK
    print("\n  --- Fix 4: BM25 CJK Tokenization ---")
    try:
        from docthinker.bm25_hybrid import _tokenize
        en_tokens = _tokenize("Jack Ma 1984")
        cn_tokens = _tokenize("\u9a6c\u4e91\u7684\u7956\u7c4d\u5728\u54ea\u91cc")  # 马云的祖籍在哪里
        report("English tokenization works", len(en_tokens) >= 2,
               f"tokens={en_tokens}")
        report("Chinese tokenization works", len(cn_tokens) >= 4,
               f"token count={len(cn_tokens)} (each CJK char is a token)")

        # Test BM25 search with Chinese
        from docthinker.bm25_hybrid import _BM25Index, BM25Entry
        entries = [
            BM25Entry(doc_id="e1",
                      text="\u9a6c\u4e91 \u7956\u7c4d\u6d59\u6c5f\u7701\u5d4a\u5dde\u5e02",
                      payload={"entity_name": "\u9a6c\u4e91"}),
            BM25Entry(doc_id="e2",
                      text="\u676d\u5dde\u5e08\u8303\u5b66\u9662 \u5916\u8bed\u7cfb",
                      payload={"entity_name": "\u676d\u5dde\u5e08\u8303\u5b66\u9662"}),
            BM25Entry(doc_id="e3",
                      text="Jack Ma entrepreneur Alibaba",
                      payload={"entity_name": "Jack Ma"}),
        ]
        idx = _BM25Index(entries)
        results = idx.search("\u9a6c\u4e91\u7684\u7956\u7c4d", top_k=3)
        top_entity = results[0][0].payload["entity_name"] if results else None
        report("BM25 Chinese search finds correct entity",
               top_entity == "\u9a6c\u4e91",
               f"top result: {top_entity}")
    except Exception as e:
        report("BM25 tokenization test", False, str(e))

    # Fix 2: try/except protection
    print("\n  --- Fix 2: Vector Query Protection ---")
    from graphcore.coregraph import operate
    source = inspect.getsource(operate)
    lines = source.split("\n")

    entity_protected = False
    relation_protected = False
    for i, line in enumerate(lines):
        if "entities_vdb.query" in line:
            context = "\n".join(lines[max(0, i-5):i+1])
            if "try:" in context:
                entity_protected = True
        if "relationships_vdb.query" in line:
            context = "\n".join(lines[max(0, i-5):i+1])
            if "try:" in context:
                relation_protected = True

    report("Entity vector query protected by try/except", entity_protected)
    report("Relation vector query protected by try/except", relation_protected)

    # Fix 3: Poisoned data
    print("\n  --- Fix 3: Poisoned Data Removed ---")
    chunks_path = KG_DIR / "kv_store_text_chunks.json"
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        poisoned = [cid for cid, c in chunks.items()
                    if "\u5b89\u5fbd" in c.get("content", "")
                    or "\u5408\u80a5" in c.get("content", "")]
        report("No poisoned chunks remain", len(poisoned) == 0)

        correct = [cid for cid, c in chunks.items()
                   if "\u5d4a\u5dde" in c.get("content", "")
                   and "\u8c37\u6765\u9547" in c.get("content", "")]
        report("Correct chunk preserved", len(correct) > 0)

    # Summary
    print(f"\n{'='*60}")
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    all_ok = failed == 0
    print(f"  Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
