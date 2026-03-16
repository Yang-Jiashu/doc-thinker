"""Test extractor minimum text length guard."""

import asyncio
import tempfile
from pathlib import Path
from docthinker.causal.dag import CausalDAG
from docthinker.causal.extractor import CausalExtractor


async def mock_llm(prompt, **kwargs):
    return "{}"


async def test():
    with tempfile.TemporaryDirectory() as td:
        dag = CausalDAG(Path(td) / "dag.json")
        ext = CausalExtractor(llm_func=mock_llm, dag=dag)

        # short text should be skipped
        r = await ext.extract_from_text("hello world", source_id="test")
        assert r.get("skipped") is True, "Short text not skipped"
        print("Short text (11 chars): SKIPPED - OK")

        # 200+ chars should proceed (but mock LLM returns empty)
        r2 = await ext.extract_from_text("x" * 300, source_id="test2")
        assert r2.get("skipped") is None, "Long text was skipped"
        print("Long text (300 chars): PROCESSED - OK")

        print("Min-length check passed!")


if __name__ == "__main__":
    asyncio.run(test())
