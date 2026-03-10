# LinearRAG Module

NER-based entity extraction as a **switchable alternative** to LLM-based extraction.

## How to enable

```bash
# In .env
ENTITY_EXTRACTION_MODE=ner
NER_SPACY_MODEL=zh_core_web_sm
```

## How to disable (default)

```bash
ENTITY_EXTRACTION_MODE=llm
```

## Dependencies

```bash
pip install spacy
python -m spacy download zh_core_web_sm  # Chinese
python -m spacy download en_core_web_sm  # English (fallback)
```

## Architecture

- `ner_extractor.py` — SpacyNER wrapper, produces `custom_kg` dicts for GraphCore
- `test_ner.py` — standalone test
- Original LinearRAG source: `../LinearRAG/`

## Integration point

`graphcore/coregraph/coregraph.py` checks `entity_extraction_mode`:
- `"llm"` → `_process_extract_entities()` (LLM-based, default)
- `"ner"` → `_process_ner_extraction()` (NER-based, zero LLM cost)
