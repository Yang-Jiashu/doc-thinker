"""Quick test for NER extractor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphcore.coregraph.ner_extractor import NERExtractor

ner = NERExtractor(spacy_model="zh_core_web_sm")

text = (
    "马云是阿里巴巴集团的创始人，1964年出生于浙江省杭州市。"
    "他在1999年创办了阿里巴巴公司。"
    "2014年阿里巴巴在纽约证券交易所上市，市值超过2000亿美元。"
    "马云曾担任联合国数字合作高级别小组联合主席。"
)

result = ner.extract_from_text(text, file_path="test.txt")

print(f"Chunks: {len(result['chunks'])}")
print(f"Entities ({len(result['entities'])}):")
for e in result["entities"]:
    print(f"  {e['entity_name']} [{e['entity_type']}]")
print(f"Relations ({len(result['relationships'])}):")
for r in result["relationships"][:8]:
    print(f"  {r['src_id']} -> {r['tgt_id']}")

# Test query entity extraction
q_ents = ner.extract_query_entities("马云的祖籍在哪里")
print(f"\nQuery entities: {q_ents}")

# Test English
ner_en = NERExtractor(spacy_model="en_core_web_sm")
en_text = "Apple was founded by Steve Jobs in Cupertino, California. Tim Cook became CEO in 2011."
en_result = ner_en.extract_from_text(en_text, file_path="test_en.txt")
print(f"\nEnglish entities ({len(en_result['entities'])}):")
for e in en_result["entities"]:
    print(f"  {e['entity_name']} [{e['entity_type']}]")
