import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

t = int(os.getenv("PDF_VLM_PAGE_THRESHOLD", "15"))
m = os.getenv("PARSE_METHOD", "auto")
print(f"PDF_VLM_PAGE_THRESHOLD = {t}")
print(f"PARSE_METHOD (MinerU)  = {m}")
print(f"TIMEOUT (LLM)          = {os.getenv('TIMEOUT')}s")
print(f"MAX_ASYNC              = {os.getenv('MAX_ASYNC')}")
print()
for pages in [10, 15, 20, 50]:
    if pages <= t:
        print(f"  {pages:>3} pages -> cloud VLM (qwen-vl-max, 16 pages parallel)")
    else:
        print(f"  {pages:>3} pages -> MinerU -m {m} (fast OCR, no local VLM)")
