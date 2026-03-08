import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from docthinker.parser import MineruParser


class _FakeMineruParser(MineruParser):
    def __init__(self, outputs_by_method: Dict[str, List[Dict[str, Any]]]):
        super().__init__()
        self.outputs_by_method = outputs_by_method
        self.run_methods: List[str] = []

    def _run_mineru_command(self, *args, **kwargs):
        method = str(kwargs.get("method") or "auto")
        self.run_methods.append(method)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

    def _read_output_files(self, output_dir, name_without_suff, method="auto"):
        return list(self.outputs_by_method.get(method, [])), ""


class ParserPdfOcrFallbackUnitTest(unittest.TestCase):
    @staticmethod
    def _write_fake_pdf(tmp_dir: str) -> Path:
        path = Path(tmp_dir) / "sample.pdf"
        path.write_bytes(b"%PDF-1.4\n%fake\n")
        return path

    def test_auto_retries_with_ocr_when_no_text(self):
        parser = _FakeMineruParser(
            {
                "auto": [{"type": "image", "text": ""}, {"type": "footer", "text": "1"}],
                "ocr": [{"type": "text", "text": "这是正文内容"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self._write_fake_pdf(tmp)
            out = Path(tmp) / "out"
            result = parser.parse_pdf(pdf, output_dir=str(out), method="auto")

        self.assertEqual(["auto", "ocr"], parser.run_methods)
        self.assertEqual([{"type": "text", "text": "这是正文内容"}], result)

    def test_auto_keeps_auto_when_text_exists(self):
        parser = _FakeMineruParser(
            {
                "auto": [{"type": "text", "text": "已有正文"}],
                "ocr": [{"type": "text", "text": "OCR正文"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self._write_fake_pdf(tmp)
            out = Path(tmp) / "out"
            result = parser.parse_pdf(pdf, output_dir=str(out), method="auto")

        self.assertEqual(["auto"], parser.run_methods)
        self.assertEqual([{"type": "text", "text": "已有正文"}], result)

    def test_non_auto_method_does_not_retry(self):
        parser = _FakeMineruParser(
            {
                "ocr": [{"type": "text", "text": "OCR正文"}],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self._write_fake_pdf(tmp)
            out = Path(tmp) / "out"
            result = parser.parse_pdf(pdf, output_dir=str(out), method="ocr")

        self.assertEqual(["ocr"], parser.run_methods)
        self.assertEqual([{"type": "text", "text": "OCR正文"}], result)


if __name__ == "__main__":
    unittest.main()
