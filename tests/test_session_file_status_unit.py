import tempfile
import unittest
from pathlib import Path

from docthinker.session_manager import SessionManager


class SessionFileStatusUnitTest(unittest.TestCase):
    def test_status_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_storage = Path(tmp) / "_system"
            data_root = Path(tmp) / "data"
            sm = SessionManager(
                base_storage_path=str(base_storage),
                data_root_path=str(data_root),
            )
            sid = sm.create_session("test")["id"]
            sm.add_document_record(
                sid,
                "sample.pdf",
                file_path=str((data_root / sid / "content" / "sample.pdf")),
                file_ext=".pdf",
            )

            files = sm.get_files(sid)
            self.assertEqual(1, len(files))
            self.assertEqual("pending", files[0]["status"])

            self.assertTrue(sm.set_document_status(sid, "sample.pdf", "processing"))
            files = sm.get_files(sid)
            self.assertEqual("processing", files[0]["status"])

            self.assertTrue(sm.set_document_status(sid, "sample.pdf", "processed"))
            files = sm.get_files(sid)
            self.assertEqual("processed", files[0]["status"])


if __name__ == "__main__":
    unittest.main()
