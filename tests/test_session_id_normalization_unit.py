import tempfile
import unittest
from pathlib import Path

from docthinker.session_manager import SessionManager


class SessionIdNormalizationUnitTest(unittest.TestCase):
    def test_short_hash_id_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_storage = Path(tmp) / "_system"
            data_root = Path(tmp) / "data"
            sm = SessionManager(
                base_storage_path=str(base_storage),
                data_root_path=str(data_root),
            )
            created = sm.create_session("test")
            sid = created["id"]
            self.assertEqual("#00001", sid)

            by_short_hash = sm.get_session("#1")
            by_short_num = sm.get_session("1")
            by_full = sm.get_session("#00001")

            self.assertIsNotNone(by_short_hash)
            self.assertIsNotNone(by_short_num)
            self.assertIsNotNone(by_full)
            self.assertEqual(sid, by_short_hash["id"])
            self.assertEqual(sid, by_short_num["id"])
            self.assertEqual(sid, by_full["id"])

            sm.add_message("#1", "user", "hello")
            history = sm.get_history("#00001")
            self.assertEqual(1, len(history))
            self.assertEqual("user", history[0]["role"])


if __name__ == "__main__":
    unittest.main()
