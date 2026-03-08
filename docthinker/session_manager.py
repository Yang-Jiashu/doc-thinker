import json
import os
import shutil
import uuid
import logging
import re
import sqlite3
from threading import RLock
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from docthinker.knowledge_base_storage import KnowledgeBaseStorage
from docthinker.knowledge_base import KnowledgeEntry
from docthinker.core import DocThinker
from docthinker.config import DocThinkerConfig

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage chat sessions, per-session storages, and history."""

    SESSION_ID_PATTERN = re.compile(r"^#\d{5}$")
    UUID_DIR_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    def __init__(self, base_storage_path: str = "./data/_system", data_root_path: Optional[str] = None):
        self.base_storage_path = Path(base_storage_path).expanduser().resolve()
        self.sessions_dir = self.base_storage_path / "sessions"
        if data_root_path:
            self.data_root = Path(data_root_path).expanduser().resolve()
        else:
            self.data_root = self._infer_data_root(self.base_storage_path)
        self._session_rag_cache: Dict[str, DocThinker] = {}
        self._session_rag_lock = RLock()

        self.base_storage_path.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._migrate_from_legacy_rag_storage()
        self.kb_storage = KnowledgeBaseStorage(str(self.base_storage_path / "knowledge_base.db"))
        self._migrate_sessions_to_numbered_format()
        self._cleanup_legacy_uuid_dirs()

    @staticmethod
    def _format_session_id(number: int) -> str:
        return f"#{number:05d}"

    @classmethod
    def _parse_session_number(cls, session_id: str) -> Optional[int]:
        if not session_id:
            return None
        sid = str(session_id).strip()
        if cls.SESSION_ID_PATTERN.match(sid):
            try:
                return int(sid[1:])
            except Exception:
                return None
        if sid.startswith("#") and sid[1:].isdigit():
            try:
                return int(sid[1:])
            except Exception:
                return None
        if sid.isdigit():
            try:
                return int(sid)
            except Exception:
                return None
        return None

    @classmethod
    def _canonical_session_id(cls, session_id: str) -> str:
        sid = str(session_id or "").strip()
        n = cls._parse_session_number(sid)
        if n is None:
            return sid
        return cls._format_session_id(n)

    @staticmethod
    def _extract_session_id_from_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        if name.startswith("session_"):
            return name[len("session_") :]
        return name

    @staticmethod
    def _infer_data_root(base_storage_path: Path) -> Path:
        if base_storage_path.name == "_system" and base_storage_path.parent.name == "data":
            return base_storage_path.parent
        if base_storage_path.name == "rag_storage_api":
            return base_storage_path.parent / "data"
        if base_storage_path.parent.name == "data":
            return base_storage_path.parent
        return base_storage_path.parent / "data"

    def _find_legacy_rag_root(self) -> Optional[Path]:
        candidates: List[Path] = []
        try:
            project_root = Path(__file__).resolve().parents[1]
            candidates.append(project_root / "rag_storage_api")
        except Exception:
            pass
        candidates.extend(
            [
                Path.cwd() / "rag_storage_api",
                self.base_storage_path.parent / "rag_storage_api",
                self.base_storage_path.parent.parent / "rag_storage_api",
            ]
        )
        checked = set()
        for cand in candidates:
            try:
                path = cand.expanduser().resolve()
            except Exception:
                continue
            if path in checked:
                continue
            checked.add(path)
            if path == self.base_storage_path:
                continue
            if path.exists() and path.is_dir():
                return path
        return None

    def _merge_episode_payload(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(existing) if isinstance(existing, dict) else {}
        existing_eps = out.get("episodes")
        if not isinstance(existing_eps, list):
            existing_eps = []
        incoming_eps = incoming.get("episodes") if isinstance(incoming, dict) else []
        if not isinstance(incoming_eps, list):
            incoming_eps = []

        by_id: Dict[str, Dict[str, Any]] = {}
        for item in existing_eps + incoming_eps:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("episode_id") or "")
            if not eid:
                continue
            by_id[eid] = item
        out["episodes"] = list(by_id.values())
        return out

    def _migrate_legacy_memory_files(self, legacy_root: Path) -> None:
        legacy_episodes = legacy_root / "episodes.json"
        if not legacy_episodes.exists():
            return

        try:
            payload = json.loads(legacy_episodes.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse legacy episodes.json, skipping migration.")
            return

        episodes = payload.get("episodes") if isinstance(payload, dict) else []
        if not isinstance(episodes, list) or not episodes:
            return

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        orphans: List[Dict[str, Any]] = []
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            sid = str(ep.get("session_id") or "").strip()
            if self.SESSION_ID_PATTERN.match(sid):
                grouped.setdefault(sid, []).append(ep)
            else:
                orphans.append(ep)

        for sid, values in grouped.items():
            defaults = self._build_default_paths(sid)
            talk_dir = Path(defaults["talk_dir"])
            talk_dir.mkdir(parents=True, exist_ok=True)
            target = talk_dir / "episodes.json"
            existing_payload: Dict[str, Any] = {}
            if target.exists():
                try:
                    existing_payload = json.loads(target.read_text(encoding="utf-8"))
                except Exception:
                    existing_payload = {}
            merged = self._merge_episode_payload(existing_payload, {"episodes": values})
            target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

        memory_graph = legacy_root / "memory_graph.json"
        episode_vectors = legacy_root / "episode_vectors.json"
        if len(grouped) == 1:
            sid = next(iter(grouped.keys()))
            talk_dir = Path(self._build_default_paths(sid)["talk_dir"])
            talk_dir.mkdir(parents=True, exist_ok=True)
            if memory_graph.exists():
                shutil.copy2(str(memory_graph), str(talk_dir / "memory_graph.json"))
            if episode_vectors.exists():
                shutil.copy2(str(episode_vectors), str(talk_dir / "episode_vectors.json"))
        elif memory_graph.exists() or episode_vectors.exists():
            legacy_backup = self.base_storage_path / "legacy_memory"
            legacy_backup.mkdir(parents=True, exist_ok=True)
            if memory_graph.exists():
                shutil.copy2(str(memory_graph), str(legacy_backup / "memory_graph.json"))
            if episode_vectors.exists():
                shutil.copy2(str(episode_vectors), str(legacy_backup / "episode_vectors.json"))

        if orphans:
            legacy_backup = self.base_storage_path / "legacy_memory"
            legacy_backup.mkdir(parents=True, exist_ok=True)
            orphan_path = legacy_backup / "episodes_orphan.json"
            orphan_path.write_text(
                json.dumps({"episodes": orphans}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _migrate_from_legacy_rag_storage(self) -> None:
        legacy_root = self._find_legacy_rag_root()
        if not legacy_root:
            return

        target_db = self.base_storage_path / "knowledge_base.db"
        legacy_db = legacy_root / "knowledge_base.db"
        if legacy_db.exists() and not target_db.exists():
            try:
                shutil.copy2(str(legacy_db), str(target_db))
            except Exception as e:
                logger.warning(f"Failed to copy legacy knowledge_base.db: {e}")

        legacy_sessions = legacy_root / "sessions"
        if legacy_sessions.exists():
            for session_dir in legacy_sessions.iterdir():
                if not session_dir.is_dir():
                    continue
                sid = session_dir.name
                target_knowledge = self.data_root / sid / "knowledge"
                try:
                    self._merge_or_move_dir(session_dir, target_knowledge)
                except Exception as e:
                    logger.warning(f"Failed to migrate legacy session dir {session_dir}: {e}")

        legacy_kbs = legacy_root / "knowledge_bases"
        if legacy_kbs.exists():
            backup_kbs = self.base_storage_path / "legacy_knowledge_bases"
            try:
                self._merge_or_move_dir(legacy_kbs, backup_kbs)
            except Exception as e:
                logger.warning(f"Failed to migrate legacy knowledge_bases: {e}")

        try:
            self._migrate_legacy_memory_files(legacy_root)
        except Exception as e:
            logger.warning(f"Failed to migrate legacy memory files: {e}")

    def _merge_or_move_dir(self, source: Path, target: Path) -> None:
        if not source.exists():
            return
        if source.resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.move(str(source), str(target))
            return
        for item in source.iterdir():
            dst = target / item.name
            if item.is_dir():
                if dst.exists() and dst.is_dir():
                    self._merge_or_move_dir(item, dst)
                elif dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                    shutil.move(str(item), str(dst))
                else:
                    shutil.move(str(item), str(dst))
            else:
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(item), str(dst))
        try:
            source.rmdir()
        except Exception:
            pass

    def _migrate_sessions_to_numbered_format(self) -> None:
        db_path = Path(self.kb_storage.db_path)
        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, metadata, created_at
                FROM knowledge_bases
                WHERE type = 'session'
                ORDER BY datetime(created_at) ASC, id ASC
                """
            )
            rows = cur.fetchall()
            if not rows:
                return

            planned: List[Dict[str, Any]] = []
            for index, row in enumerate(rows, start=1):
                row_id = int(row["id"])
                name = str(row["name"] or "")
                try:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                except Exception:
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}

                old_sid = str(metadata.get("session_id") or self._extract_session_id_from_name(name))
                if not old_sid:
                    old_sid = self._format_session_id(index)

                new_sid = self._format_session_id(index)
                new_name = f"session_{new_sid}"
                defaults = self._build_default_paths(new_sid)
                new_meta = dict(metadata)
                new_meta.update(defaults)
                new_meta["session_id"] = new_sid
                new_meta["code_dir"] = str(Path(defaults["data_dir"]) / "code")

                planned.append(
                    {
                        "row_id": row_id,
                        "old_sid": old_sid,
                        "new_sid": new_sid,
                        "new_name": new_name,
                        "old_meta": metadata,
                        "new_meta": new_meta,
                    }
                )

            for item in planned:
                old_meta = item["old_meta"]
                old_sid = item["old_sid"]
                new_sid = item["new_sid"]
                defaults_old = self._build_default_paths(old_sid)
                defaults_new = self._build_default_paths(new_sid)

                legacy_work = Path(str(old_meta.get("path") or "")).expanduser()
                legacy_data = Path(str(old_meta.get("data_dir") or "")).expanduser()

                for src, dst in [
                    (legacy_work, Path(defaults_new["path"])),
                    (Path(defaults_old["path"]), Path(defaults_new["path"])),
                    (self.sessions_dir / old_sid, Path(defaults_new["path"])),
                    (legacy_data, Path(defaults_new["data_dir"])),
                    (Path(defaults_old["data_dir"]), Path(defaults_new["data_dir"])),
                ]:
                    try:
                        self._merge_or_move_dir(src, dst)
                    except Exception:
                        continue

                self._ensure_session_dirs(item["new_meta"])

            for item in planned:
                cur.execute(
                    "UPDATE knowledge_bases SET name = ?, metadata = ? WHERE id = ?",
                    (item["new_name"], json.dumps(item["new_meta"], ensure_ascii=False), item["row_id"]),
                )
            conn.commit()
        finally:
            conn.close()

    def _cleanup_legacy_uuid_dirs(self) -> None:
        """Best-effort cleanup for old UUID-named session folders."""
        for root in (self.data_root, self.sessions_dir):
            try:
                if not root.exists():
                    continue
                for entry in root.iterdir():
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    if self.SESSION_ID_PATTERN.match(name):
                        continue
                    if self.UUID_DIR_PATTERN.match(name):
                        try:
                            shutil.rmtree(entry)
                        except Exception:
                            pass
            except Exception:
                continue

    def _next_session_id(self) -> str:
        numbers = set()
        try:
            for kb in self.kb_storage.list_knowledge_bases(kb_type="session"):
                sid = str((kb.metadata or {}).get("session_id", ""))
                n = self._parse_session_number(sid)
                if n is not None and n > 0:
                    numbers.add(n)
        except Exception:
            pass

        try:
            for p in self.data_root.iterdir():
                if not p.is_dir():
                    continue
                n = self._parse_session_number(p.name)
                if n is not None and n > 0:
                    numbers.add(n)
        except Exception:
            pass

        n = 1
        while n in numbers:
            n += 1
        return self._format_session_id(n)

    def _build_default_paths(self, session_id: str) -> Dict[str, str]:
        data_dir = self.data_root / session_id
        content_dir = data_dir / "content"
        talk_dir = data_dir / "talk"
        code_dir = data_dir / "code"
        knowledge_dir = data_dir / "knowledge"
        talk_file = talk_dir / "talk.json"
        work_dir = knowledge_dir
        return {
            "path": str(work_dir),
            "data_dir": str(data_dir),
            "content_dir": str(content_dir),
            "talk_dir": str(talk_dir),
            "code_dir": str(code_dir),
            "knowledge_dir": str(knowledge_dir),
            "talk_file": str(talk_file),
        }

    def _ensure_session_dirs(self, metadata: Dict[str, Any]) -> None:
        for key in ["path", "data_dir", "content_dir", "talk_dir", "code_dir", "knowledge_dir"]:
            p = metadata.get(key)
            if p:
                Path(p).mkdir(parents=True, exist_ok=True)

        talk_file = metadata.get("talk_file")
        if not talk_file:
            return

        talk_path = Path(talk_file)
        talk_path.parent.mkdir(parents=True, exist_ok=True)
        if talk_path.exists():
            return

        now_iso = datetime.now().isoformat()
        payload = {
            "session_id": metadata.get("session_id"),
            "title": metadata.get("title", "New Chat"),
            "created_at": now_iso,
            "updated_at": now_iso,
            "messages": [],
        }
        talk_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_talk_message(self, session_id: str, role: str, content: str, ts: datetime, entry_id: str) -> None:
        session = self.get_session(session_id)
        if not session:
            return

        metadata = session.get("metadata") or {}
        talk_file = metadata.get("talk_file") or self._build_default_paths(session_id)["talk_file"]
        talk_path = Path(talk_file)
        talk_path.parent.mkdir(parents=True, exist_ok=True)

        if talk_path.exists():
            try:
                payload = json.loads(talk_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        payload.setdefault("session_id", session_id)
        payload.setdefault("title", session.get("title", "New Chat"))
        payload.setdefault("created_at", ts.isoformat())
        payload.setdefault("messages", [])

        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = []
            payload["messages"] = messages

        messages.append(
            {
                "id": entry_id,
                "role": role,
                "content": content,
                "timestamp": ts.isoformat(),
            }
        )
        payload["updated_at"] = ts.isoformat()
        talk_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_session_content_dir(self, session_id: str) -> Path:
        session_id = self._canonical_session_id(session_id)
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        metadata = session.get("metadata") or {}
        content_dir = Path(metadata.get("content_dir") or self._build_default_paths(session_id)["content_dir"])
        content_dir.mkdir(parents=True, exist_ok=True)
        return content_dir

    def get_session_code_dir(self, session_id: str) -> Path:
        session_id = self._canonical_session_id(session_id)
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        metadata = session.get("metadata") or {}
        code_dir = Path(metadata.get("code_dir") or self._build_default_paths(session_id)["code_dir"])
        code_dir.mkdir(parents=True, exist_ok=True)
        return code_dir

    def get_session_talk_dir(self, session_id: str) -> Path:
        session_id = self._canonical_session_id(session_id)
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        metadata = session.get("metadata") or {}
        talk_dir = Path(metadata.get("talk_dir") or self._build_default_paths(session_id)["talk_dir"])
        talk_dir.mkdir(parents=True, exist_ok=True)
        return talk_dir

    def allocate_session_file_path(self, session_id: str, filename: str) -> Path:
        session_id = self._canonical_session_id(session_id)
        content_dir = self.get_session_content_dir(session_id)
        base = Path(filename).name or "upload.bin"
        stem = Path(base).stem
        suffix = Path(base).suffix
        candidate = content_dir / base
        if not candidate.exists():
            return candidate

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        idx = 1
        while True:
            candidate = content_dir / f"{stem}_{ts}_{idx}{suffix}"
            if not candidate.exists():
                return candidate
            idx += 1

    def create_session(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new numbered session (#00001, #00002, ...)."""
        session_id = self._next_session_id()
        name = f"session_{session_id}"
        display_title = title or f"Chat {session_id}"
        metadata = {
            "title": display_title,
            "session_id": session_id,
            **self._build_default_paths(session_id),
        }
        self._ensure_session_dirs(metadata)

        kb = self.kb_storage.create_knowledge_base(name=name, kb_type="session", metadata=metadata)

        return {
            "id": session_id,
            "title": display_title,
            "created_at": kb.created_at.isoformat(),
            "path": metadata["path"],
            "talk_dir": metadata["talk_dir"],
            "content_dir": metadata["content_dir"],
            "code_dir": metadata["code_dir"],
            "knowledge_dir": metadata["knowledge_dir"],
            "talk_file": metadata["talk_file"],
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        kbs = self.kb_storage.list_knowledge_bases(kb_type="session")
        sessions: List[Dict[str, Any]] = []

        for kb in kbs:
            metadata = kb.metadata or {}
            sid = str(metadata.get("session_id", kb.name.replace("session_", "")))
            defaults = self._build_default_paths(sid)
            merged = {**defaults, **metadata}
            self._ensure_session_dirs(merged)

            title = merged.get("title", kb.name)
            if title.startswith("session_") or title.startswith("test_session_"):
                title = "Untitled Chat"

            sessions.append(
                {
                    "id": sid,
                    "title": title,
                    "created_at": kb.created_at.isoformat(),
                    "file_count": len([e for e in kb.entries.values() if e.entry_type == "document"]),
                    "content_dir": merged.get("content_dir"),
                    "talk_dir": merged.get("talk_dir"),
                    "code_dir": merged.get("code_dir"),
                    "knowledge_dir": merged.get("knowledge_dir"),
                    "talk_file": merged.get("talk_file"),
                }
            )

        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        canonical_id = self._canonical_session_id(session_id)
        name = f"session_{canonical_id}"
        kb = self.kb_storage.get_knowledge_base(name)
        if not kb and canonical_id != str(session_id).strip():
            kb = self.kb_storage.get_knowledge_base(f"session_{str(session_id).strip()}")
        if not kb:
            return None

        metadata = kb.metadata or {}
        sid = str(metadata.get("session_id", canonical_id))
        defaults = self._build_default_paths(sid)
        merged = {**defaults, **metadata}
        self._ensure_session_dirs(merged)

        return {
            "id": sid,
            "title": merged.get("title", kb.name),
            "created_at": kb.created_at.isoformat(),
            "path": merged.get("path"),
            "data_dir": merged.get("data_dir"),
            "content_dir": merged.get("content_dir"),
            "talk_dir": merged.get("talk_dir"),
            "code_dir": merged.get("code_dir"),
            "knowledge_dir": merged.get("knowledge_dir"),
            "talk_file": merged.get("talk_file"),
            "metadata": merged,
        }

    def update_session(self, session_id: str, title: str) -> bool:
        """Update session title."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        kb = self.kb_storage.get_knowledge_base(name)
        if not kb:
            return False

        if not kb.metadata:
            kb.metadata = {}
        kb.metadata["title"] = title
        return self.kb_storage.update_knowledge_base(kb)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its local files."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        kb = self.kb_storage.get_knowledge_base(name)
        if not kb:
            return False

        with self._session_rag_lock:
            self._session_rag_cache.pop(session_id, None)

        self.kb_storage.delete_knowledge_base(name)

        session_path = self.sessions_dir / session_id
        if session_path.exists():
            shutil.rmtree(session_path)

        data_path = self.data_root / session_id
        if data_path.exists():
            shutil.rmtree(data_path)

        return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Add a chat message to session history with timestamp."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        entry_id = str(uuid.uuid4())
        ts = timestamp or datetime.now()

        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            entry_type="question" if role == "user" else "answer",
            metadata={"role": role, "timestamp": ts.isoformat()},
            created_at=ts,
            updated_at=ts,
        )

        self.kb_storage.add_entry(name, entry)
        try:
            self._append_talk_message(session_id, role, content, ts, entry_id)
        except Exception as e:
            logger.warning(f"Failed to append talk log for session {session_id}: {e}")
        return entry_id

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        entries = self.kb_storage.query_entries(name, "")

        history = []
        for entry in entries:
            if entry.entry_type in ["question", "answer"]:
                ts_iso = entry.created_at.isoformat() if entry.created_at else ""
                ts_float = entry.created_at.timestamp() if entry.created_at else 0.0
                history.append(
                    {
                        "role": entry.metadata.get("role", "user"),
                        "content": entry.content,
                        "created_at": ts_iso,
                        "timestamp": ts_float,
                    }
                )
        history.sort(key=lambda x: x["created_at"])
        return history

    def get_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get files uploaded in a session with processing status."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        entries = self.kb_storage.query_entries(name, "")

        doc_status_map = {}
        try:
            session = self.get_session(session_id)
            session_path = Path((session or {}).get("path", ""))
            status_file = session_path / "kv_store_doc_status.json"
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    doc_status_data = json.load(f)
                    for _, data in doc_status_data.items():
                        if "file_path" in data:
                            fname = os.path.basename(data["file_path"])
                            doc_status_map[fname] = data.get("status", "unknown")
        except Exception as e:
            logger.warning(f"Failed to load doc status for session {session_id}: {e}")

        files = []
        for entry in entries:
            if entry.entry_type == "document":
                filename = entry.metadata.get("filename", "unknown")
                status = doc_status_map.get(filename) or str(
                    entry.metadata.get("status") or "pending"
                )
                files.append(
                    {
                        "id": entry.id,
                        "filename": filename,
                        "summary": entry.metadata.get("summary", ""),
                        "created_at": entry.created_at.isoformat(),
                        "status": status,
                        "file_path": entry.metadata.get("file_path"),
                        "file_size": entry.metadata.get("file_size"),
                        "file_ext": entry.metadata.get("file_ext"),
                    }
                )

        files.sort(key=lambda x: x["created_at"], reverse=True)
        return files

    def set_document_status(
        self,
        session_id: str,
        filename: str,
        status: str,
        message: Optional[str] = None,
    ) -> bool:
        """Update latest uploaded document status for a session by filename."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        entries = self.kb_storage.query_entries(name, "")
        candidates = [
            e
            for e in entries
            if e.entry_type == "document"
            and str((e.metadata or {}).get("filename") or "") == str(filename or "")
        ]
        if not candidates:
            return False
        candidates.sort(key=lambda e: e.created_at or datetime.min, reverse=True)
        entry = candidates[0]
        meta = dict(entry.metadata or {})
        meta["status"] = str(status or "pending")
        if message is not None:
            meta["status_message"] = str(message)
        entry.metadata = meta
        entry.updated_at = datetime.now()
        self.kb_storage.update_entry(name, entry)
        return True

    def add_document_record(
        self,
        session_id: str,
        filename: str,
        content_summary: str = "",
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        file_ext: Optional[str] = None,
    ):
        """Record a file upload in session history."""
        session_id = self._canonical_session_id(session_id)
        name = f"session_{session_id}"
        entry_id = str(uuid.uuid4())

        metadata = {
            "filename": filename,
            "summary": content_summary,
            "status": "pending",
        }
        if file_path:
            metadata["file_path"] = file_path
        if file_size is not None:
            metadata["file_size"] = file_size
        if file_ext:
            metadata["file_ext"] = file_ext

        entry = KnowledgeEntry(
            id=entry_id,
            content=f"Uploaded file: {filename}",
            entry_type="document",
            metadata=metadata,
        )

        self.kb_storage.add_entry(name, entry)

    def get_session_rag(
        self,
        session_id: str,
        config: DocThinkerConfig,
        graphcore_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DocThinker:
        """Get a DocThinker instance for a specific session."""
        session_id = self._canonical_session_id(session_id)
        with self._session_rag_lock:
            cached = self._session_rag_cache.get(session_id)
            if cached is not None:
                return cached

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session_config = DocThinkerConfig()
        for key, value in config.__dict__.items():
            if hasattr(session_config, key):
                setattr(session_config, key, value)

        session_config.working_dir = session["path"]
        rag = DocThinker(config=session_config, graphcore_kwargs=graphcore_kwargs or {})
        with self._session_rag_lock:
            self._session_rag_cache[session_id] = rag
        return rag
