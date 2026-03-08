from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

IMAGE_NODE_COLOR = "#8b5cf6"
EXPANDED_NODE_COLOR = "#FFD700"
DEFAULT_NODE_COLOR = "#3498db"


def _truthy_flag(value: Any) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def is_image_node(node_data: Dict[str, Any]) -> bool:
    entity_type = str(
        node_data.get("entity_type") or node_data.get("type") or ""
    ).strip().lower()
    if entity_type in {"image", "image_asset", "figure", "diagram"}:
        return True

    if _truthy_flag(node_data.get("is_image_node")):
        return True

    source_id = str(node_data.get("source_id") or "").strip().lower()
    return source_id.startswith("image_asset:")


def resolve_graph_node_color(node_data: Dict[str, Any], *, is_expanded: bool) -> str:
    if is_image_node(node_data):
        return IMAGE_NODE_COLOR
    if is_expanded:
        return EXPANDED_NODE_COLOR
    return DEFAULT_NODE_COLOR


def build_image_asset_id(
    *,
    session_id: str,
    doc_id: str,
    source_pdf: str,
    page_idx: int,
    source_image_path: str,
    index: int,
) -> str:
    raw = (
        f"{session_id}|{doc_id}|{source_pdf}|{page_idx}|"
        f"{source_image_path}|{index}"
    )
    return f"img-{hashlib.md5(raw.encode('utf-8')).hexdigest()}"


def extract_image_items(content_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in content_list or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type != "image":
            continue
        img_path = str(item.get("img_path") or "").strip()
        if not img_path:
            continue
        items.append(item)
    return items


class ImageAssetTable:
    """Session-local image asset table persisted under knowledge/multimodal."""

    def __init__(self, knowledge_dir: str | Path):
        self.knowledge_dir = Path(knowledge_dir)
        self.multimodal_dir = self.knowledge_dir / "multimodal"
        self.images_dir = self.multimodal_dir / "images"
        self.table_path = self.multimodal_dir / "image_assets.json"
        self.multimodal_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, Any]]:
        if not self.table_path.exists():
            return []
        try:
            payload = json.loads(self.table_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return []
        output: List[Dict[str, Any]] = []
        for item in records:
            if isinstance(item, dict):
                output.append(dict(item))
        return output

    def save(self, records: List[Dict[str, Any]]) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "records": records,
        }
        self.table_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        current = self.load()
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in current:
            image_id = str(item.get("image_id") or "").strip()
            if image_id:
                by_id[image_id] = item

        for item in records:
            image_id = str(item.get("image_id") or "").strip()
            if not image_id:
                continue
            merged = dict(by_id.get(image_id, {}))
            merged.update(item)
            by_id[image_id] = merged

        merged_records = list(by_id.values())
        self.save(merged_records)
        return merged_records

    def copy_image_to_store(self, source_path: str, image_id: str) -> str:
        src = Path(source_path)
        suffix = src.suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}:
            suffix = ".png"
        target = self.images_dir / f"{image_id}{suffix}"
        if src.exists():
            if not target.exists():
                shutil.copy2(src, target)
        return str(target.resolve())

    @staticmethod
    def _to_float_vector(value: Any) -> Optional[List[float]]:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list) or not value:
            return None
        out: List[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                return None
        return out if out else None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for av, bv in zip(a, b):
            dot += av * bv
            na += av * av
            nb += bv * bv
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-9)

    def select_activated_by_embedding(
        self,
        *,
        query_embedding: Any,
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        qvec = self._to_float_vector(query_embedding)
        if qvec is None:
            return []

        candidates: List[Dict[str, Any]] = []
        for record in self.load():
            emb = self._to_float_vector(record.get("embedding"))
            if emb is None:
                continue
            score = self._cosine_similarity(qvec, emb)
            if score < float(threshold):
                continue
            item = dict(record)
            item["activation_score"] = float(score)
            candidates.append(item)

        candidates.sort(key=lambda x: float(x.get("activation_score") or 0.0), reverse=True)
        if top_k > 0:
            return candidates[:top_k]
        return candidates
