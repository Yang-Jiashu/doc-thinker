#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


DONE_STATUSES = {
    "processed",
    "completed",
    "complete",
    "done",
    "success",
    "ready",
    "finished",
}


def _is_generic_answer(answer: str) -> bool:
    text = (answer or "").lower()
    keys = [
        "无法",
        "抱歉",
        "请提供更具体",
        "没有给出具体",
        "i don't know",
        "not enough information",
    ]
    return any(k in text for k in keys)


def _print_stage(title: str) -> None:
    print(f"\n[Stage] {title}")


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}


def _poll_file_status(base_url: str, session_id: str, timeout_sec: int) -> List[Dict[str, Any]]:
    started = time.time()
    last_statuses: List[str] = []
    while True:
        url = f"{base_url}/sessions/{requests.utils.quote(session_id, safe='')}/files"
        resp = requests.get(url, timeout=20)
        data = _safe_json(resp)
        files = data.get("files") or []
        statuses = [str(x.get("status") or "").lower() for x in files]
        if statuses != last_statuses:
            print(f"  statuses: {statuses}")
            last_statuses = statuses
        if statuses and all(s in DONE_STATUSES for s in statuses):
            return files
        if time.time() - started > timeout_sec:
            return files
        time.sleep(2)


def _count_unknown_source_nodes(graphml_path: Path) -> int:
    if not graphml_path.exists():
        return 0
    txt = graphml_path.read_text(encoding="utf-8", errors="ignore")
    return txt.count("<data key=\"file_path\">unknown_source</data>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Absolute PDF path for upload test",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Polling timeout for file processing",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    base_url = args.base_url.rstrip("/")

    _print_stage("Create session")
    resp = requests.post(f"{base_url}/sessions", json={"title": "pipeline-debug"}, timeout=20)
    data = _safe_json(resp)
    if resp.status_code != 200:
        raise RuntimeError(f"Create session failed: {resp.status_code} {data}")
    session = (data.get("session") or {})
    sid = str(session.get("id") or "")
    if not sid:
        raise RuntimeError(f"No session id in response: {data}")
    print(f"  session_id: {sid}")

    _print_stage("Upload PDF")
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        form = {"session_id": sid}
        up = requests.post(f"{base_url}/upload", files=files, data=form, timeout=60)
    up_data = _safe_json(up)
    print(f"  status_code: {up.status_code}")
    print(f"  response: {json.dumps(up_data, ensure_ascii=False)}")

    _print_stage("Immediate query right after upload")
    payload = {
        "question": "请分析上传的文件",
        "mode": "hybrid",
        "session_id": sid,
        "enable_rerank": True,
    }
    q1 = requests.post(f"{base_url}/query", json=payload, timeout=120)
    q1_data = _safe_json(q1)
    ans1 = str(q1_data.get("answer") or "")
    print(f"  status_code: {q1.status_code}")
    print(f"  answer_mode: {q1_data.get('answer_mode')}")
    print(f"  answer_head: {ans1[:180]}")

    _print_stage("Poll file processing status")
    files_done = _poll_file_status(base_url, sid, args.timeout_sec)
    print(f"  file_count: {len(files_done)}")

    _print_stage("Query again after processing")
    q2 = requests.post(f"{base_url}/query", json=payload, timeout=120)
    q2_data = _safe_json(q2)
    ans2 = str(q2_data.get("answer") or "")
    print(f"  status_code: {q2.status_code}")
    print(f"  answer_mode: {q2_data.get('answer_mode')}")
    print(f"  answer_head: {ans2[:180]}")

    _print_stage("Fetch KG data")
    kg = requests.get(f"{base_url}/knowledge-graph/data", params={"session_id": sid}, timeout=40)
    kg_data = _safe_json(kg)
    nodes = kg_data.get("nodes") or []
    edges = kg_data.get("edges") or []
    print(f"  nodes: {len(nodes)} | edges: {len(edges)}")
    image_nodes = [n for n in nodes if n.get("is_image_node")]
    print(f"  image_nodes: {len(image_nodes)}")

    session_detail = requests.get(
        f"{base_url}/sessions/{requests.utils.quote(sid, safe='')}",
        timeout=20,
    )
    s_data = _safe_json(session_detail)
    session_obj = s_data.get("session") or {}
    knowledge_dir = Path(str(session_obj.get("knowledge_dir") or ""))
    graphml = knowledge_dir / "graph_chunk_entity_relation.graphml"
    unknown_source_nodes = _count_unknown_source_nodes(graphml)
    print(f"  unknown_source_nodes_in_graphml: {unknown_source_nodes}")

    _print_stage("Diagnosis")
    if _is_generic_answer(ans1):
        print("  [R1] Immediate query is generic. This indicates race with background processing.")
    if files_done and any(str(x.get("status") or "").lower() not in DONE_STATUSES for x in files_done):
        print("  [R2] File status not reaching done. Frontend will appear stuck/failed.")
    if unknown_source_nodes > 0:
        print("  [R3] Graph contains unknown_source nodes (often from chat-turn ingestion or non-file inserts).")
    if len(image_nodes) > 0 and len(nodes) <= len(image_nodes) + 5:
        print("  [R4] PDF parsing is mostly image-driven blocks; text may not be extracted.")
    print("  done.")


if __name__ == "__main__":
    main()
