"""
TXT Upload + Query End-to-End Timing Test
==========================================
Tests the full pipeline: upload → wait for processing → query in all 3 modes.
Measures timing at each stage to identify bottlenecks.

Usage:
    python scripts/test_txt_upload_timing.py
    python scripts/test_txt_upload_timing.py --file path/to/your.txt
    python scripts/test_txt_upload_timing.py --session "#00035"
    python scripts/test_txt_upload_timing.py --skip-upload --session "#00035"  # only test queries
"""
import argparse
import json
import sys
import time
import requests

API_BASE = "http://127.0.0.1:8000/api/v1"
FLASK_BASE = "http://127.0.0.1:5000/api/v1"

SAMPLE_TXT = """人工智能的发展历程

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。
从1956年达特茅斯会议正式提出"人工智能"概念以来，AI经历了多个发展阶段。

早期阶段（1956-1974）：
在这个时期，研究者们充满乐观。Alan Turing提出了著名的"图灵测试"。
John McCarthy创造了"人工智能"这个术语。早期的程序如ELIZA和SHRDLU展示了自然语言处理的可能性。

知识工程时期（1980-1987）：
专家系统成为AI的重要商业应用。MYCIN系统能够诊断血液感染疾病。
日本启动了"第五代计算机"项目，推动了AI研究的国际竞争。

机器学习革命（2006-至今）：
深度学习的突破性进展彻底改变了AI领域。Geoffrey Hinton、Yann LeCun和Yoshua Bengio
被称为"深度学习三巨头"。AlphaGo在2016年击败围棋世界冠军李世石，标志着AI的里程碑事件。
GPT系列模型的出现推动了大语言模型的发展，OpenAI和Google DeepMind成为领军企业。
"""

SEPARATOR = "=" * 70


def _check_backend():
    try:
        resp = requests.post("http://127.0.0.1:8000/api/v1/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _check_flask():
    try:
        resp = requests.get("http://127.0.0.1:5000/", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _create_session() -> str:
    try:
        resp = requests.post(
            f"{API_BASE}/sessions",
            json={"title": "Timing Test"},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            session = data.get("session", {})
            sid = session.get("id", "")
            if sid:
                return sid
    except Exception as e:
        print(f"  [warn] session create failed: {e}")
    return f"#test-{int(time.time())}"


# ─── Phase 1: Upload ─────────────────────────────────────────────

def test_upload(session_id: str, content: str, filename: str = "test_timing.txt"):
    print(f"\n{SEPARATOR}")
    print(f"PHASE 1: TXT UPLOAD")
    print(f"  file: {filename} ({len(content)} chars)")
    print(f"  session: {session_id}")
    print(SEPARATOR)

    file_data = content.encode("utf-8")
    files = {"files": (filename, file_data, "text/plain")}
    data = {"session_id": session_id}

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{API_BASE}/ingest", files=files, data=data, timeout=300)
        t_http = time.perf_counter() - t0
        print(f"  [HTTP响应] {t_http:.2f}s | status={resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            bg = result.get("background_processing", False)
            print(f"  background_processing: {bg}")
        else:
            print(f"  ERROR: {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"  UPLOAD FAILED: {e}")
        return False

    # Poll for processing completion
    print(f"\n  [后台处理] 等待文档处理完成...")
    t_poll_start = time.perf_counter()
    poll_interval = 2.0
    last_status = ""

    while True:
        elapsed = time.perf_counter() - t_poll_start
        if elapsed > 600:
            print(f"  TIMEOUT: 超过10分钟未完成")
            return False

        try:
            resp = requests.get(f"{API_BASE}/sessions/{session_id}/files", timeout=10)
            if resp.status_code == 200:
                files_list = resp.json().get("files", [])
                target = next((f for f in files_list if f.get("filename") == filename), None)
                if target:
                    status = target.get("status", "unknown")
                    if status != last_status:
                        print(f"    T+{elapsed:6.1f}s  状态: {last_status or '(初始)'} → {status}")
                        last_status = status

                    if status.lower() in {"processed", "completed", "done", "success", "ready", "finished"}:
                        total_upload = time.perf_counter() - t0
                        print(f"\n  [处理完成] 后台处理耗时: {elapsed:.1f}s")
                        print(f"  [总上传耗时] {total_upload:.1f}s (HTTP {t_http:.2f}s + 后台 {elapsed:.1f}s)")
                        return True
                    elif status.lower() in {"failed", "error"}:
                        print(f"  PROCESSING FAILED after {elapsed:.1f}s")
                        return False
        except Exception:
            pass

        time.sleep(poll_interval)
        if elapsed > 30:
            poll_interval = 5.0
        if elapsed > 120:
            poll_interval = 10.0

    return False


# ─── Phase 2: Query in all modes ─────────────────────────────────

def test_query_stream(session_id: str, question: str, ui_mode: str):
    """Send a streaming query via Flask proxy and measure timing."""
    mode_labels = {"quick": "⚡ 快速", "standard": "⚖️ 标准", "deep": "🧠 深度"}
    mode_map = {"quick": "naive", "standard": "hybrid", "deep": "mix"}

    label = mode_labels.get(ui_mode, ui_mode)
    backend_mode = mode_map.get(ui_mode, "hybrid")
    enable_thinking = (ui_mode == "deep")

    print(f"\n  ── {label} 模式 (mode={backend_mode}, thinking={enable_thinking}) ──")

    payload = {
        "question": question,
        "mode": backend_mode,
        "enable_thinking": enable_thinking,
        "session_id": session_id,
    }

    t0 = time.perf_counter()
    t_first_chunk = None
    full_answer = ""
    status_messages = []
    meta_info = {}
    chunk_count = 0

    try:
        resp = requests.post(
            f"{API_BASE}/query/stream",
            json=payload,
            stream=True,
            timeout=300,
        )

        if resp.status_code != 200:
            print(f"    ERROR: HTTP {resp.status_code}: {resp.text[:200]}")
            return

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except Exception:
                continue

            etype = event.get("type", "")
            elapsed = time.perf_counter() - t0

            if etype == "status":
                msg = event.get("message", "")
                status_messages.append((elapsed, msg))
                print(f"    T+{elapsed:6.2f}s  [状态] {msg}")

            elif etype == "meta":
                meta_info = event
                thinking = event.get("thinking_process", "")
                mode_reported = event.get("mode", "")
                mem_summaries = event.get("memory_summaries", [])
                expanded = event.get("expanded_matches", [])
                print(f"    T+{elapsed:6.2f}s  [元数据] mode={mode_reported} 记忆={len(mem_summaries)} 扩展节点={len(expanded)}")

                if thinking and isinstance(thinking, dict):
                    for k, v in thinking.items():
                        print(f"              {k}: {v}")

            elif etype == "chunk":
                chunk_count += 1
                text = event.get("content", "")
                full_answer += text
                if t_first_chunk is None:
                    t_first_chunk = elapsed
                    print(f"    T+{elapsed:6.2f}s  [首个文本块] ({len(text)} chars)")

    except requests.exceptions.Timeout:
        print(f"    TIMEOUT after {time.perf_counter() - t0:.1f}s")
        return
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    t_total = time.perf_counter() - t0

    # Summary
    print(f"\n    ┌─ {label} 模式结果 ─────────────────────")
    print(f"    │ 总耗时:       {t_total:.2f}s")
    if t_first_chunk is not None:
        print(f"    │ 首块延迟:     {t_first_chunk:.2f}s (等待检索+LLM)")
    print(f"    │ 文本块数:     {chunk_count}")
    print(f"    │ 回答长度:     {len(full_answer)} chars")
    if status_messages:
        print(f"    │ 状态消息:     {len(status_messages)} 条")
        for t, msg in status_messages:
            print(f"    │   T+{t:.2f}s {msg}")
    print(f"    └{'─' * 40}")

    return {
        "mode": ui_mode,
        "total_seconds": round(t_total, 2),
        "first_chunk_seconds": round(t_first_chunk, 2) if t_first_chunk else None,
        "answer_length": len(full_answer),
        "chunk_count": chunk_count,
    }


def test_all_queries(session_id: str, question: str):
    print(f"\n{SEPARATOR}")
    print(f"PHASE 2: QUERY TIMING TEST")
    print(f"  question: {question}")
    print(f"  session:  {session_id}")
    print(SEPARATOR)

    results = []
    for mode in ["quick", "standard", "deep"]:
        r = test_query_stream(session_id, question, mode)
        if r:
            results.append(r)
        time.sleep(2)

    # Comparison table
    if results:
        print(f"\n{SEPARATOR}")
        print("COMPARISON SUMMARY")
        print(SEPARATOR)
        print(f"{'模式':<12} {'总耗时':>10} {'首块延迟':>10} {'回答长度':>10}")
        print("-" * 50)
        for r in results:
            mode_labels = {"quick": "⚡ 快速", "standard": "⚖️ 标准", "deep": "🧠 深度"}
            label = mode_labels.get(r["mode"], r["mode"])
            fc = f"{r['first_chunk_seconds']}s" if r['first_chunk_seconds'] else "N/A"
            print(f"{label:<12} {r['total_seconds']:>8}s {fc:>10} {r['answer_length']:>8} chars")

    return results


def main():
    parser = argparse.ArgumentParser(description="TXT Upload + Query Timing Test")
    parser.add_argument("--session", "-s", help="Use existing session ID")
    parser.add_argument("--file", "-f", help="Path to TXT file to upload")
    parser.add_argument("--question", "-q", default="详细介绍一下人工智能的发展历程", help="Query to test")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload, only test queries")
    args = parser.parse_args()

    print(SEPARATOR)
    print("DocThinker TXT Upload + Query End-to-End Timing Test")
    print(SEPARATOR)

    # Check servers
    backend_ok = _check_backend()
    flask_ok = _check_flask()
    print(f"FastAPI Backend (8000): {'OK' if backend_ok else 'NOT RUNNING'}")
    print(f"Flask UI (5000):        {'OK' if flask_ok else 'NOT RUNNING'}")

    if not backend_ok:
        print("\nERROR: Backend server is not running. Start with:")
        print("  python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Session
    session_id = args.session
    if not session_id:
        print("\nCreating test session...")
        session_id = _create_session()
    print(f"Session: {session_id}")

    # Content
    content = SAMPLE_TXT
    filename = "test_timing.txt"
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        import os
        filename = os.path.basename(args.file)
    print(f"File: {filename} ({len(content)} chars)")

    # Phase 1: Upload
    if not args.skip_upload:
        upload_ok = test_upload(session_id, content, filename)
        if not upload_ok:
            print("\nUpload failed or timed out. Check server logs.")
            print("You can retry queries with: --skip-upload --session", session_id)
            sys.exit(1)
    else:
        print("\n[Skipping upload — using existing session data]")

    # Phase 2: Query
    results = test_all_queries(session_id, args.question)

    # Final summary
    print(f"\n{SEPARATOR}")
    print("DONE")
    print(f"Session ID: {session_id}")
    print(f"\nCheck server logs for detailed per-stage timing:")
    print(f"  [TXT T+...]       — 文件摄入各阶段")
    print(f"  [ainsert T+...]   — GraphCore 插入 (enqueue / pipeline / total)")
    print(f"  [pipeline T+...]  — chunk embedding / entity extraction / merge / persist")
    print(f"  [kg_query T+...]  — 标准/深度模式: 关键词提取 / 上下文构建 / LLM 生成")
    print(f"  [naive_query T+...] — 快速模式: 向量检索 / LLM 生成")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
