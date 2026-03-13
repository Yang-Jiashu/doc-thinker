#!/usr/bin/env python3
"""
whitecat Web UI

Simple web interface for testing and managing whitecat functionality
"""

import sys
import os
from pathlib import Path
from urllib.parse import quote

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from flask_cors import CORS
from jinja2 import FileSystemLoader

# Try to import DocThinker configuration
# Use mock configuration if full import fails
try:
    from docthinker.config import DocThinkerConfig
    from docthinker.api_config import api_config

    # 模板目录与静态目录：固定为与本文件同目录
    _UI_DIR = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(_UI_DIR, 'templates')
    static_dir = os.path.join(_UI_DIR, 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='/static')
    app.jinja_loader = FileSystemLoader(template_dir)

    # Load configuration
    config = DocThinkerConfig()

    # Enable CORS if configured
    if api_config.enable_cors:
        CORS(app, origins=api_config.cors_origins)

    # Configure app
    app.config['DEBUG'] = False
    app.config['SECRET_KEY'] = 'dev-secret-key'
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True

except ImportError as e:
    print(f"Warning: Failed to import full DocThinker configuration: {e}")
    print("Starting UI with minimal configuration...")

    # Create a minimal configuration
    class MockConfig:
        def __init__(self):
            self.ui_host = '0.0.0.0'
            self.ui_port = 5000

    class MockAPIConfig:
        def __init__(self):
            self.api_prefix = '/api/v1'
            self.enable_cors = True
            self.cors_origins = ['*']

    # 模板目录与静态目录：固定为与本文件同目录
    _UI_DIR = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(_UI_DIR, 'templates')
    static_dir = os.path.join(_UI_DIR, 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='/static')
    app.jinja_loader = FileSystemLoader(template_dir)

    # Minimal configuration
    config = MockConfig()
    api_config = MockAPIConfig()

    # Enable CORS
    CORS(app, origins=['*'])
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True

    # Configure app
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = 'dev-secret-key'

# 开发时禁用页面缓存，确保模板更新后立即生效
@app.after_request
def _no_cache_html(response):
    if response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        try:
            html = response.get_data(as_text=True)
            if "</head>" in html and "ui-runtime-css" not in html:
                inject = """
<style id="ui-runtime-css">
aside img.w-10.h-10.rounded-xl.object-contain.flex-shrink-0,
aside img[src*="logo.png"],
aside img[alt="DocThinker"] {
    width: 96px !important;
    height: 96px !important;
    min-width: 96px !important;
    min-height: 96px !important;
    max-width: 96px !important;
    max-height: 96px !important;
    object-fit: contain !important;
    border-radius: 14px !important;
}
</style>
"""
                html = html.replace("</head>", inject + "</head>")
                response.set_data(html)
        except Exception as _e:
            pass
    return response

# Logo: 从项目根目录直接提供，确保左上角能够正确显示
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGO_PATH = _PROJECT_ROOT / "logo.png"

@app.route('/logo.png')
def serve_logo():
    """Serve logo.png from project root as the primary logo source."""
    if _LOGO_PATH.exists():
        return send_file(_LOGO_PATH, mimetype='image/png', max_age=3600)
    return send_file(Path(__file__).resolve().parent / 'static' / 'logo.png', mimetype='image/png', max_age=3600)

# Home route
@app.route('/')
def index():
    """Home page - redirect to chat interface"""
    return redirect(url_for('query_page'))

# Configuration page - Modern UI
@app.route('/config')
def config_page():
    """Configuration page with modern UI"""
    return render_template('config_modern.html', config=config, api_config=api_config)

# Query testing page - Modern UI
@app.route('/query')
def query_page():
    """Query testing page with modern UI"""
    return render_template('query_modern.html', config=config, api_config=api_config)

# Knowledge graph visualization page - Modern UI
@app.route('/knowledge-graph')
def knowledge_graph_page():
    """Knowledge graph visualization page"""
    return render_template('kg_viz_modern.html', config=config, api_config=api_config)

@app.route('/kg-viz')
def kg_viz_page():
    """Legacy alias route for the knowledge graph page."""
    return redirect(url_for('knowledge_graph_page'))

# Upload page - Modern UI
@app.route('/upload')
def upload_page():
    """File upload page with modern UI"""
    return render_template('upload_modern.html', config=config, api_config=api_config)

# File upload API endpoint - Connects to backend
@app.route(f'{api_config.api_prefix}/upload', methods=['POST'])
def upload_files():
    """API endpoint for file uploads - proxies to FastAPI (supports chat multi-file with key 'files')"""
    import requests
    import time
    
    # 聊天页使用 formData.append('files', file)，多个文件使用同一个 key
    files_list = request.files.getlist('files')
    if files_list and any(f and f.filename for f in files_list):
        try:
            backend_url = "http://127.0.0.1:8000/api/v1/ingest"
            files_to_send = [
                ('files', (f.filename, f.stream.read(), f.content_type or 'application/octet-stream'))
                for f in files_list if f and f.filename
            ]
            data = {k: request.form.get(k) for k in ['session_id', 'query', 'mode'] if request.form.get(k)}
            response = requests.post(backend_url, files=files_to_send, data=data, timeout=300)
            ct = response.headers.get('content-type', '')
            return jsonify(response.json() if 'application/json' in ct else {'status': response.text}), response.status_code
        except requests.exceptions.ConnectionError:
            return jsonify({'success': False, 'message': '无法连接后端，请先启动: python -m uvicorn docthinker.server.app:app --port 8000'}), 503
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Empty filename'
            }), 400
        
        # Save file locally first
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # Try to forward to backend asynchronously (don't block UI)
        try:
            backend_url = "http://127.0.0.1:8000/api/v1/ingest"
            
            with open(file_path, 'rb') as f:
                files = {'files': (file.filename, f, file.content_type or 'application/octet-stream')}
                response = requests.post(backend_url, files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and processed successfully',
                    'filename': file.filename,
                    'size': file_size,
                    'document_id': result.get('document_id', f"doc_{int(time.time())}"),
                    'backend_response': result
                }), 200
            else:
                # Backend error but file saved locally
                return jsonify({
                    'success': True,
                    'message': 'File saved locally (backend processing failed)',
                    'filename': file.filename,
                    'size': file_size,
                    'document_id': f"doc_{int(time.time())}",
                    'backend_error': response.text
                }), 200
                
        except requests.exceptions.ConnectionError:
            # Backend not available
            return jsonify({
                'success': True,
                'message': 'File saved locally (backend not connected)',
                'filename': file.filename,
                'size': file_size,
                'document_id': f"doc_{int(time.time())}"
            }), 200
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Upload failed: {str(e)}'
        }), 500

# Proxy endpoints for Sessions
@app.route(f'{api_config.api_prefix}/sessions', methods=['GET', 'POST'])
def sessions_proxy():
    import requests
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions"
    try:
        if request.method == 'GET':
            resp = requests.get(backend_url)
        else:
            resp = requests.post(backend_url, json=request.get_json())
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/sessions/<session_id>', methods=['GET', 'DELETE', 'PUT'])
def session_detail_proxy(session_id):
    import requests
    encoded = quote(session_id, safe="")
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions/{encoded}"
    try:
        if request.method == 'GET':
            resp = requests.get(backend_url)
        elif request.method == 'PUT':
            resp = requests.put(backend_url, json=request.get_json())
        else:
            resp = requests.delete(backend_url)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/sessions/<session_id>/history', methods=['GET'])
def session_history_proxy(session_id):
    import requests
    encoded = quote(session_id, safe="")
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions/{encoded}/history"
    try:
        resp = requests.get(backend_url)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/sessions/<session_id>/files', methods=['GET'])
def session_files_proxy(session_id):
    import requests
    encoded = quote(session_id, safe="")
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions/{encoded}/files"
    try:
        resp = requests.get(backend_url)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/knowledge-graph/data', methods=['GET'])
def kg_data_proxy():
    import requests as req_lib
    session_id = request.args.get('session_id', '')
    if not session_id:
        return jsonify({
            'nodes': [],
            'edges': [],
            'metadata': {'error': 'session_id is required'}
        }), 400
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/knowledge-graph/data"
    try:
        resp = req_lib.get(backend_url, params={'session_id': session_id}, timeout=30)
        if resp.status_code == 404:
            return jsonify({
                'nodes': [],
                'edges': [],
                'metadata': {'error': '后端 404', 'hint': '请先启动 DocThinker 后端: cd doc-thinker && python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000'}
            }), 200
        try:
            data = resp.json()
        except Exception:
            data = {'nodes': [], 'edges': [], 'metadata': {'error': f'后端返回非 JSON (HTTP {resp.status_code})', 'raw': resp.text[:200] if resp.text else ''}}
        return jsonify(data), 200
    except req_lib.exceptions.ConnectionError:
        return jsonify({
            'nodes': [],
            'edges': [],
            'metadata': {'error': '无法连接后端 (port 8000)', 'hint': '请先启动: cd doc-thinker && python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000'}
        }), 200
    except Exception as e:
        return jsonify({'nodes': [], 'edges': [], 'metadata': {'error': str(e)}}), 200

@app.route(f'{api_config.api_prefix}/knowledge-graph/expand', methods=['POST'])
def kg_expand_proxy():
    import requests
    data = request.get_json() or {}
    session_id = data.get('session_id')
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/knowledge-graph/expand"
    try:
        resp = requests.post(backend_url, json={
            'session_id': session_id,
            'angle_indices': data.get('angle_indices'),
            'apply': data.get('apply', True),
        }, timeout=120)
        body = resp.json() if resp.content and 'application/json' in resp.headers.get('content-type', '') else {}
        if resp.status_code == 404:
            body = {
                'success': False,
                'error': '扩展接口不存在 (404)。请使用以下命令重启后端: python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000',
                'detail': body.get('detail', 'Not Found'),
            }
        elif resp.status_code >= 400 and 'detail' not in body and 'error' not in body:
            body['error'] = (resp.text[:200] if resp.text else None) or f'HTTP {resp.status_code}'
        return jsonify(body), resp.status_code if resp.status_code != 404 else 200
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': '无法连接后端 (127.0.0.1:8000)。请先启动: python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000'
        }), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/knowledge-graph/debug-expanded', methods=['GET'])
def kg_debug_expanded_proxy():
    import requests
    session_id = request.args.get('session_id', '')
    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/knowledge-graph/debug-expanded"
    try:
        resp = requests.get(backend_url, params={'session_id': session_id}, timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/knowledge-graph/stats', methods=['GET'])
def kg_stats_proxy():
    import requests
    session_id = request.args.get('session_id', '')
    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/knowledge-graph/stats"
    try:
        resp = requests.get(backend_url, params={'session_id': session_id}, timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

_IMG_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _serve_kg_image(session_id, filename):
    """Serve image files directly from the session's multimodal/images directory."""
    from urllib.parse import unquote
    sid = unquote(session_id)
    safe_name = Path(unquote(filename)).name
    img_path = _PROJECT_ROOT / "data" / sid / "knowledge" / "multimodal" / "images" / safe_name
    if img_path.is_file():
        mime = _IMG_MIME.get(img_path.suffix.lower(), "image/png")
        return send_file(str(img_path), mimetype=mime)
    return jsonify({"error": f"Image not found: {safe_name}", "path_checked": str(img_path)}), 404

@app.route(f'{api_config.api_prefix}/knowledge-graph/image/<session_id>/<filename>', methods=['GET'])
def kg_image_serve(session_id, filename):
    return _serve_kg_image(session_id, filename)

@app.route(f'{api_config.api_prefix}/knowledge-graph/stats-all', methods=['GET'])
def kg_stats_all_proxy():
    """Diagnostic endpoint: aggregate graph stats for all sessions."""
    import requests
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/knowledge-graph/stats-all"
    try:
        resp = requests.get(backend_url, timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/graph/memory/graph-data', methods=['GET'])
def memory_graph_proxy():
    import requests
    session_id = request.args.get('session_id', '')
    if not session_id:
        return jsonify({'nodes': [], 'edges': [], 'error': 'session_id is required'}), 400
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/memory/graph-data"
    try:
        resp = requests.get(backend_url, params={'session_id': session_id}, timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'nodes': [], 'edges': [], 'error': str(e)}), 500

# Query endpoint - Connect to backend
@app.route(f'{api_config.api_prefix}/query/stream', methods=['POST'])
def query_stream():
    """Streaming query endpoint - proxies SSE to FastAPI backend"""
    import requests
    from flask import Response, stream_with_context
    
    try:
        data = request.get_json() or {}
        
        backend_url = "http://127.0.0.1:8000/api/v1/query/stream"
        ui_mode = str(data.get('ui_mode', 'standard') or 'standard').lower()
        mode_map = {
            "standard": "hybrid",
            "deep": "mix",
            "quick": "naive",
        }
        
        payload = {
            "question": data.get('question', data.get('text', '')),
            "memory_mode": data.get('memory_mode', 'session'),
            "mode": mode_map.get(ui_mode, "hybrid"),
            "enable_thinking": ui_mode == "deep",
            "session_id": data.get('session_id')
        }

        # Make streaming request to backend
        req = requests.post(backend_url, json=payload, stream=True, timeout=300)
        
        if req.status_code != 200:
            return jsonify({
                'success': False,
                'response': f'后端错误 (状态码: {req.status_code}): {req.text}'
            }), 500

        def generate():
            for line in req.iter_lines():
                if line:
                    yield f"{line.decode('utf-8')}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'response': '无法连接到后端服务 (127.0.0.1:8000)。\n\n请确认后端已启动:\npython -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000'
        }), 503
    except requests.exceptions.ReadTimeout:
        return jsonify({
            'success': False,
            'response': '后端处理超时（>300秒）。'
        }), 504
    except Exception as e:
        return jsonify({
            'success': False,
            'response': f'发生未知错误: {str(e)}'
        }), 500

@app.route(f'{api_config.api_prefix}/query/text', methods=['POST'])
def text_query():
    """Text query endpoint - connects to FastAPI backend on port 8000"""
    import requests
    
    try:
        data = request.get_json() or {}
        
        # Try to connect to backend
        backend_url = "http://127.0.0.1:8000/api/v1/query"
        ui_mode = str(data.get('ui_mode', 'standard') or 'standard').lower()
        mode_map = {
            "standard": "hybrid",
            "deep": "mix",
            "quick": "naive",
        }
        
        payload = {
            "question": data.get('question', data.get('text', '')),
            "memory_mode": data.get('memory_mode', 'session'),
            "mode": mode_map.get(ui_mode, "hybrid"),
            "enable_thinking": ui_mode == "deep",
            "session_id": data.get('session_id')
        }

        response = requests.post(backend_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'response': result.get('response', result.get('answer', 'No response')),
                'sources': result.get('sources', []),
                'reasoning': result.get('reasoning', ''),
                'thinking_process': result.get('thinking_process', ''),
                'answer_mode': result.get('answer_mode', ''),
                'expanded_matches': result.get('expanded_matches', []),
                'memory_summaries': result.get('memory_summaries', []),
            })
        else:
            return jsonify({
                'success': False,
                'response': f'后端错误 (状态码: {response.status_code}): {response.text}'
            }), 500
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'response': '无法连接到后端服务 (127.0.0.1:8000)。\n\n请确认后端已启动:\npython -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000'
        }), 503
    except requests.exceptions.ReadTimeout:
        return jsonify({
            'success': False,
            'response': '后端处理超时（>300秒）。当前会话首次检索或正在入库时可能较慢，请稍后重试。'
        }), 504
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'response': f'请求处理失败: {str(e)}'
        }), 500


@app.route(f'{api_config.api_prefix}/ingest/stream', methods=['POST'])
def ingest_stream_proxy():
    data = request.get_json()
    try:
        import requests
        backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/ingest/stream"
        resp = requests.post(backend_url, json=data, timeout=60)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Config update endpoint
@app.route(f'{api_config.api_prefix}/config', methods=['POST'])
def update_config_proxy():
    """Proxy for updating configuration"""
    data = request.get_json()
    try:
        import requests
        backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/config"
        resp = requests.post(backend_url, json=data)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Settings GET/POST — proxy to FastAPI /settings
@app.route(f'{api_config.api_prefix}/settings', methods=['GET', 'POST'])
def settings_proxy():
    import requests as req_lib
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/settings"
    try:
        if request.method == 'GET':
            resp = req_lib.get(backend_url, timeout=10)
        else:
            resp = req_lib.post(backend_url, json=request.get_json(), timeout=10)
        return jsonify(resp.json()), resp.status_code
    except req_lib.exceptions.ConnectionError:
        return jsonify({'success': False, 'message': '后端服务未启动'}), 503
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# 通用 API 代理：将未单独定义的 /api/v1/* 转发到 FastAPI 后端 (8000)
@app.route(f'{api_config.api_prefix}/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def api_proxy(subpath):
    import requests
    import re
    m = re.match(r'^knowledge-graph/image/(.+)/([^/]+)$', subpath)
    if m and request.method == 'GET':
        return _serve_kg_image(m.group(1), m.group(2))
    backend = f"http://127.0.0.1:8000{api_config.api_prefix}/{quote(subpath, safe='/')}"
    try:
        if request.method == 'GET':
            r = requests.get(backend, params=request.args, timeout=30)
        elif request.method == 'POST':
            r = requests.post(backend, json=request.get_json(silent=True) or {}, params=request.args, timeout=60)
        elif request.method == 'PUT':
            r = requests.put(backend, json=request.get_json(silent=True) or {}, params=request.args, timeout=30)
        elif request.method == 'DELETE':
            r = requests.delete(backend, params=request.args, timeout=30)
        else:
            r = requests.request(request.method, backend, params=request.args, timeout=30)
        try:
            body = r.json()
        except Exception:
            body = {'detail': f'后端返回非 JSON (HTTP {r.status_code})'}
        return jsonify(body), r.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({'detail': '请先启动 FastAPI 后端: uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000'}), 503
    except Exception as e:
        return jsonify({'detail': str(e)}), 500

# Run the app if this file is executed directly
if __name__ == '__main__':
    _UI_DIR = os.path.abspath(os.path.dirname(__file__))
    static_dir = os.path.join(_UI_DIR, 'static')
    templates_dir = os.path.join(_UI_DIR, 'templates')
    Path(static_dir).mkdir(parents=True, exist_ok=True)
    Path(templates_dir).mkdir(parents=True, exist_ok=True)
    print()
    print("  ========================================")
    print("  whitecat UI")
    print("  ========================================")
    print("  Templates:", templates_dir)
    print("  URL:      http://127.0.0.1:{}/query".format(config.ui_port))
    print("  知识图谱: http://127.0.0.1:{}/knowledge-graph".format(config.ui_port))
    print("  ========================================")
    print()
    app.run(host=config.ui_host, port=config.ui_port, debug=False, use_reloader=False)

