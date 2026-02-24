#!/usr/bin/env python3
"""
whitecat Web UI

Simple web interface for testing and managing whitecat functionality
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from jinja2 import FileSystemLoader

# Try to import RAGAnything configuration
# Use mock configuration if full import fails
try:
    from raganything.config import RAGAnythingConfig
    from raganything.api_config import api_config, api_routes
    from raganything.knowledge_graph import KnowledgeGraph
    
    # 模板目录与静态目录：固定为与本文件同目录
    _UI_DIR = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(_UI_DIR, 'templates')
    static_dir = os.path.join(_UI_DIR, 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='/static')
    app.jinja_loader = FileSystemLoader(template_dir)
    print("[whitecat] 模板目录 templates =", template_dir)
    
    # Load configuration
    config = RAGAnythingConfig()
    
    # Enable CORS if configured
    if api_config.enable_cors:
        CORS(app, origins=api_config.cors_origins)
    
    # Configure app
    app.config['DEBUG'] = False
    app.config['SECRET_KEY'] = 'dev-secret-key'
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True
    
    try:
        _src, _fname, _ = app.jinja_loader.get_source(app.jinja_env, 'base_modern.html')
        print("[whitecat] 加载的 base_modern.html 文件 =", _fname)
        print("[whitecat] 模板包含 w-10 类 =", "w-10 h-10" in _src)
    except Exception as _e:
        print("[whitecat] 读取 base_modern.html 失败:", _e)
    
    # Simple in-memory storage for demo purposes
    # In production, this would be replaced with actual RAGAnything instance
    rag_instance = None
    knowledge_graph = None
except ImportError as e:
    print(f"Warning: Failed to import full RAGAnything configuration: {e}")
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
    
    class MockAPIRoutes:
        def __init__(self):
            self.kg_base = '/knowledge-graph'
            self.kg_stats = '/stats'
            self.query_base = '/query'
            self.query_text = '/text'
            self.viz_base = '/visualization'
            self.viz_data = '/data'
            self.vc_base = '/version-control'
            self.vc_snapshots = '/snapshots'
    
    # 模板目录与静态目录：固定为与本文件同目录
    _UI_DIR = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(_UI_DIR, 'templates')
    static_dir = os.path.join(_UI_DIR, 'static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='/static')
    app.jinja_loader = FileSystemLoader(template_dir)
    print("[whitecat] 模板目录 templates =", template_dir)
    
    # Minimal configuration
    config = MockConfig()
    api_config = MockAPIConfig()
    api_routes = MockAPIRoutes()
    
    # Enable CORS
    CORS(app, origins=['*'])
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True
    
    # Configure app
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = 'dev-secret-key'
    
    rag_instance = None
    knowledge_graph = None

# 注册知识图谱可视化蓝图
try:
    from raganything.ui.routers.kg_visualization import kg_viz_bp
    app.register_blueprint(kg_viz_bp)
    print("[whitecat] 知识图谱可视化路由已注册: /kg-viz")
except ImportError as e:
    print(f"[whitecat] 知识图谱可视化路由注册失败: {e}")

# 开发时禁止缓存页面，确保模板修改后刷新即生效
@app.after_request
def _no_cache_html(response):
    if response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Whitecat-Postprocess"] = "1"
        try:
            html = response.get_data(as_text=True)
            if "</head>" in html and "logo-override-css" not in html:
                inject = """
<style id="logo-override-css">
/* runtime override: enlarge sidebar logo */
aside img.w-10.h-10.rounded-xl.object-contain.flex-shrink-0,
aside img[src*="logo.png"],
aside img[alt="DocThinker"] {
    width: 140px !important;
    height: 140px !important;
    min-width: 140px !important;
    min-height: 140px !important;
    max-width: 140px !important;
    max-height: 140px !important;
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

# Entity management page
@app.route('/entities')
def entities_page():
    """Entity management page"""
    return render_template('entities.html', config=config, api_config=api_config)

# Relationship management page
@app.route('/relationships')
def relationships_page():
    """Relationship management page"""
    return render_template('relationships.html', config=config, api_config=api_config)

# Version control page
@app.route('/version-control')
def version_control_page():
    """Version control page"""
    return render_template('version_control.html', config=config, api_config=api_config)

# Reasoning rules page
@app.route('/reasoning-rules')
def reasoning_rules_page():
    """Reasoning rules page"""
    return render_template('reasoning_rules.html', config=config, api_config=api_config)

# Upload page - Modern UI
@app.route('/upload')
def upload_page():
    """File upload page with modern UI"""
    return render_template('upload_modern.html', config=config, api_config=api_config)

# File upload API endpoint - Connects to backend
@app.route(f'{api_config.api_prefix}/upload', methods=['POST'])
def upload_files():
    """API endpoint for file uploads - connects to FastAPI backend"""
    import requests
    import time
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            }), 400
        
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
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions/{session_id}"
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
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions/{session_id}/history"
    try:
        resp = requests.get(backend_url)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route(f'{api_config.api_prefix}/sessions/<session_id>/files', methods=['GET'])
def session_files_proxy(session_id):
    import requests
    backend_url = f"http://127.0.0.1:8000{api_config.api_prefix}/sessions/{session_id}/files"
    try:
        resp = requests.get(backend_url)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Query endpoint - Connect to backend
@app.route(f'{api_config.api_prefix}/query/text', methods=['POST'])
def text_query():
    """Text query endpoint - connects to FastAPI backend on port 8000"""
    import requests
    
    try:
        data = request.get_json() or {}
        
        # Try to connect to backend
        backend_url = "http://127.0.0.1:8000/api/v1/query"
        
        payload = {
            "question": data.get('question', data.get('text', '')),
            "memory_mode": data.get('memory_mode', 'standard'),
            "session_id": data.get('session_id')
        }
        
        response = requests.post(backend_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'response': result.get('response', result.get('answer', '无回答')),
                'sources': result.get('sources', []),
                'reasoning': result.get('reasoning', '')
            })
        else:
            return jsonify({
                'success': False,
                'response': f'后端错误 (状态码: {response.status_code}): {response.text}'
            }), 500
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'response': '无法连接到后端服务 (127.0.0.1:8000)。\n\n请确保后端已启动:\npython -m uvicorn raganything.server.app:app --host 0.0.0.0 --port 8000'
        }), 503
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

# Snapshots endpoint (若后端未实现则用 mock)
@app.route(f'{api_config.api_prefix}/version-control/snapshots')
def mock_snapshots():
    """Mock snapshots endpoint"""
    return jsonify({
        'snapshots': [
            {
                'id': 'snapshot1',
                'description': 'Initial knowledge graph',
                'user': 'system',
                'timestamp': 1704067200,
                'entities': 100,
                'relationships': 200
            },
            {
                'id': 'snapshot2',
                'description': 'After adding Q4 data',
                'user': 'admin',
                'timestamp': 1704153600,
                'entities': 120,
                'relationships': 250
            },
            {
                'id': 'snapshot3',
                'description': 'After validation pass',
                'user': 'admin',
                'timestamp': 1704240000,
                'entities': 150,
                'relationships': 320
            }
        ]
    })

# 通用 API 代理：将未单独定义的 /api/v1/* 转发到 FastAPI 后端 (8000)，供知识图谱、记忆图等使用
@app.route(f'{api_config.api_prefix}/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def api_proxy(subpath):
    import requests
    backend = f"http://127.0.0.1:8000{api_config.api_prefix}/{subpath}"
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
        return jsonify(r.json() if r.headers.get('content-type', '').startswith('application/json') else {}), r.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({'detail': '请先启动 FastAPI 后端: uvicorn raganything.server.app:app --host 0.0.0.0 --port 8000'}), 503
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
