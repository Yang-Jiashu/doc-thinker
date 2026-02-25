"""
Shim package to expose the actual GraphCore implementation under graphcore.coregraph/graphcore.coregraph.

Upstream installs the project so that `graphcore.coregraph` is the importable package. When using
the source tree directly (which contains an outer `graphcore.coregraph` folder that holds the
real package as a subdirectory), we provide this thin wrapper so that
`import graphcore.coregraph` behaves exactly like the published package.
"""

import os
from importlib import import_module
from pathlib import Path
from pkgutil import extend_path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_INNER_PATH = _THIS_DIR / "coregraph"

# Ensure the real package directory is part of the package search path
FORCE_LOCAL = os.getenv("GraphcoreForceLocal")
if FORCE_LOCAL and FORCE_LOCAL.lower() in {"1", "true", "yes"}:
    # Only use the inner source tree, avoid extending to site-packages.
    if _INNER_PATH.exists():
        __path__ = [str(_INNER_PATH)]
else:
    __path__ = extend_path(__path__, __name__)
    if _INNER_PATH.exists():
        inner_str = str(_INNER_PATH)
        if inner_str not in __path__:
            __path__.append(inner_str)

_inner = import_module(".coregraph", __name__)

GraphCore: Any = getattr(_inner, "GraphCore")
QueryParam: Any = getattr(_inner, "QueryParam")
__all__ = getattr(_inner, "__all__", [])
__version__ = getattr(_inner, "__version__", "dev")
__author__ = getattr(_inner, "__author__", "")
__url__ = getattr(_inner, "__url__", "")

# Expose remaining public attributes transparently
for name in dir(_inner):
    if name.startswith("_"):
        continue
    globals().setdefault(name, getattr(_inner, name))
