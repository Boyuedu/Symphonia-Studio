"""
General helpers for Symphonia Studio.
"""

import os
import sys
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_root() -> str:
    return _ROOT


def ensure_data_path(*segments: str) -> str:
    """Return path under data/; ensure parent dir exists if you plan to write."""
    path = os.path.join(_ROOT, "data", *segments)
    return path


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Navigate nested dict: safe_get(d, 'a','b') -> d.get('a',{}).get('b', default)."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def format_duration(seconds: int) -> str:
    """Format seconds as M:SS."""
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"
