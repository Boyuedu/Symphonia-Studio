"""
Musical structure analysis for guided listening.
Builds section timelines and can augment with LLM-generated cues.
"""

import csv
import os
from typing import Any, Optional

# Project root relative to this file
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METADATA_PATH = os.path.join(_ROOT, "data", "piece_metadata.csv")


def load_piece_metadata() -> list[dict[str, Any]]:
    """Load piece_metadata.csv into a list of dicts."""
    path = _METADATA_PATH
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_structure_timeline(piece_id: str) -> Optional[dict[str, Any]]:
    """
    Get structure timeline for a piece: intro, theme_a, theme_b, development, recap end times (seconds).
    Returns None if piece_id not found.
    """
    for row in load_piece_metadata():
        if row.get("piece_id") == piece_id:
            try:
                intro_end = int(row.get("intro_end", 0) or 0)
                theme_a_end = int(row.get("theme_a_end", 0) or 0)
                theme_b_end = int(row.get("theme_b_end", 0) or 0)
                development_end = int(row.get("development_end", 0) or 0)
                recap_end = int(row.get("recap_end", 0) or 0)
                sections = [
                    ("intro", 0, intro_end),
                    ("theme_a", intro_end, theme_a_end),
                    ("theme_b", theme_a_end, theme_b_end),
                    ("development", theme_b_end, development_end),
                    ("recap", development_end, recap_end),
                ]
                return {
                    "piece_id": piece_id,
                    "title": row.get("title", ""),
                    "composer_id": row.get("composer_id", ""),
                    "duration_seconds": int(row.get("duration_seconds", 0)),
                    "structure_template_id": row.get("structure_template_id", ""),
                    "has_orchestra": (row.get("has_orchestra", "0").strip().lower() in ("1", "true", "yes")),
                    "sections": sections,
                    "intro_end": intro_end,
                    "theme_a_end": theme_a_end,
                    "theme_b_end": theme_b_end,
                    "development_end": development_end,
                    "recap_end": recap_end,
                }
            except (ValueError, TypeError):
                pass
            break
    return None


def structure_timeline_for_llm(piece_id: str) -> Optional[str]:
    """Format structure as a short text summary for LLM prompts."""
    s = get_structure_timeline(piece_id)
    if not s or not s.get("sections"):
        return None
    lines = [f"Total duration: {s['duration_seconds']} seconds"]
    for name, start, end in s["sections"]:
        if end > start:
            lines.append(f"- {name}: {start}s – {end}s")
    return "\n".join(lines)
