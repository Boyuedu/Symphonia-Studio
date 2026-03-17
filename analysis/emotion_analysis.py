"""
Emotion / music landscape analysis.
Produces an "emotional journey" (interpretive) over time for visualization.
"""

import os
import sys
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.structure_analysis import get_structure_timeline, structure_timeline_for_llm


def get_emotion_journey_from_llm(
    piece_id: str,
    piece_title: str,
    composer_name: str,
    llm_complete_json: Any,
) -> list[dict[str, Any]]:
    """
    Call LLM to generate emotion journey; return list of {time_sec, label, description}.
    llm_complete_json(system, user) -> dict or None.
    """
    from llm.prompts import EMOTION_LANDSCAPE

    timeline = structure_timeline_for_llm(piece_id)
    struct = get_structure_timeline(piece_id)
    duration = struct["duration_seconds"] if struct else 0
    structure_summary = timeline or "No structure data."

    user = EMOTION_LANDSCAPE.format(
        piece_title=piece_title,
        composer_name=composer_name,
        duration_seconds=duration,
        structure_summary=structure_summary,
    )
    out = llm_complete_json(
        "You output only valid JSON. No markdown, no explanation.",
        user,
    )
    if not out or not isinstance(out, dict):
        return _default_emotion_journey(piece_id, duration)

    journey = out.get("journey")
    if not isinstance(journey, list):
        return _default_emotion_journey(piece_id, duration)

    result = []
    for p in journey:
        if isinstance(p, dict) and "time_sec" in p:
            result.append({
                "time_sec": int(p.get("time_sec", 0)),
                "label": str(p.get("label", "")).strip() or "Moment",
                "description": str(p.get("description", "")).strip() or "",
            })
    result.sort(key=lambda x: x["time_sec"])
    return result if result else _default_emotion_journey(piece_id, duration)


def _default_emotion_journey(piece_id: str, duration_seconds: int) -> list[dict[str, Any]]:
    """Fallback journey when LLM is unavailable or fails."""
    step = max(1, duration_seconds // 5)
    return [
        {"time_sec": 0, "label": "Opening", "description": "The piece begins—notice the initial mood."},
        {"time_sec": step, "label": "Developing", "description": "The music unfolds; textures or dynamics may shift."},
        {"time_sec": 2 * step, "label": "Middle", "description": "A possible peak or contrast in the middle."},
        {"time_sec": 3 * step, "label": "Return", "description": "Familiar material may return or transform."},
        {"time_sec": duration_seconds, "label": "Closing", "description": "The journey winds down to the end."},
    ]


def get_emotion_journey(
    piece_id: str,
    piece_title: str,
    composer_name: str,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """
    Get emotion journey for a piece. If use_llm=True, calls LLM; otherwise returns default.
    """
    if use_llm:
        try:
            from llm.nemotron_client import complete_json
            return get_emotion_journey_from_llm(
                piece_id, piece_title, composer_name,
                lambda sys_p, usr_p: complete_json(sys_p, usr_p, temperature=0.6),
            )
        except Exception:
            pass
    struct = get_structure_timeline(piece_id)
    duration = (struct.get("duration_seconds") if struct else None) or 300
    if duration == 300:
        try:
            from utils.retrieval import get_piece_by_id
            piece = get_piece_by_id(piece_id)
            if piece and piece.get("duration_seconds"):
                duration = piece["duration_seconds"]
        except Exception:
            pass
    return _default_emotion_journey(piece_id, duration)
