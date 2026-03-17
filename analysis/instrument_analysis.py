"""
Instrument / orchestra analysis for orchestral pieces.
Describes roles of instrument groups and supports "orchestra conversation" and listening modes.
"""

import os
import sys
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Default instrument groups for orchestral pieces (can be overridden per piece later)
DEFAULT_ORCHESTRA_GROUPS = [
    "Strings (violins, violas, cellos, basses)",
    "Woodwinds (flutes, oboes, clarinets, bassoons)",
    "Brass (horns, trumpets, trombones, tuba)",
    "Piano (solo or with orchestra)",
    "Percussion",
]


def get_instrument_groups_for_piece(piece_id: str, has_orchestra: bool) -> list[str]:
    """Return list of instrument group names for this piece. Extensible from metadata later."""
    if not has_orchestra:
        return ["Piano (right hand – melody)", "Piano (left hand – accompaniment)"]
    # Could load from piece_metadata or a separate orchestration.json
    return list(DEFAULT_ORCHESTRA_GROUPS)


def get_instrument_interaction_from_llm(
    piece_id: str,
    piece_title: str,
    composer_name: str,
    has_orchestra: bool,
    llm_complete_json: Any,
) -> dict[str, Any]:
    """
    Use LLM to describe each instrument group's role and one listening cue.
    Returns { "groups": [...], "message": "..." }.
    """
    from llm.prompts import INSTRUMENT_INTERACTION

    groups = get_instrument_groups_for_piece(piece_id, has_orchestra)
    groups_str = ", ".join(groups)

    user = INSTRUMENT_INTERACTION.format(
        piece_title=piece_title,
        composer_name=composer_name,
        instrument_groups=groups_str,
    )
    out = llm_complete_json(
        "You output only valid JSON. No markdown, no explanation.",
        user,
    )
    if not out or not isinstance(out, dict):
        return _default_instrument_interaction(piece_id, has_orchestra)

    return {
        "groups": out.get("groups", []) if isinstance(out.get("groups"), list) else _default_groups(has_orchestra),
        "message": str(out.get("message", "")).strip() or "",
    }


def _default_groups(has_orchestra: bool) -> list[dict]:
    if has_orchestra:
        return [
            {"name": "Strings", "role": "Carry melody and harmony; often the main emotional voice.", "listen_for": "When the main theme returns, listen for the strings leading."},
            {"name": "Woodwinds", "role": "Add color and dialogue; often answer or comment on the melody.", "listen_for": "Notice moments when flutes or clarinets take over the tune."},
            {"name": "Brass", "role": "Power and climaxes; used for emphasis and grandeur.", "listen_for": "Listen for brass in the loudest, most dramatic sections."},
            {"name": "Piano", "role": "Soloist in concertos; melody and virtuosity.", "listen_for": "Follow the piano in solo passages and cadenzas."},
        ]
    return [
        {"name": "Right hand", "role": "Usually carries the melody or main line.", "listen_for": "Notice when the melody moves to the high register or becomes more decorative."},
        {"name": "Left hand", "role": "Provides accompaniment, bass, and harmony.", "listen_for": "Listen for the left hand's wave-like or chordal patterns."},
    ]


def _default_instrument_interaction(piece_id: str, has_orchestra: bool) -> dict[str, Any]:
    return {
        "groups": _default_groups(has_orchestra),
        "message": "Try focusing on one group at a time to hear how the layers fit together." if has_orchestra else "In piano pieces, the dialogue is often between the two hands.",
    }


def get_instrument_interaction(
    piece_id: str,
    piece_title: str,
    composer_name: str,
    has_orchestra: bool,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Get instrument group roles and cues; use LLM if enabled and available."""
    if use_llm:
        try:
            from llm.nemotron_client import complete_json
            return get_instrument_interaction_from_llm(
                piece_id, piece_title, composer_name, has_orchestra,
                lambda sys_p, usr_p: complete_json(sys_p, usr_p, temperature=0.5),
            )
        except Exception:
            pass
    return _default_instrument_interaction(piece_id, has_orchestra)
