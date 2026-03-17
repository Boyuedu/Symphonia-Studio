"""
Step 2.5: Natural language explanations for the top 3 pieces.
Warm, concise, accessible. Reusable across composers and pieces.
"""

from typing import Any

from recommendation.piece_features import get_piece_features


def _first_tags(piece: dict[str, Any], n: int = 2) -> list[str]:
    """First n tags for template use."""
    tags = piece.get("tags") or []
    return [str(t) for t in tags[:n]]


def build_why_match(match_reason: str, piece: dict[str, Any]) -> str:
    """
    One short sentence: why this piece fits the user.
    Uses match_reason from Step 1 and piece tags. No hardcoded examples.
    """
    reason = (match_reason or "").strip()
    tags = _first_tags(piece, 2)
    if not reason:
        reason = "your taste in mood and atmosphere"
    if tags:
        return f"This piece fits {reason}—{tags[0]}" + (f" and {tags[1]}" if len(tags) > 1 else "") + " make it a great match."
    return f"This piece fits {reason}—a welcoming choice to explore."


def build_intro(composer_name: str, match_reason: str) -> str:
    """One sentence intro for the top 3 block."""
    name = (composer_name or "this composer").strip()
    if match_reason:
        return f"Here are three pieces by {name} that fit your taste—each in a different way."
    return f"Here are three pieces by {name} we think you'll enjoy."


def build_explanations(
    top3_pieces: list[dict[str, Any]],
    composer_name: str,
    match_reason: str,
    fun_facts_by_piece_id: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Build intro and per-piece explanations (tagline = why it matches, fun fact).
    fun_facts_by_piece_id: optional map piece_id -> list of fact strings (from API or LLM).
    Returns dict with 'intro' and 'pieces' (list of { piece_id, title, tagline, fun_fact }).
    """
    fun_facts_by_piece_id = fun_facts_by_piece_id or {}
    intro = build_intro(composer_name, match_reason)
    pieces_out = []
    for p in top3_pieces:
        piece_id = p.get("piece_id", "")
        title = p.get("title", "")
        tagline = build_why_match(match_reason, p)
        facts = fun_facts_by_piece_id.get(piece_id, [])
        fun_fact = facts[0] if facts else "A beloved piece from the repertoire—worth discovering."
        pieces_out.append({
            "piece_id": piece_id,
            "title": title,
            "tagline": tagline,
            "fun_fact": fun_fact,
            **{k: v for k, v in p.items() if k not in ("piece_id", "title", "tagline")},
        })
    return {"intro": intro, "pieces": pieces_out}
