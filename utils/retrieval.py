"""
Data retrieval: artist→composer matching, top pieces, fun facts.
All logic is data-driven; no hardcoded artist or piece names for matching.
"""

import json
import os
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")


def _load_json(name: str) -> dict:
    path = os.path.join(_DATA, name)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_artist(artist: str) -> str:
    """Normalize for lookup: strip, lowercase, collapse spaces."""
    return " ".join((artist or "").strip().lower().split())


def _hash_01(value: str, salt: str) -> float:
    """
    Deterministic pseudo-random in [0, 1) based on input text.
    Keeps fallback behavior stable for the same unknown artist.
    """
    key = f"{salt}:{value}"
    # Builtin hash is intentionally salted per process; use a simple custom hash instead.
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % 1000) / 1000.0


def _estimate_style_dimensions(artist_input: str) -> dict[str, float]:
    """
    Estimate soft style dimensions for an unknown artist using only the name.

    This is intentionally lightweight and deterministic so that:
    - the same unknown artist always maps to the same style profile
    - logic stays data-driven and easy to extend later
    """
    norm = normalize_artist(artist_input) or "unknown"
    return {
        "emotional_tone": _hash_01(norm, "emotional_tone"),
        "atmosphere": _hash_01(norm, "atmosphere"),
        "texture": _hash_01(norm, "texture"),
        "energy": _hash_01(norm, "energy"),
    }


def _score_fallback_composer(
    artist_style: dict[str, float],
    composer: dict[str, Any],
) -> float:
    """
    Compute a weighted similarity score between the artist's inferred style
    and a gateway composer's style profile.
    """
    comp_style = composer.get("style", {})
    dims = ("emotional_tone", "atmosphere", "texture", "energy")
    # If no style is defined yet, fall back to the composer's base weight.
    if not comp_style:
        return float(composer.get("base_weight", 0.5))

    # Normalized Euclidean distance in style space.
    sq_sum = 0.0
    for d in dims:
        a = float(artist_style.get(d, 0.5))
        b = float(comp_style.get(d, 0.5))
        sq_sum += (a - b) ** 2
    dist = (sq_sum / len(dims)) ** 0.5  # in [0, 1]
    similarity = max(0.0, 1.0 - dist)   # 1 = identical, 0 = opposite

    base = float(composer.get("base_weight", 0.5))
    # Blend base gateway weight with style similarity so both matter.
    return 0.4 * base + 0.6 * similarity


def _pick_fallback_composer(
    artist_input: str,
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], float]:
    """
    Choose a gateway composer for an unknown artist using soft style dimensions.

    Returns (composer_dict, artist_style_profile, confidence_score).
    """
    fallbacks = data.get("_fallback_composers", [])
    if not fallbacks:
        # Safety net: behave like the legacy single-default behavior
        default = data.get("_default_composer", {})
        if not default:
            default = {
                "composer_id": "debussy",
                "composer_name": "Claude Debussy",
                "reason": "Dreamy and atmospheric—a great gateway to classical.",
            }
        return default, _estimate_style_dimensions(artist_input), 0.5

    artist_style = _estimate_style_dimensions(artist_input)
    best = None
    best_score = -1.0
    for comp in fallbacks:
        score = _score_fallback_composer(artist_style, comp)
        if score > best_score:
            best_score = score
            best = comp

    if best is None:
        best = fallbacks[0]
        best_score = 0.5

    # Clamp and round confidence for readability
    confidence = max(0.0, min(1.0, best_score))
    return best, artist_style, confidence


def match_pop_to_composer(artist_input: str) -> dict[str, Any]:
    """
    Match a pop artist to a classical composer.
    - If we have a mapping: return it with composer_id, composer_name, reason, artist_display, confidence, and match_type="exact".
    - If unknown: use a multi-candidate gateway fallback with soft style dimensions to pick the best composer,
      returning match_type="inferred" and a confidence score.
    """
    data = _load_json("artists_to_composers.json")
    mappings = data.get("mappings", [])
    suggestions = data.get("_suggestions_message", "Try different artists from our suggestions.")

    key = normalize_artist(artist_input)
    for m in mappings:
        if m.get("artist_key") == key:
            return {
                "composer_id": m["composer_id"],
                "composer_name": m["composer_name"],
                "reason": m.get("reason", ""),
                "weight": m.get("weight"),
                "suggestion": False,
                 "confidence": float(m.get("weight", 0.95)),
                 "match_type": "exact",
                "artist_display": m.get("artist_display", artist_input.strip()),
            }

    # Unknown or empty artist: infer a gateway composer using style dimensions.
    fallback, style_profile, confidence = _pick_fallback_composer(artist_input, data)
    reason = fallback.get(
        "reason",
        "Based on the mood, atmosphere, and energy implied by the name, this composer is a good gateway match.",
    )
    return {
        "composer_id": fallback.get("composer_id", "debussy"),
        "composer_name": fallback.get("composer_name", "Claude Debussy"),
        "reason": reason,
        "confidence": confidence,
        "match_type": "inferred",
        "style_profile": style_profile,
        "suggestion": True,
        "suggestions_message": suggestions,
        "artist_display": artist_input.strip() or "you",
    }


def get_pieces_for_composer(composer_id: str) -> list[dict[str, Any]]:
    """
    Step 2.2: Build candidate pool of all pieces for a composer.
    From structured data; extensible to many composers and pieces.
    """
    data = _load_json("composer_pieces.json")
    composers = data.get("composers", {}) or {}
    if composer_id not in composers:
        return []
    pieces = composers[composer_id].get("pieces", []) or []
    return list(pieces)


def get_top3_pieces(composer_id: str) -> list[dict[str, Any]]:
    """
    Legacy: return top 3 by score only. Prefer recommendation pipeline for ranked + diverse top 3.
    """
    pieces = get_pieces_for_composer(composer_id)
    sorted_pieces = sorted(pieces, key=lambda p: float(p.get("score", 0)), reverse=True)
    return sorted_pieces[:3]


def get_composer_name(composer_id: str) -> str:
    """Return display name for composer."""
    data = _load_json("composer_pieces.json")
    comp = data.get("composers", {}).get(composer_id, {})
    return comp.get("composer_name", composer_id.replace("_", " ").title())


def get_piece_by_id(piece_id: str) -> Optional[dict[str, Any]]:
    """Return full piece info (title, subtitle, duration, has_orchestra, etc.) from composer_pieces.json."""
    data = _load_json("composer_pieces.json")
    for cid, comp in data.get("composers", {}).items():
        for p in comp.get("pieces", []):
            if p.get("piece_id") == piece_id:
                out = dict(p)
                out["composer_id"] = cid
                out["composer_name"] = comp.get("composer_name", "")
                return out
    return None


def get_raw_facts_for_fun_facts(piece_id: str, composer_id: str) -> list[str]:
    """Gather raw facts for a piece and its composer for the fun-fact LLM."""
    data = _load_json("fun_facts.json")
    facts = []
    by_piece = data.get("by_piece", {})
    by_composer = data.get("by_composer", {})
    facts.extend(by_piece.get(piece_id, []))
    facts.extend(by_composer.get(composer_id, []))
    return facts


def get_audio_info(piece_id: str) -> dict[str, Any]:
    """
    Return audio metadata for a piece: audio_url, cover_art_url, availability_status.
    Data from data/audio_sources.json. Replace mock URLs there with real CDN/API URLs.
    """
    data = _load_json("audio_sources.json")
    sources = (data or {}).get("sources", {}) or {}
    info = sources.get(piece_id) or {}
    audio_url = (info.get("audio_url") or "").strip()
    status = (info.get("availability_status") or "unavailable").strip().lower()
    if not audio_url or status != "available":
        return {
            "audio_url": "",
            "cover_art_url": (info.get("cover_art_url") or "").strip(),
            "availability_status": "unavailable",
        }
    return {
        "audio_url": audio_url,
        "cover_art_url": (info.get("cover_art_url") or "").strip(),
        "availability_status": "available",
    }
