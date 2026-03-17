"""
Step 2.3: Piece feature tags for ranking and explanation.
Structured feature representation derived from data or tags.
Extensible: support explicit 'features' in piece data or derive from tags/score.
"""

from typing import Any

# Feature dimensions used for similarity and diversity. All in [0, 1].
FEATURE_DIMS = (
    "emotional_tone",
    "atmosphere",
    "energy",
    "texture",
    "familiarity",
    "accessibility",
    "lyrical_quality",
    "dramatic_level",
)

# Tag keywords -> (dimension, value). Used when piece has no explicit 'features'.
_TAG_TO_FEATURE: list[tuple[set[str], str, float]] = [
    ({"dreamy", "soft", "reflective", "contemplative"}, "emotional_tone", 0.3),
    ({"dramatic", "powerful", "intense", "triumphant"}, "emotional_tone", 0.95),
    ({"lyrical", "melodic", "song", "song-like"}, "emotional_tone", 0.75),
    ({"atmospheric", "evocative", "impressionist"}, "atmosphere", 0.9),
    ({"intimate", "minimal", "sparse"}, "atmosphere", 0.4),
    ({"light", "flowing", "gentle", "floating"}, "energy", 0.35),
    ({"dramatic", "powerful", "virtuosic", "driving"}, "energy", 0.85),
    ({"hypnotic", "repetitive"}, "energy", 0.5),
    ({"piano", "chamber"}, "texture", 0.4),
    ({"orchestral", "symphonic", "lush"}, "texture", 0.9),
    ({"minimal", "sparse"}, "texture", 0.2),
    ({"romantic", "nocturne", "ballad"}, "lyrical_quality", 0.85),
    ({"narrative", "ballade"}, "dramatic_level", 0.7),
    ({"dramatic", "orchestral"}, "dramatic_level", 0.9),
    ({"light", "graceful", "accessible"}, "accessibility", 0.9),
    ({"gentle", "flowing", "melodic"}, "accessibility", 0.8),
    ({"virtuosic", "complex"}, "accessibility", 0.4),
]


def _default_features() -> dict[str, float]:
    """Neutral defaults for each dimension."""
    return {d: 0.5 for d in FEATURE_DIMS}


def _derive_from_tags(tags: list[str]) -> dict[str, float]:
    """Build feature vector from tag list. Multiple tags can affect same dimension (max)."""
    f: dict[str, float] = _default_features()
    tag_set = {t.lower().strip() for t in (tags or [])}
    for keyword_set, dim, value in _TAG_TO_FEATURE:
        if keyword_set & tag_set:
            f[dim] = max(f.get(dim, 0.5), value)
    return f


def _apply_score_to_representativeness(features: dict[str, float], score: float) -> None:
    """Use piece 'score' (importance/representativeness) to set familiarity/accessibility hints."""
    s = max(0.0, min(1.0, float(score)))
    features["familiarity"] = max(features.get("familiarity", 0.5), s * 0.9)
    features["accessibility"] = max(features.get("accessibility", 0.5), s * 0.85)


def get_piece_features(piece: dict[str, Any]) -> dict[str, float]:
    """
    Return a structured feature vector for a piece.
    Uses piece['features'] if present (extensible), else derives from tags and score.
    """
    explicit = piece.get("features")
    if isinstance(explicit, dict) and explicit:
        f = _default_features()
        for d in FEATURE_DIMS:
            if d in explicit and isinstance(explicit[d], (int, float)):
                f[d] = max(0.0, min(1.0, float(explicit[d])))
        return f

    tags = piece.get("tags") or []
    f = _derive_from_tags(tags)
    score = piece.get("score")
    if score is not None:
        _apply_score_to_representativeness(f, score)
    return f
