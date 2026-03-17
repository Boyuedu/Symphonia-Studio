"""
Step 2.4: Piece-level scoring and diversity-aware top-k selection.
Uses Step 1 output (user taste) and piece features.
"""

from typing import Any, Callable

from recommendation.piece_features import FEATURE_DIMS, get_piece_features

# Weights for composite piece score. Easy to adjust.
WEIGHTS = {
    "similarity": 0.35,      # match to user taste
    "representativeness": 0.30,  # importance / recognizability of piece for composer
    "accessibility": 0.20,   # welcoming for new listeners
    "diversity_penalty_scale": 0.15,  # how much to penalize similarity to already selected
}


def user_taste_from_step1(step1_result: dict[str, Any]) -> dict[str, float]:
    """
    Step 2.1 input: build user taste profile from Step 1 output.
    Uses style_profile when present (inferred match), else neutral profile.
    """
    profile = step1_result.get("style_profile")
    if isinstance(profile, dict) and profile:
        dims = {d: max(0.0, min(1.0, float(profile.get(d, 0.5)))) for d in FEATURE_DIMS if d in ("emotional_tone", "atmosphere", "texture", "energy")}
        base = {d: 0.5 for d in FEATURE_DIMS}
        base.update(dims)
        return base
    return {d: 0.5 for d in FEATURE_DIMS}


def _similarity(user: dict[str, float], features: dict[str, float]) -> float:
    """Cosine-like similarity over overlapping dimensions (user has emotional_tone, atmosphere, texture, energy)."""
    overlap = ["emotional_tone", "atmosphere", "texture", "energy"]
    u = [float(user.get(d, 0.5)) for d in overlap]
    f = [float(features.get(d, 0.5)) for d in overlap]
    dot = sum(a * b for a, b in zip(u, f))
    norm_u = (sum(x * x for x in u)) ** 0.5 or 1e-6
    norm_f = (sum(x * x for x in f)) ** 0.5 or 1e-6
    return max(0.0, min(1.0, dot / (norm_u * norm_f)))


def score_piece(
    piece: dict[str, Any],
    features: dict[str, float],
    user_taste: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Composite score: similarity to user + representativeness + accessibility.
    """
    w = weights or WEIGHTS
    sim = _similarity(user_taste, features)
    rep = float(piece.get("score", 0.5))
    acc = float(features.get("accessibility", 0.5))
    return (
        w.get("similarity", 0.35) * sim
        + w.get("representativeness", 0.30) * rep
        + w.get("accessibility", 0.20) * acc
    )


def _feature_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance in feature space (normalized 0-1)."""
    d = sum((a.get(k, 0.5) - b.get(k, 0.5)) ** 2 for k in FEATURE_DIMS)
    return (d / len(FEATURE_DIMS)) ** 0.5


def select_diverse_top_k(
    candidates: list[dict[str, Any]],
    user_taste: dict[str, float],
    k: int = 3,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Select k pieces that balance quality (score) and diversity.
    Iterative: pick 1st by score; then each next by score minus diversity penalty to already selected.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)

    w = weights or WEIGHTS
    div_scale = w.get("diversity_penalty_scale", 0.15)
    selected: list[dict[str, Any]] = []
    feats_cache: dict[str, dict[str, float]] = {}

    def get_f(p: dict[str, Any]) -> dict[str, float]:
        pid = p.get("piece_id", "")
        if pid not in feats_cache:
            feats_cache[pid] = get_piece_features(p)
        return feats_cache[pid]

    def base_score(p: dict[str, Any]) -> float:
        return score_piece(p, get_f(p), user_taste, w)

    def diversity_bonus(p: dict[str, Any]) -> float:
        if not selected:
            return 0.0
        fp = get_f(p)
        return max(_feature_distance(fp, get_f(s)) for s in selected)  # high distance = bonus

    remaining = list(candidates)
    for _ in range(k):
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda p: base_score(p) + div_scale * diversity_bonus(p),
        )
        selected.append(best)
        remaining.remove(best)

    return selected
