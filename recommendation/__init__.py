"""
Step 2: Recommend Top 3 Pieces (Fine Matching).
Input: Step 1 output (composer, taste profile).
Output: Ranked, diverse top 3 pieces with NL explanations.
"""

from recommendation.piece_features import get_piece_features
from recommendation.ranking import (
    user_taste_from_step1,
    score_piece,
    select_diverse_top_k,
)
from recommendation.explanations import build_explanations

__all__ = [
    "get_piece_features",
    "user_taste_from_step1",
    "score_piece",
    "select_diverse_top_k",
    "build_explanations",
]
