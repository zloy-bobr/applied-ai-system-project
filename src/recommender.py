"""Backward-compatible import path. The real code lives in src/core/recommender.py."""
from src.core.recommender import (  # noqa: F401
    Song,
    UserProfile,
    Recommender,
    load_songs,
    score_song,
    recommend_songs,
)
