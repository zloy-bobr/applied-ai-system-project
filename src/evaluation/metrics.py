"""Ranking-quality and bias metrics for the recommender. Pure functions, easy to test."""
import math
from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List, Sequence


def precision_at_k(ranked_ids: Sequence[int], relevant: Iterable[int], k: int = 5) -> float:
    relevant = set(relevant)
    top = list(ranked_ids)[:k]
    return sum(1 for i in top if i in relevant) / k if k else 0.0


def recall_at_k(ranked_ids: Sequence[int], relevant: Iterable[int], k: int = 5) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(1 for i in list(ranked_ids)[:k] if i in relevant) / len(relevant)


def hit_at_1(ranked_ids: Sequence[int], relevant: Iterable[int]) -> float:
    return 1.0 if ranked_ids and ranked_ids[0] in set(relevant) else 0.0


def ndcg_at_k(ranked_ids: Sequence[int], relevant: Iterable[int], k: int = 5) -> float:
    """Normalized discounted cumulative gain with binary relevance (1 = relevant)."""
    relevant = set(relevant)
    dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(list(ranked_ids)[:k]) if i in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def diversity(songs: List[Dict]) -> Dict[str, float]:
    """Share of distinct genres and artists in one list (1.0 = all different)."""
    if not songs:
        return {"genre_diversity": 0.0, "artist_diversity": 0.0}
    return {
        "genre_diversity": len({s["genre"] for s in songs}) / len(songs),
        "artist_diversity": len({s["artist"] for s in songs}) / len(songs),
    }


def bias_report(recommended_lists: List[List[Dict]], catalog: List[Dict]) -> Dict:
    """
    Look across many recommendation lists for systematic skew:
      - which genres / artists get most of the exposure
      - whether recommended songs are more popular than the catalog average
      - which songs are never recommended at all (coverage)
    """
    shown = [song for songs in recommended_lists for song in songs]
    if not shown:
        return {}
    genre_counts = Counter(s["genre"] for s in shown)
    artist_counts = Counter(s["artist"] for s in shown)
    song_counts = Counter(s["title"] for s in shown)
    never = sorted(s["title"] for s in catalog if s["title"] not in song_counts)
    catalog_genres = Counter(s["genre"] for s in catalog)

    # exposure share vs. catalog share for each genre (>1 means over-exposed)
    exposure_ratio = {
        genre: round((genre_counts[genre] / len(shown)) / (catalog_genres[genre] / len(catalog)), 2)
        for genre in catalog_genres
    }
    return {
        "lists": len(recommended_lists),
        "slots": len(shown),
        "top_genres": genre_counts.most_common(5),
        "genre_exposure_ratio": dict(sorted(exposure_ratio.items(), key=lambda kv: -kv[1])),
        "top_artists": artist_counts.most_common(3),
        "most_recommended_songs": song_counts.most_common(5),
        "mean_popularity_recommended": round(mean(s["popularity"] for s in shown), 1),
        "mean_popularity_catalog": round(mean(s["popularity"] for s in catalog), 1),
        "catalog_coverage": round(len(song_counts) / len(catalog), 2),
        "never_recommended": never,
    }
