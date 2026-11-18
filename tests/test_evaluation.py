from src.core.recommender import load_songs
from src.evaluation import run_eval
from src.evaluation.metrics import bias_report, diversity, hit_at_1, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics_known_values():
    ranked, relevant = [1, 2, 3, 4, 5], {1, 3, 9}
    assert precision_at_k(ranked, relevant, 5) == 0.4
    assert recall_at_k(ranked, relevant, 5) == 2 / 3
    assert hit_at_1(ranked, relevant) == 1.0
    assert ndcg_at_k([9, 1, 3], relevant, 3) == 1.0  # all relevant, perfect order
    assert 0 < ndcg_at_k(ranked, relevant, 5) < 1
    assert ndcg_at_k([7, 8], relevant, 5) == 0.0


def test_diversity_and_bias_report():
    songs = load_songs("data/songs.csv")
    assert diversity(songs[:3])["genre_diversity"] == 1.0
    report = bias_report([songs[:5], songs[:5]], songs)
    assert report["slots"] == 10
    assert report["catalog_coverage"] == round(5 / len(songs), 2)
    assert songs[-1]["title"] in report["never_recommended"]


def test_evaluation_harness_passes_offline():
    assert run_eval.main(["--offline"]) == 0
