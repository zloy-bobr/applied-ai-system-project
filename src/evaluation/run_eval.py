"""
Evaluation harness: runs the agent on labeled requests and edge cases and prints a report.

    python -m src.evaluation.run_eval            # uses Claude/Gemini if a key is set, else offline
    python -m src.evaluation.run_eval --offline  # force the keyword parser (fully reproducible)
    python -m src.evaluation.run_eval --save     # also write reports/evaluation_report.md

Exit code is 1 if any reliability case fails, so it can run in CI.
"""
import argparse
import json
import os
import sys
from statistics import mean
from typing import Dict, List

from src.ai.agent import RecommendationAgent
from src.ai.guardrails import MAX_REQUEST_CHARS
from src.core.recommender import load_songs
from src.evaluation.metrics import bias_report, diversity, hit_at_1, ndcg_at_k, precision_at_k, recall_at_k

K = 5


def popularity_baseline(songs: List[Dict], k: int = K) -> List[int]:
    """What you'd get with no personalization at all: the k most popular songs."""
    return [s["id"] for s in sorted(songs, key=lambda s: -s["popularity"])[:k]]


def run_ranking(agent: RecommendationAgent, cases: List[Dict], songs: List[Dict]):
    rows, lists = [], []
    baseline = popularity_baseline(songs)
    for case in cases:
        result = agent.run(case["request"])
        ids = [r.song["id"] for r in result.recommendations]
        lists.append([r.song for r in result.recommendations])
        rows.append({
            "request": case["request"],
            "top3": [r.song["title"] for r in result.recommendations[:3]],
            "hit@1": hit_at_1(ids, case["relevant"]),
            "precision@5": precision_at_k(ids, case["relevant"], K),
            "recall@5": recall_at_k(ids, case["relevant"], K),
            "ndcg@5": ndcg_at_k(ids, case["relevant"], K),
            "baseline_ndcg@5": ndcg_at_k(baseline, case["relevant"], K),
            "confidence": result.confidence,
            **diversity(lists[-1]),
        })
    return rows, lists


def run_reliability(agent: RecommendationAgent, cases: List[Dict]):
    rows = []
    for case in cases:
        request = "chill songs " * 60 if case["request"] == "LONG" else case["request"]
        result = agent.run(request)
        expect = case["expect"]
        if expect == "blocked":
            ok = result.blocked
        elif expect == "clarification":
            ok = bool(result.clarification) and not result.recommendations
        elif expect == "warning":
            ok = True
        else:  # recommendations
            ok = len(result.recommendations) == agent.k
        if case.get("warning_contains"):
            ok = ok and any(case["warning_contains"].lower() in w.lower() for w in result.warnings)
        if case.get("top_genre"):
            ok = ok and result.recommendations and result.recommendations[0].song["genre"] == case["top_genre"]
        outcome = ("blocked" if result.blocked else "clarified" if result.clarification
                   else f"{len(result.recommendations)} recs")
        rows.append({"name": case["name"], "expect": expect, "got": outcome, "pass": bool(ok)})

    # Consistency: same request twice -> same list
    first = [r.song["id"] for r in agent.run("chill lofi beats to study to").recommendations]
    second = [r.song["id"] for r in agent.run("chill lofi beats to study to").recommendations]
    rows.append({"name": "consistency (same request twice)", "expect": "identical",
                 "got": "identical" if first == second else "different", "pass": first == second})
    return rows


def format_report(provider: str, ranking: List[Dict], reliability: List[Dict], bias: Dict) -> str:
    lines = [f"# Evaluation report\n", f"Provider: `{provider}`  |  k = {K}  |  {len(ranking)} labeled requests\n"]

    def avg(key):
        return mean(row[key] for row in ranking)

    lines += [
        "## Ranking quality\n",
        "| Metric | Agent | Popularity baseline |",
        "|---|---|---|",
        f"| Hit@1 (top pick is relevant) | {avg('hit@1'):.2f} | – |",
        f"| Precision@5 | {avg('precision@5'):.2f} | – |",
        f"| Recall@5 | {avg('recall@5'):.2f} | – |",
        f"| NDCG@5 | {avg('ndcg@5'):.2f} | {avg('baseline_ndcg@5'):.2f} |",
        f"| Avg confidence | {avg('confidence'):.2f} | – |",
        f"| Genre diversity | {avg('genre_diversity'):.2f} | – |",
        f"| Artist diversity | {avg('artist_diversity'):.2f} | – |",
        "",
        "Precision@5 is capped by small relevant sets (e.g. only 1 r&b song exists), so Recall and NDCG are the fairer measures.\n",
        "| Request | Top 3 | Hit@1 | Recall@5 | NDCG@5 | Conf. |",
        "|---|---|---|---|---|---|",
    ]
    for row in ranking:
        lines.append(f"| {row['request']} | {', '.join(row['top3'])} | {row['hit@1']:.0f} | "
                     f"{row['recall@5']:.2f} | {row['ndcg@5']:.2f} | {row['confidence']:.2f} |")

    passed = sum(row["pass"] for row in reliability)
    lines += ["", f"## Reliability and guardrails: {passed}/{len(reliability)} passed\n",
              "| Case | Expected | Got | Pass |", "|---|---|---|---|"]
    for row in reliability:
        lines.append(f"| {row['name']} | {row['expect']} | {row['got']} | {'✅' if row['pass'] else '❌'} |")

    lines += ["", "## Bias check (across all labeled requests)\n",
              f"- Catalog coverage: {bias['catalog_coverage']:.0%} of songs were recommended at least once",
              f"- Never recommended: {', '.join(bias['never_recommended']) or 'none'}",
              f"- Most recommended songs: {', '.join(f'{t} ({n})' for t, n in bias['most_recommended_songs'])}",
              f"- Top artists: {', '.join(f'{a} ({n})' for a, n in bias['top_artists'])}",
              f"- Mean popularity of recommendations: {bias['mean_popularity_recommended']} "
              f"(catalog average {bias['mean_popularity_catalog']})",
              f"- Genre exposure vs. catalog share (>1 = over-exposed): "
              + ", ".join(f"{g} {r}" for g, r in bias["genre_exposure_ratio"].items())]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="ignore API keys; use the keyword parser")
    parser.add_argument("--save", action="store_true", help="write reports/evaluation_report.md")
    parser.add_argument("--cases", default="data/eval_cases.json")
    args = parser.parse_args(argv)

    songs = load_songs("data/songs.csv")
    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    agent = RecommendationAgent(songs, client=None if args.offline else "auto", log_path=None)

    ranking, lists = run_ranking(agent, cases["ranking_cases"], songs)
    reliability = run_reliability(agent, cases["reliability_cases"])
    report = format_report(agent.provider, ranking, reliability, bias_report(lists, songs))
    print(report)

    if args.save:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", "evaluation_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved {path}")
    return 0 if all(row["pass"] for row in reliability) else 1


if __name__ == "__main__":
    sys.exit(main())
