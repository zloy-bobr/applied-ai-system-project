"""
Command line runner.

    python -m src.main                                   # interactive AI chat
    python -m src.main --ask "chill lofi for studying"   # one request through the AI agent
    python -m src.main --ask "..." --trace               # also show every agent step
    python -m src.main --profiles                        # original Module 3 demo (fixed profiles)
    add --offline to ignore API keys and use the keyword parser
"""
import argparse

from src.recommender import load_songs, recommend_songs


PROFILES = {
    "High-Energy Pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
        "target_valence": 0.8,
    },
    "Chill Lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "likes_acoustic": True,
        "target_valence": 0.6,
    },
    "Deep Intense Rock": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.9,
        "likes_acoustic": False,
        "target_valence": 0.45,
    },
    # Edge cases: profiles meant to trick or stress the scoring logic
    "Edge: Sad Banger (conflicting mood vs. energy)": {
        "favorite_genre": "edm",
        "favorite_mood": "sad",
        "target_energy": 0.9,
        "likes_acoustic": False,
        "target_valence": 0.2,
    },
    "Edge: Capitalized Genre ('Pop' vs. 'pop')": {
        "favorite_genre": "Pop",
        "favorite_mood": "Happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    },
    "Edge: Out-of-Range Energy (1.5)": {
        "favorite_genre": "classical",
        "target_energy": 1.5,
    },
    "Edge: Empty Profile": {},
    # Advanced features: mood tags, decade, popularity, instrumentalness, language
    "Advanced: High-Energy Pop + Mood Tags": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
        "target_valence": 0.8,
        "desired_mood_tags": ["happy", "sunny", "feel-good"],
    },
    "Advanced: Nostalgic Late-Night Drive": {
        "favorite_genre": "synthwave",
        "favorite_mood": "moody",
        "target_energy": 0.7,
        "likes_acoustic": False,
        "desired_mood_tags": ["nostalgic", "late-night", "dreamy"],
        "favorite_decade": 2010,
        "target_popularity": 60,
    },
    "Advanced: Underground Instrumental Study": {
        "favorite_mood": "focused",
        "target_energy": 0.35,
        "likes_acoustic": True,
        "desired_mood_tags": ["study", "calm", "cozy"],
        "target_instrumentalness": 0.9,
        "target_popularity": 40,
        "preferred_language": "english",
    },
}


def print_recommendations(name: str, user_prefs: dict, recommendations: list) -> None:
    """Print one profile's top recommendations with scores and reasons."""
    profile = ", ".join(f"{key}={value}" for key, value in user_prefs.items()) or "(none)"
    print(f"\n### {name}")
    print(f"User profile: {profile}")
    print("=" * 60)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} by {song['artist']}")
        print(f"   {song['genre']} / {song['mood']}   Score: {score:.2f}")
        for reason in explanation.split("; ") if explanation else ["(no preferences matched)"]:
            print(f"     - {reason}")
        print("-" * 60)


def run_profiles() -> None:
    """Original Module 3 behavior: score the hand-written profiles above."""
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, user_prefs in PROFILES.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(name, user_prefs, recommendations)


def print_agent_result(result, show_trace: bool = False) -> None:
    print("=" * 64)
    print(f"Request: {result.request}")
    print(f"Model:   {result.provider}")
    if result.clarification:
        print(f"\n🤔 {result.clarification}")
    else:
        if result.interpretation:
            print(f"Understood as: {result.interpretation}")
        print(f"Profile: {result.profile}")
        print(f"Overall match confidence: {result.confidence:.0%}\n")
        for rank, rec in enumerate(result.recommendations, start=1):
            song = rec.song
            print(f"{rank}. {song['title']} by {song['artist']}  [{song['genre']} / {song['mood']}]")
            print(f"   match {rec.confidence:.0%}  (score {rec.score:.2f})")
            print(f"   {rec.explanation}")
        print(f"\n{result.summary}")
    for warning in result.warnings:
        print(f"⚠️  {warning}")
    if show_trace:
        print("\nAgent trace:")
        for step in result.trace:
            ms = f" {step['ms']}ms" if "ms" in step else ""
            print(f"  [{step['step']}] {step['status']}{ms}: {step['detail']}")
    print("=" * 64)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="AI music recommender")
    parser.add_argument("--ask", help="a plain-English music request")
    parser.add_argument("--trace", action="store_true", help="print each agent step")
    parser.add_argument("--offline", action="store_true", help="don't call an LLM")
    parser.add_argument("--profiles", action="store_true", help="run the original Module 3 demo")
    args = parser.parse_args(argv)

    if args.profiles:
        run_profiles()
        return

    from src.ai.agent import RecommendationAgent

    agent = RecommendationAgent(client=None if args.offline else "auto")
    if args.ask:
        print_agent_result(agent.run(args.ask), args.trace)
        return

    print(f"🎵 AI Music Recommender ({agent.provider}). Describe what you want; empty line to quit.")
    while True:
        try:
            request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not request:
            break
        print_agent_result(agent.run(request), args.trace)


if __name__ == "__main__":
    main()
