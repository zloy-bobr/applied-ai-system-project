"""
Streamlit UI for the AI music recommender.

    streamlit run app.py
"""
import os

import streamlit as st

from src.ai.agent import AgentResult, RecommendationAgent
from src.ai.guardrails import MAX_REQUEST_CHARS
from src.ai.llm import get_client

try:  # read API keys from .env before deciding which models to offer
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

EXAMPLES = [
    "chill lofi beats to study to",
    "something heavy and aggressive for the gym",
    "sad acoustic songs for a rainy night",
    "spanish songs for the beach",
    "k-pop for dancing",
]
MODEL_LABELS = {"gemini": "Gemini", "claude": "Claude", "offline": "Offline (no AI)"}

st.set_page_config(page_title="VibeMatch AI", page_icon="🎵", layout="centered")


def available_models() -> list:
    """Offer only the models that have an API key, so every option actually works."""
    if os.getenv("LLM_PROVIDER", "").strip().lower() == "offline":
        return ["offline"]
    models = []
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        models.append("gemini")
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude")
    return models + ["offline"]


@st.cache_resource
def build_agent(provider: str, ai_explanations: bool) -> RecommendationAgent:
    client = None if provider == "offline" else get_client(provider)
    return RecommendationAgent(client=client, ai_explanations=ai_explanations,
                               log_path=os.getenv("AGENT_LOG", "logs/agent_runs.jsonl"))


@st.cache_data(ttl=3600, max_entries=200, show_spinner="Finding songs...")
def recommend(provider: str, ai_explanations: bool, request: str) -> AgentResult:
    """
    Run the agent once per (model, setting, request). Streamlit reruns the script on every
    interaction; this cache makes a duplicate submit wait for the first answer instead of
    calling the model again, and makes repeated requests instant.
    """
    return build_agent(provider, ai_explanations).run(request)


def submit_search() -> None:
    st.session_state.query = st.session_state.draft.strip()


def use_example() -> None:
    example = st.session_state.example
    if example:  # clicking the selected pill again unselects it; ignore that
        st.session_state.draft = st.session_state.query = example
    st.session_state.example = None  # make the pills behave like buttons


def show_song(rank: int, rec) -> None:
    song = rec.song
    with st.container(border=True):
        info, score = st.columns([4, 1], vertical_alignment="center")
        info.markdown(
            f"**{rank}. {song['title']}** — {song['artist']}  \n"
            f":gray-badge[{song['genre']}] :gray-badge[{song['mood']}] :gray-badge[energy {song['energy']}]"
        )
        score.metric("Match", f"{rec.confidence:.0%}")
        st.write(rec.explanation)
        with st.expander("Why this score"):
            for reason in rec.reasons:
                st.write(f"- {reason}")


def show_details(result: AgentResult) -> None:
    with st.expander("How the agent decided"):
        seconds = {s["step"]: s["ms"] / 1000 for s in result.trace if s["step"] in ("plan", "explain") and "ms" in s}
        if seconds:
            st.caption("Time spent: " + ", ".join(f"{step} {sec:.1f} s" for step, sec in seconds.items())
                       + " (asking the same thing again is instant)")
        st.markdown("**Taste profile sent to the scorer**")
        st.json(result.profile)
        st.markdown("**Each step**")
        for step in result.trace:
            ms = f" · {step['ms']} ms" if "ms" in step else ""
            st.markdown(f"`{step['step']}` {step['status']}{ms}")
            st.json(step["detail"], expanded=False)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    models = available_models()
    provider = st.selectbox("AI model", models, format_func=MODEL_LABELS.get,
                            help="Only models with an API key in your .env file are listed.")
    if models == ["offline"]:
        st.info("No API key found. Add GEMINI_API_KEY to your .env file and restart the app to use AI.")
    ai_explanations = st.toggle(
        "AI-written explanations", value=True, disabled=provider == "offline",
        help="Off = one AI call instead of two, so about twice as fast. "
             "Explanations then come straight from the scoring rules.")
    st.divider()
    st.caption("The AI reads your request and writes the explanations. The ranking itself "
               "always comes from the transparent Module 3 scoring rule over an 18-song catalog.")

agent = build_agent(provider, ai_explanations)

# ---------------------------------------------------------------- main page
st.title("🎵 VibeMatch AI")
st.caption("Describe the music you want in your own words and get five songs, each with a reason.")

st.session_state.setdefault("query", "")
with st.form("search", border=False):
    box, button = st.columns([5, 1], vertical_alignment="bottom")
    box.text_input("What do you want to hear?", key="draft", max_chars=MAX_REQUEST_CHARS,
                   placeholder="e.g. dreamy synthwave for a late night drive")
    button.form_submit_button("Search", type="primary", width="stretch", key="search_button",
                              on_click=submit_search)
st.pills("Or try one:", EXAMPLES, key="example", on_change=use_example)

if agent.client:
    st.caption(f"AI model: `{agent.provider}`")
elif provider != "offline":
    st.warning(f"{MODEL_LABELS[provider]} could not start (usually a wrong API key in .env), "
               "so the app is using offline keyword matching. The terminal window shows the reason.")
else:
    st.caption("Offline mode: keyword matching, no AI.")

query = st.session_state.query
if query:
    result = recommend(provider, ai_explanations, query)
    for warning in result.warnings:
        st.warning(warning)
    if result.clarification:
        st.info(result.clarification)
    else:
        st.subheader("Your picks")
        if result.interpretation:
            st.markdown(f"**Understood as:** {result.interpretation}")
        match = f"Overall match {result.confidence:.0%}"
        st.caption(result.summary if match in result.summary else f"{result.summary}  ·  {match}")
        for rank, rec in enumerate(result.recommendations, start=1):
            show_song(rank, rec)
    show_details(result)
