"""
Streamlit UI for the AI music recommender.

    streamlit run app.py

Dark theme colors live in .streamlit/config.toml; the CSS below styles the song cards.
"""
import os
from html import escape

import streamlit as st

from src.ai.agent import AgentResult, RecommendationAgent
from src.ai.guardrails import MAX_REQUEST_CHARS
from src.ai.llm import get_client

try:  # read API keys from .env before deciding which models to offer
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# DESIGN SYSTEM - Polish & Elegance Theme
# ============================================================================

EXAMPLES = [
    "chill lofi beats to study to",
    "something heavy and aggressive for the gym",
    "sad acoustic songs for a rainy night",
    "spanish songs for the beach",
    "k-pop for dancing",
]
MODEL_LABELS = {"gemini": "Gemini", "claude": "Claude", "offline": "Offline (no AI)"}

# Custom CSS for sophisticated styling
CUSTOM_CSS = """
<style>
:root {
    --primary: #8B5CF6;
    --accent: #14B8A6;
    --gold: #F59E0B;
    --bg-dark: #0F1419;
    --text: #F8F7FF;
    --text-secondary: #cbd5e1;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark);
    color: var(--text);
}

/* Typography */
h1 {
    font-family: Georgia, 'Garamond', serif;
    font-size: 2.5rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

h2 {
    font-family: Georgia, 'Garamond', serif;
    font-size: 1.75rem;
    font-weight: 400;
}

/* Input styling */
[data-testid="textInputRootElement"] input {
    background-color: rgba(20, 184, 166, 0.05);
    border: 2px solid var(--accent);
    color: var(--text);
    font-size: 1.1rem;
    padding: 1rem;
    border-radius: 12px;
    transition: all 0.3s ease;
}

[data-testid="textInputRootElement"] input:focus {
    background-color: rgba(20, 184, 166, 0.08);
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
}

[data-testid="textInputRootElement"] input::placeholder {
    color: rgba(248, 247, 255, 0.5);
}

/* Button styling */
button[type="submit"] {
    background-color: var(--primary);
    color: var(--text);
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 1rem;
    transition: all 0.3s ease;
}

button[type="submit"]:hover {
    background-color: #A78BFA;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3);
}

/* Container styling */
[data-testid="stContainer"] {
    border-radius: 12px;
    background: rgba(26, 31, 46, 0.6);
    border: 1px solid rgba(20, 184, 166, 0.2);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

[data-testid="stContainer"]:hover {
    border-color: rgba(139, 92, 246, 0.4);
    background: rgba(26, 31, 46, 0.8);
}

/* Pills/example buttons */
[data-testid="stPillContainer"] button {
    background-color: rgba(139, 92, 246, 0.1);
    border: 1px solid var(--accent);
    color: var(--text);
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    transition: all 0.3s ease;
}

[data-testid="stPillContainer"] button:hover {
    background-color: rgba(20, 184, 166, 0.1);
    border-color: var(--primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(10, 10, 20, 0.5);
    border-right: 1px solid rgba(20, 184, 166, 0.1);
}

/* Warnings and info */
[data-testid="stWarning"] {
    background-color: rgba(245, 158, 11, 0.1);
    border-left: 3px solid var(--gold);
}

[data-testid="stInfo"] {
    background-color: rgba(20, 184, 166, 0.1);
    border-left: 3px solid var(--accent);
}

/* JSON display */
[data-testid="stJson"] {
    background-color: rgba(10, 10, 20, 0.8);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 8px;
    padding: 1rem;
}

/* Metric styling */
[data-testid="metric-container"] {
    background-color: rgba(139, 92, 246, 0.05);
    border-left: 3px solid var(--gold);
}

/* Caption and secondary text */
.stCaption {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

/* Markdown links */
div[data-testid="stMarkdown"] a {
    color: var(--accent);
    text-decoration: none;
    transition: color 0.2s ease;
}

div[data-testid="stMarkdown"] a:hover {
    color: var(--primary);
}

/* Strong text accent */
div[data-testid="stMarkdown"] strong {
    color: var(--accent);
    font-weight: 600;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    h2 {
        font-size: 1.5rem;
    }
    [data-testid="stContainer"] {
        padding: 1rem;
    }
}
</style>
"""

st.set_page_config(page_title="VibeMatch AI", page_icon="🎵", layout="centered")

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


@st.cache_data(ttl=3600, max_entries=200, show_spinner="🎵 Finding your songs...")
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


def show_confidence_bar(confidence: float) -> None:
    """Display confidence as a visual progress bar with color gradient."""
    progress_css = f"""
    <div style="margin: 0.75rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 0.9rem; color: var(--text-secondary);">Match confidence</span>
            <span style="font-weight: 600; color: {'var(--accent)' if confidence >= 0.7 else 'var(--gold)' if confidence >= 0.5 else 'var(--primary)'};">
                {confidence:.0%}
            </span>
        </div>
        <div style="
            width: 100%;
            height: 6px;
            background-color: rgba(20, 184, 166, 0.1);
            border-radius: 3px;
            overflow: hidden;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
        ">
            <div style="
                width: {confidence:.0%};
                height: 100%;
                background: linear-gradient(90deg, var(--primary), var(--accent));
                border-radius: 3px;
                transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
                box-shadow: 0 0 8px rgba(139, 92, 246, 0.4);
            "></div>
        </div>
    </div>
    """
    st.markdown(progress_css, unsafe_allow_html=True)


def show_song(rank: int, rec) -> None:
    song = rec.song
    with st.container(border=True):
        # Rank, title, artist (escaped, since they're inserted as HTML)
        st.markdown(
            f"<h3 style='margin: 0; font-family: Georgia, serif; font-size: 1.3rem;'>"
            f"{rank}. {escape(song['title'])}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='margin: 0.25rem 0 0 0; color: var(--text-secondary);'>"
            f"{escape(song['artist'])}</p>",
            unsafe_allow_html=True
        )

        # Genre/Mood/Energy tags
        tags_html = ""
        for value in [escape(song['genre']), escape(song['mood']), f"⚡ {song['energy']:.1f}"]:
            tags_html += f"""
            <span style='
                display: inline-block;
                background-color: rgba(139, 92, 246, 0.15);
                border: 1px solid rgba(139, 92, 246, 0.3);
                color: var(--text-secondary);
                padding: 0.3rem 0.75rem;
                border-radius: 6px;
                font-size: 0.85rem;
                margin-right: 0.5rem;
                margin-top: 0.75rem;
            '>{value}</span>
            """
        st.markdown(tags_html, unsafe_allow_html=True)

        # Confidence bar
        show_confidence_bar(rec.confidence)

        # Explanation
        st.write(rec.explanation)

        # Reasons in expander
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


# ================================================================ sidebar
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

# ================================================================ main page
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
