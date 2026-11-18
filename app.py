"""
Streamlit UI for the AI music recommender.

    streamlit run app.py
"""
import streamlit as st

from src.ai.agent import RecommendationAgent
from src.ai.llm import get_client

EXAMPLES = [
    "chill lofi beats to study to",
    "something heavy and aggressive for the gym",
    "sad acoustic songs for a rainy night",
    "spanish songs for the beach",
    "k-pop for dancing",
]

st.set_page_config(page_title="AI Music Recommender", page_icon="🎵", layout="centered")


@st.cache_resource
def build_agent(provider: str) -> RecommendationAgent:
    client = None if provider == "offline" else get_client(provider if provider != "auto" else None)
    return RecommendationAgent(client=client)


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Language model", ["auto", "claude", "gemini", "offline"],
                            help="auto picks Claude or Gemini from your API keys, else offline.")
    show_trace = st.checkbox("Show agent reasoning steps", value=True)
    st.caption("The ranking itself always comes from the transparent Module 3 scoring rule. "
               "The AI only reads your request and writes the explanations.")

agent = build_agent(provider)

st.title("🎵 AI Music Recommender")
st.caption(f"Describe what you want to hear. Model in use: `{agent.provider}`")

if "request" not in st.session_state:
    st.session_state.request = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.request = example

request = st.text_input("Your request", key="request",
                        placeholder="e.g. dreamy synthwave for a late night drive")

clicked = st.button("Recommend", type="primary")
# Streamlit reruns this whole script on every click; only call the agent (and the paid API)
# when the request or model changed, or the user pressed Recommend again.
key = (provider, request)
if request and (clicked or st.session_state.get("last_key") != key):
    with st.spinner("Thinking..."):
        st.session_state.last_key, st.session_state.last_result = key, agent.run(request)
result = st.session_state.get("last_result") if request else None

if result:
    if result.clarification:
        st.info(result.clarification)
    else:
        if result.interpretation:
            st.write(f"**Understood as:** {result.interpretation}")
        st.metric("Overall match confidence", f"{result.confidence:.0%}")
        for rank, rec in enumerate(result.recommendations, start=1):
            song = rec.song
            with st.container(border=True):
                st.markdown(f"**{rank}. {song['title']}** — {song['artist']}  \n"
                            f"`{song['genre']}` · `{song['mood']}` · energy {song['energy']}")
                st.progress(rec.confidence, text=f"match {rec.confidence:.0%}")
                st.write(rec.explanation)
                with st.expander("Scoring details"):
                    for reason in rec.reasons:
                        st.write(f"- {reason}")
        st.success(result.summary)

    for warning in result.warnings:
        st.warning(warning)

    if show_trace:
        with st.expander("Agent trace (each step the agent took)"):
            for step in result.trace:
                ms = f" · {step['ms']} ms" if "ms" in step else ""
                st.markdown(f"**{step['step']}** — {step['status']}{ms}")
                st.json(step["detail"], expanded=False)
        with st.expander("Profile sent to the scorer"):
            st.json(result.profile)
