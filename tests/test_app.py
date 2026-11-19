"""
Tests for the Streamlit app (app.py) using Streamlit's built-in AppTest.

They run the real app script without a browser, in offline mode, so no API key or
network is needed. Skipped automatically if Streamlit isn't installed.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "offline")  # never call a real model from tests
    monkeypatch.setenv("AGENT_LOG", str(tmp_path / "runs.jsonl"))  # keep the real log clean
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def page_text(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


def test_starts_with_search_box_and_no_results(app):
    assert app.text_input(key="draft").value == ""
    assert app.pills(key="example").value is None
    assert "Your picks" not in [h.value for h in app.subheader]
    assert app.selectbox[0].options == ["Offline (no AI)"]  # only models that can actually work


def test_search_button_shows_five_songs(app):
    app.text_input(key="draft").set_value("chill lofi beats to study to")
    app.button(key="search_button").click().run()
    assert not app.exception
    assert "Your picks" in [h.value for h in app.subheader]
    assert "**1. Midnight Coding**" in page_text(app)
    assert len([m for m in app.metric if m.label == "Match"]) == 5


def test_example_pill_runs_search_and_fills_box(app):
    app.pills(key="example").set_value("something heavy and aggressive for the gym").run()
    assert not app.exception
    assert app.text_input(key="draft").value == "something heavy and aggressive for the gym"
    assert app.pills(key="example").value is None  # pills act like buttons
    assert "**1. Gym Hero**" in page_text(app)


def test_missing_genre_shows_warning(app):
    app.text_input(key="draft").set_value("k-pop for dancing")
    app.button(key="search_button").click().run()
    assert any("k-pop" in w.value for w in app.warning)


def test_off_topic_request_asks_for_clarification(app):
    app.text_input(key="draft").set_value("what is the capital of France?")
    app.button(key="search_button").click().run()
    assert app.info and "Your picks" not in [h.value for h in app.subheader]
