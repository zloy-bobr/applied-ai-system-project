# AI Interactions Log

> ✏️ Draft filled in from the build session. Edit it so it reflects your own experience.

---

## Agentic Workflow (SF8)

**What task did you give the agent?**

I gave Claude my CodePath final-project brief and my repo (the Module 3 music recommender, already moved into `src/core/` with architecture diagrams). I asked it to implement the AI integration from my diagrams: a multi-step recommendation agent with confidence scores, an execution trace, and an evaluation system with metrics, bias detection and reliability tests.

**Prompts used:**

- "I have a final project to do for CodePath where we use our last project to integrate AI in it to solve a problem. I did all of the setup and now I need your help to actually implement it."
- *(add any follow-up prompts you used)*

**What did the agent generate or change?**

- Found that `src/__init__.py`, `src/core/__init__.py` and `tests/__init__.py` were saved as UTF-16 (from a Windows `echo > file`), which made `pytest` fail before running any test. It emptied them.
- Turned `src/recommender.py` (a duplicate copy) into a thin re-export of `src/core/recommender.py`.
- Created `src/ai/` (agent, LLM providers, guardrails, parser, catalog vocabulary, logger), `src/evaluation/` (metrics, bias report, harness), `data/eval_cases.json`, `app.py`, `.env.example`, and 20 new tests.
- Rewrote `src/main.py` (chat, `--ask`, `--trace`, `--profiles`), the README, both Mermaid diagrams, and added section 10 of the model card.
- Ran `pytest` and the evaluation harness after each change.

**What did you verify or fix manually?**

- The evaluation exposed that "angry metal" ranked a sad folk song #2; the keyword rules were changed so "angry" also means high energy.
- Template explanations first listed weak reasons ("acoustic sound" for a 0.22-acoustic song); changed to the highest-point reasons only.
- **To do:** run with a real API key (`python -m src.evaluation.run_eval --save`) and compare against the offline numbers, since the LLM path was only tested with a fake model in the build environment.

---

## Design Pattern (SF10)

**Which design pattern did you use?**

**Strategy**, for the language-model provider.

**How did AI help you brainstorm or implement it?**

The project needed to work with Claude, with Gemini (free tier), and with no API key at all (for tests and graders). Instead of `if provider == ...` checks spread through the agent, the AI suggested one small interface, `complete_json(system, prompt) -> dict`, with one class per provider and a factory that picks one from environment variables.

**How does the pattern appear in your final code?**

- `src/ai/llm.py`: the `LLMClient` protocol, `ClaudeClient`, `GeminiClient`, and `get_client()` (returns `None` for offline mode).
- `src/ai/agent.py`: `RecommendationAgent` receives any client and only calls `complete_json()`.
- `tests/test_ai_agent.py`: `FakeLLM` is a fourth strategy used only in tests, which is what makes the LLM path testable without a network.

---

## Code review pass (SF8, continued)

**Prompt:** "Check the code for possible bugs or logic errors, then try to shorten the code to make it less complex and have the same functionality and check the new modified code for possible bugs or logic errors."

**Bugs found and fixed** (1–8 have regression tests in `tests/test_ai_agent.py`; 9 is UI code and was only checked by reading it, because Streamlit could not be installed in the build environment):
1. "music for working out" matched the *study* rule (`work(ing)?`), giving calm low-energy songs.
2. "1920s" became 2020: the parser threw away the century of four-digit decades.
3. A synonym could beat the word the user actually typed ("chill late-night lofi" → mood "moody").
4. If the model returned `unavailable` as a string, each *letter* became a warning.
5. `on_topic: "false"` (a string) was treated as on-topic.
6. `get_client()` crashed the app when an SDK was installed but the API key was missing.
7. The genre warning counted the diversified list, not the catalog ("only has 1 lofi song" when there were 2 by the same artist).
8. *(introduced during the rewrite and caught in the second review)* exact catalog labels could be remapped through the synonym table.
9. The Streamlit app re-ran the agent, and re-called the paid API, on every widget click.

**Simplified:** merged duplicate matchers, made profile validation table-driven, removed an unused field, shorter reliability checks. About 100 fewer lines; the offline evaluation report is byte-identical before and after.
