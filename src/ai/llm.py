"""
Language-model providers (Strategy pattern).

Every provider has the same small interface, `complete_json(system, prompt) -> dict`,
so the agent never cares which model is behind it. Which one runs is picked from
environment variables:

    LLM_PROVIDER=claude | gemini | offline   (optional; auto-detected otherwise)
    ANTHROPIC_API_KEY=...   -> Claude
    GEMINI_API_KEY=...      -> Gemini (GOOGLE_API_KEY also works)

With no key, get_client() returns None and the agent uses its offline rule-based
parser, so the whole project (and every test) still runs without internet access.
"""
import json
import os
import re
from typing import Any, Dict, Optional, Protocol


class LLMError(Exception):
    """The model call failed or returned something we couldn't use."""


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, prompt: str) -> Dict[str, Any]:
        ...


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    if not text:
        raise LLMError("empty reply from model")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"no JSON object in reply: {text[:120]!r}")
    try:
        data = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("model returned JSON that is not an object")
    return data


class ClaudeClient:
    """Anthropic Claude via the official `anthropic` SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise LLMError("pip install anthropic to use Claude") from exc
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set (add it to your .env file)")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
        self.name = f"claude:{self.model}"
        self._client = anthropic.Anthropic(api_key=key)

    def complete_json(self, system: str, prompt: str) -> Dict[str, Any]:
        try:
            reply = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # network, auth, rate limit...
            raise LLMError(f"Claude request failed: {exc}") from exc
        text = "".join(getattr(block, "text", "") for block in reply.content)
        return extract_json(text)


class GeminiClient:
    """Google Gemini via the `google-genai` SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise LLMError("pip install google-genai to use Gemini") from exc
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (add it to your .env file)")
        self._client = genai.Client(api_key=key)
        # Gemini 3 models "think" before answering, which roughly doubles the wait for
        # these small JSON tasks. Default to minimal thinking; GEMINI_THINKING=default turns it back on.
        self.thinking = os.getenv("GEMINI_THINKING", "minimal").strip().lower()

    @property
    def name(self) -> str:  # shows the thinking level, so you can see the speed setting is on
        thinking = "" if self.thinking in ("", "default") else f" (thinking: {self.thinking})"
        return f"gemini:{self.model}{thinking}"

    def _generate(self, system: str, prompt: str) -> str:
        config: Dict[str, Any] = {
            "system_instruction": system,
            "temperature": 0,
            "response_mime_type": "application/json",
        }
        if self.thinking not in ("", "default"):
            config["thinking_config"] = {"thinking_level": self.thinking}
        reply = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        return reply.text or ""

    def complete_json(self, system: str, prompt: str) -> Dict[str, Any]:
        try:
            try:
                text = self._generate(system, prompt)
            except Exception as exc:
                if self.thinking in ("", "default") or "thinking" not in str(exc).lower():
                    raise
                # this model doesn't accept the thinking setting: drop it for this and later calls
                print(f"[llm] {self.model} rejected thinking_level={self.thinking}; using the model default.")
                self.thinking = "default"
                text = self._generate(system, prompt)
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return extract_json(text)


def get_client(provider: Optional[str] = None) -> Optional[LLMClient]:
    """
    Return a model client, or None to run offline.

    Never raises: if the chosen provider's package or key is missing, it prints why
    and falls back to offline mode so the app keeps working.
    """
    try:  # load a .env file if python-dotenv is installed
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    choice = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not choice:
        if os.getenv("ANTHROPIC_API_KEY"):
            choice = "claude"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            choice = "gemini"
        else:
            choice = "offline"

    providers = {"claude": ClaudeClient, "gemini": GeminiClient}
    if choice not in providers:
        if choice != "offline":
            print(f"[llm] unknown LLM_PROVIDER '{choice}'; running in offline mode.")
        return None
    try:
        return providers[choice]()
    except Exception as exc:  # missing package, missing/invalid key, SDK init errors
        print(f"[llm] {choice} unavailable ({exc}); running in offline mode instead.")
        return None
