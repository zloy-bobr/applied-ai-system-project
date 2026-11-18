"""
Guardrails around the language model.

1. check_request()      - before the model sees anything: empty, too long, prompt injection.
2. sanitize_profile()   - after the model answers: keep only known fields, map labels onto
                          the real catalog, clamp numbers. Only a valid profile reaches the scorer.
3. check_explanations() - the model's explanations may only talk about songs we recommended.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.ai.catalog import CatalogVocab

MAX_REQUEST_CHARS = 500
INJECTION_WARNING = ("Your request contained instructions aimed at the AI. They were ignored; "
                     "only music preferences are used.")
INJECTION = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)"
    r"|disregard (all |the )?(previous|prior|above)|system prompt|you are now"
    r"|pretend (to be|you are)|reveal your|api[_ ]?key"
)
NUMERIC_RANGES = {"target_energy": (0, 1), "target_valence": (0, 1),
                  "target_instrumentalness": (0, 1), "target_popularity": (0, 100)}
ALLOWED_FIELDS = {*NUMERIC_RANGES, "favorite_genre", "favorite_mood", "likes_acoustic",
                  "desired_mood_tags", "favorite_decade", "preferred_language"}
META_FIELDS = {"on_topic", "interpretation", "unavailable"}


class GuardrailError(ValueError):
    """The request can't be processed safely."""


def check_request(text: Any) -> Tuple[str, bool]:
    """Return (cleaned text, injection suspected). Raises GuardrailError for refused input."""
    if not isinstance(text, str):
        raise GuardrailError("Request must be text.")
    cleaned = " ".join(text.split())  # collapse whitespace and newlines
    if not cleaned:
        raise GuardrailError("Please describe the music you want, e.g. 'chill songs for studying'.")
    if len(cleaned) > MAX_REQUEST_CHARS:
        raise GuardrailError(f"Request is {len(cleaned)} characters; please keep it under {MAX_REQUEST_CHARS}.")
    return cleaned, bool(INJECTION.search(cleaned.lower()))


def to_bool(value: Any) -> Optional[bool]:
    """True/False for real booleans and yes/no strings; None if unclear."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in ("true", "yes", "1") else False if text in ("false", "no", "0") else None


def _clamp(value: Any, low: float, high: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else round(min(high, max(low, number)), 2)  # NaN -> None


def _decade(value: Any) -> Optional[int]:
    """1985 -> 1980, "80" -> 1980, "10" -> 2010; None if not a plausible decade."""
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    if 0 <= year < 100:
        year += 1900 if year >= 30 else 2000
    year = year // 10 * 10
    return year if 1900 <= year <= 2030 else None


def sanitize_profile(raw: Dict[str, Any], vocab: CatalogVocab) -> Tuple[Dict[str, Any], List[str]]:
    """Turn the parser's output into a profile the scorer can trust. Returns (profile, notes)."""
    raw = raw if isinstance(raw, dict) else {}
    profile: Dict[str, Any] = {}
    notes: List[str] = []

    unknown = sorted(set(raw) - ALLOWED_FIELDS - META_FIELDS)
    if unknown:
        notes.append(f"Ignored unexpected fields from the model: {', '.join(unknown)}.")

    if raw.get("favorite_genre"):
        profile["favorite_genre"] = vocab.match_genre(raw["favorite_genre"])
        if not profile["favorite_genre"]:
            notes.append(f"The catalog has no '{raw['favorite_genre']}' songs, so genre was left open.")
    if raw.get("favorite_mood"):
        profile["favorite_mood"] = vocab.match_mood(raw["favorite_mood"])
        if not profile["favorite_mood"]:
            notes.append(f"No songs are labeled '{raw['favorite_mood']}'; mood was left open.")

    for key, (low, high) in NUMERIC_RANGES.items():
        profile[key] = _clamp(raw.get(key), low, high)
    if profile["target_popularity"] is not None:
        profile["target_popularity"] = int(profile["target_popularity"])

    profile["likes_acoustic"] = None if raw.get("likes_acoustic") is None else to_bool(raw["likes_acoustic"])
    profile["favorite_decade"] = _decade(raw.get("favorite_decade"))

    tags = raw.get("desired_mood_tags") or []
    tags = re.split(r"[;,]", tags) if isinstance(tags, str) else tags if isinstance(tags, list) else []
    tags = [str(tag).strip().lower() for tag in tags]
    profile["desired_mood_tags"] = list(dict.fromkeys(t for t in tags if t in vocab.mood_tags))[:4]

    language = str(raw.get("preferred_language") or "").strip().lower()
    if language in vocab.languages and language != "instrumental":
        profile["preferred_language"] = language
    elif language not in ("", "any", "none"):
        notes.append(f"No songs in '{language}' in the catalog; language was left open.")

    unavailable = raw.get("unavailable") or []
    for item in [unavailable] if isinstance(unavailable, str) else unavailable if isinstance(unavailable, list) else []:
        notes.append(f"Not in this catalog: {item}.")

    return {key: value for key, value in profile.items() if value not in (None, [])}, notes


def check_explanations(explanations: Any, allowed_titles: List[str], max_chars: int = 300) -> Dict[str, str]:
    """Keep explanations only for songs we recommended (drops hallucinations); trim long ones."""
    if not isinstance(explanations, dict):
        return {}
    by_lower = {title.lower(): title for title in allowed_titles}
    clean: Dict[str, str] = {}
    for title, text in explanations.items():
        real = by_lower.get(str(title).strip().lower())
        if real and isinstance(text, str) and text.strip():
            text = " ".join(text.split())
            clean[real] = text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
    return clean
