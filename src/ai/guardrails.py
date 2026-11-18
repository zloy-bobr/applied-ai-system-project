"""
Guardrails around the language model.

1. check_request()   - before the model sees anything: empty, too long, prompt-injection.
2. sanitize_profile() - after the model answers: keep only known fields, map labels onto
                        the real catalog, clamp numbers into range. Whatever the model says,
                        only a valid profile ever reaches the Module 3 scorer.
3. check_explanations() - the model's explanations may only talk about songs we recommended.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.ai.catalog import CatalogVocab

MAX_REQUEST_CHARS = 500

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"system prompt",
    r"you are now",
    r"pretend (to be|you are)",
    r"reveal your",
    r"api[_ ]?key",
]


class GuardrailError(ValueError):
    """The request can't be processed safely."""


@dataclass
class RequestCheck:
    text: str
    warnings: List[str] = field(default_factory=list)
    injection_suspected: bool = False


def check_request(text: Any) -> RequestCheck:
    """Validate the raw user text. Raises GuardrailError for requests we refuse outright."""
    if not isinstance(text, str):
        raise GuardrailError("Request must be text.")
    cleaned = " ".join(text.split())  # collapse whitespace and newlines
    if not cleaned:
        raise GuardrailError("Please describe the music you want, e.g. 'chill songs for studying'.")
    if len(cleaned) > MAX_REQUEST_CHARS:
        raise GuardrailError(
            f"Request is {len(cleaned)} characters; please keep it under {MAX_REQUEST_CHARS}."
        )

    result = RequestCheck(text=cleaned)
    lowered = cleaned.lower()
    if any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS):
        result.injection_suspected = True
        result.warnings.append(
            "Your request contained instructions aimed at the AI. They were ignored; "
            "only music preferences are used."
        )
    return result


ALLOWED_FIELDS = {
    "favorite_genre", "favorite_mood", "target_energy", "likes_acoustic", "target_valence",
    "desired_mood_tags", "favorite_decade", "target_popularity", "target_instrumentalness",
    "preferred_language",
}


def _clamp(value: Any, low: float, high: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(min(high, max(low, number)), 2)


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "0"):
        return False
    return None


def sanitize_profile(raw: Dict[str, Any], vocab: CatalogVocab) -> Tuple[Dict[str, Any], List[str]]:
    """
    Turn whatever the parser produced into a profile the scorer can trust.
    Returns (profile, notes) where notes explain anything dropped or changed.
    """
    notes: List[str] = []
    profile: Dict[str, Any] = {}
    raw = raw if isinstance(raw, dict) else {}

    unknown = sorted(set(raw) - ALLOWED_FIELDS - {"on_topic", "interpretation", "unavailable"})
    if unknown:
        notes.append(f"Ignored unexpected fields from the model: {', '.join(unknown)}.")

    if raw.get("favorite_genre"):
        genre = vocab.match_genre(str(raw["favorite_genre"]))
        if genre:
            profile["favorite_genre"] = genre
        else:
            notes.append(
                f"The catalog has no '{raw['favorite_genre']}' songs, so genre was left open."
            )

    if raw.get("favorite_mood"):
        mood = vocab.match_mood(str(raw["favorite_mood"]))
        if mood:
            profile["favorite_mood"] = mood
        else:
            notes.append(f"No songs are labeled '{raw['favorite_mood']}'; mood was left open.")

    for key in ("target_energy", "target_valence", "target_instrumentalness"):
        if raw.get(key) is not None:
            value = _clamp(raw[key], 0.0, 1.0)
            if value is not None:
                profile[key] = value

    if raw.get("target_popularity") is not None:
        value = _clamp(raw["target_popularity"], 0, 100)
        if value is not None:
            profile["target_popularity"] = int(value)

    if raw.get("likes_acoustic") is not None:
        value = _to_bool(raw["likes_acoustic"])
        if value is not None:
            profile["likes_acoustic"] = value

    tags = raw.get("desired_mood_tags") or []
    if isinstance(tags, str):
        tags = re.split(r"[;,]", tags)
    kept = []
    for tag in tags if isinstance(tags, list) else []:
        tag = str(tag).strip().lower()
        if tag in vocab.mood_tags and tag not in kept:
            kept.append(tag)
    if kept:
        profile["desired_mood_tags"] = kept[:4]

    if raw.get("favorite_decade") is not None:
        try:
            decade = int(float(raw["favorite_decade"])) // 10 * 10
            if decade < 100:  # "80" -> 1980, "10" -> 2010
                decade += 1900 if decade >= 30 else 2000
            if 1900 <= decade <= 2030:
                profile["favorite_decade"] = decade
        except (TypeError, ValueError):
            pass

    if raw.get("preferred_language"):
        language = str(raw["preferred_language"]).strip().lower()
        if language in vocab.languages and language != "instrumental":
            profile["preferred_language"] = language
        elif language not in ("", "any", "none"):
            notes.append(f"No songs in '{language}' in the catalog; language was left open.")

    return profile, notes


def check_explanations(explanations: Any, allowed_titles: List[str], max_chars: int = 300) -> Dict[str, str]:
    """
    Keep only explanations for songs we actually recommended, and trim long ones.
    Anything the model wrote about other songs (a hallucination) is dropped.
    """
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
