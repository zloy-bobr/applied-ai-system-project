"""
What the catalog actually contains.

The agent shows this vocabulary to the language model (so it can only pick labels that
exist) and the guardrails use it to throw away anything the model invents. This is the
"retrieval" part of the system: the model's choices are grounded in the real data.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


# Words people actually type -> the label used in data/songs.csv
GENRE_SYNONYMS: Dict[str, str] = {
    "lo-fi": "lofi",
    "lo fi": "lofi",
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "rap": "hip hop",
    "rnb": "r&b",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "electronic": "edm",
    "dance music": "edm",
    "techno": "edm",
    "house music": "edm",
    "rave": "edm",
    "heavy metal": "metal",
    "orchestral": "classical",
    "piano": "classical",
    "retro wave": "synthwave",
    "retrowave": "synthwave",
    "80s synth": "synthwave",
    "acoustic folk": "folk",
    "singer-songwriter": "folk",
    "ambient music": "ambient",
}

MOOD_SYNONYMS: Dict[str, str] = {
    "cheerful": "happy",
    "joyful": "happy",
    "upbeat": "happy",
    "sunny": "happy",
    "relaxing": "relaxed",
    "relax": "relaxed",
    "laid-back": "relaxed",
    "laid back": "relaxed",
    "calm": "peaceful",
    "sleep": "peaceful",
    "chilled": "chill",
    "chill out": "chill",
    "mellow": "chill",
    "melancholy": "sad",
    "heartbroken": "sad",
    "depressed": "sad",
    "crying": "sad",
    "aggressive": "intense",
    "hype": "intense",
    "rage": "angry",
    "furious": "angry",
    "love": "romantic",
    "date night": "romantic",
    "concentrate": "focused",
    "focus": "focused",
    "studying": "focused",
    "dreamy": "moody",
    "late night": "moody",
    "late-night": "moody",
    "swagger": "confident",
    "ecstatic": "euphoric",
    "hopeful": "uplifting",
    "homesick": "nostalgic",
}


def _lookup(value: Optional[str], labels: List[str], synonyms: Dict[str, str]) -> Optional[str]:
    """Return the catalog label for value (directly or via a synonym), else None."""
    text = str(value or "").strip().lower()
    text = text if text in labels else synonyms.get(text)
    return text if text in labels else None


@dataclass
class CatalogVocab:
    """The exact labels found in the song catalog."""
    genres: List[str]
    moods: List[str]
    mood_tags: List[str]
    decades: List[int]
    languages: List[str]

    @classmethod
    def from_songs(cls, songs: List[Dict]) -> "CatalogVocab":
        def labels(key):
            return sorted({str(song[key]).lower() for song in songs})

        return cls(
            genres=labels("genre"),
            moods=labels("mood"),
            mood_tags=sorted({tag for song in songs for tag in song.get("mood_tags", [])}),
            decades=sorted({int(song["release_decade"]) for song in songs}),
            languages=labels("language"),
        )

    def match_genre(self, value: Optional[str]) -> Optional[str]:
        return _lookup(value, self.genres, GENRE_SYNONYMS)

    def match_mood(self, value: Optional[str]) -> Optional[str]:
        return _lookup(value, self.moods, MOOD_SYNONYMS)

    def as_prompt_text(self) -> str:
        """Compact description of the catalog for the language model."""
        return (
            f"genres: {', '.join(self.genres)}\n"
            f"moods: {', '.join(self.moods)}\n"
            f"mood_tags: {', '.join(self.mood_tags)}\n"
            f"release decades: {', '.join(map(str, self.decades))}\n"
            f"languages: {', '.join(self.languages)}"
        )
