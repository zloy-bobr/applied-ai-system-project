"""
What the catalog actually contains.

The agent shows this vocabulary to the language model (so it can only pick labels that
exist) and the guardrails use it to throw away anything the model invents. This is the
"retrieval" part of the system: the model's choices are grounded in the real data.
"""
from dataclasses import dataclass, field
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


@dataclass
class CatalogVocab:
    """The exact labels found in the song catalog."""
    genres: List[str]
    moods: List[str]
    mood_tags: List[str]
    decades: List[int]
    languages: List[str]
    titles: List[str] = field(default_factory=list)

    @classmethod
    def from_songs(cls, songs: List[Dict]) -> "CatalogVocab":
        tags = sorted({tag for song in songs for tag in song.get("mood_tags", [])})
        return cls(
            genres=sorted({song["genre"].lower() for song in songs}),
            moods=sorted({song["mood"].lower() for song in songs}),
            mood_tags=tags,
            decades=sorted({int(song["release_decade"]) for song in songs}),
            languages=sorted({song["language"].lower() for song in songs}),
            titles=[song["title"] for song in songs],
        )

    def match_genre(self, value: Optional[str]) -> Optional[str]:
        """Map a free-text genre onto a catalog genre, or None if it isn't in the catalog."""
        text = (value or "").strip().lower()
        if not text:
            return None
        if text in self.genres:
            return text
        mapped = GENRE_SYNONYMS.get(text)
        if mapped in self.genres:
            return mapped
        return None

    def match_mood(self, value: Optional[str]) -> Optional[str]:
        text = (value or "").strip().lower()
        if not text:
            return None
        if text in self.moods:
            return text
        mapped = MOOD_SYNONYMS.get(text)
        if mapped in self.moods:
            return mapped
        return None

    def as_prompt_text(self) -> str:
        """Compact description of the catalog for the language model."""
        return (
            f"genres: {', '.join(self.genres)}\n"
            f"moods: {', '.join(self.moods)}\n"
            f"mood_tags: {', '.join(self.mood_tags)}\n"
            f"release decades: {', '.join(str(d) for d in self.decades)}\n"
            f"languages: {', '.join(self.languages)}"
        )
