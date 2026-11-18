"""
The recommendation agent.

    request ──► 1 GUARD ──► 2 PLAN ──► 3 VALIDATE ──► 4 SCORE & RANK ──► 5 SELF-CHECK ──► 6 EXPLAIN
                (input       (LLM or     (guardrails   (Module 3          (confidence,       (LLM or
                 checks)      offline)    + catalog)    score_song)        diversity, may     template,
                                                                           refine once)       checked)

Every step is written to a trace so you can see *why* the agent did what it did.
The language model only translates words into a profile and writes friendly explanations;
the actual ranking is always the transparent Module 3 scoring rule.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ai.catalog import CatalogVocab
from src.ai.guardrails import GuardrailError, check_explanations, check_request, sanitize_profile
from src.ai.llm import LLMClient, LLMError, get_client
from src.ai.logger import log_run
from src.ai.parser import parse_offline, parse_with_llm
from src.core.recommender import load_songs, recommend_songs

LOW_CONFIDENCE = 0.55  # average match below this triggers a warning


def max_possible_score(profile: Dict[str, Any]) -> float:
    """Best score any song could get for this profile (mirrors the weights in score_song)."""
    weights = {
        "favorite_genre": 2.0, "favorite_mood": 1.0, "target_energy": 1.5, "likes_acoustic": 0.5,
        "target_valence": 1.0, "favorite_decade": 0.75, "target_popularity": 0.5,
        "target_instrumentalness": 0.5, "preferred_language": 0.5,
    }
    total = sum(weight for key, weight in weights.items() if profile.get(key) is not None)
    tags = profile.get("desired_mood_tags") or []
    total += min(1.0, 0.5 * len(tags))
    return total


@dataclass
class Recommendation:
    song: Dict[str, Any]
    score: float
    confidence: float  # score / best possible score, 0-1
    reasons: List[str]
    explanation: str = ""


@dataclass
class AgentResult:
    request: str
    provider: str
    profile: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    recommendations: List[Recommendation] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    clarification: Optional[str] = None  # set when we need more info instead of recommending
    blocked: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "provider": self.provider,
            "profile": self.profile,
            "interpretation": self.interpretation,
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "warnings": self.warnings,
            "clarification": self.clarification,
            "blocked": self.blocked,
            "recommendations": [
                {
                    "id": rec.song["id"],
                    "title": rec.song["title"],
                    "artist": rec.song["artist"],
                    "genre": rec.song["genre"],
                    "score": round(rec.score, 3),
                    "confidence": round(rec.confidence, 3),
                    "reasons": rec.reasons,
                    "explanation": rec.explanation,
                }
                for rec in self.recommendations
            ],
            "trace": self.trace,
        }


EXPLAIN_SYSTEM = """You write short, friendly explanations for music recommendations.
Use ONLY the facts provided for each song (its attributes and the scoring reasons). Do not invent
facts about artists, lyrics, or songs that are not listed. One or two sentences per song.
Return JSON: {"summary": "<one sentence about the whole list>",
              "explanations": {"<exact song title>": "<explanation>", ...}}"""


class RecommendationAgent:
    def __init__(
        self,
        songs: Optional[List[Dict[str, Any]]] = None,
        client: Optional[LLMClient] = "auto",  # type: ignore[assignment]
        k: int = 5,
        log_path: Optional[str] = "logs/agent_runs.jsonl",
        diversify: bool = True,
    ):
        self.songs = songs if songs is not None else load_songs("data/songs.csv")
        self.vocab = CatalogVocab.from_songs(self.songs)
        self.client = get_client() if client == "auto" else client
        self.k = k
        self.log_path = log_path
        self.diversify = diversify

    @property
    def provider(self) -> str:
        return self.client.name if self.client else "offline"

    # ------------------------------------------------------------------ main entry point
    def run(self, request: str) -> AgentResult:
        result = AgentResult(request=str(request), provider=self.provider)
        try:
            self._run_steps(request, result)
        except GuardrailError as exc:
            result.blocked = True
            result.clarification = str(exc)
            self._step(result, "guard", "blocked", {"reason": str(exc)})
        if self.log_path:
            log_run(result.to_dict(), self.log_path)
        return result

    # ------------------------------------------------------------------ the steps
    def _run_steps(self, request: str, result: AgentResult) -> None:
        # 1. GUARD
        start = time.perf_counter()
        checked = check_request(request)
        result.request = checked.text
        result.warnings += checked.warnings
        self._step(result, "guard", "ok", {"chars": len(checked.text),
                                             "injection_suspected": checked.injection_suspected}, start)

        # 2. PLAN: words -> raw profile
        start = time.perf_counter()
        raw, source = self._plan(checked.text, result)
        result.interpretation = str(raw.get("interpretation") or "")
        self._step(result, "plan", source, {"raw_profile": raw}, start)

        # 3. VALIDATE against the catalog
        start = time.perf_counter()
        profile, notes = sanitize_profile(raw, self.vocab)
        for item in raw.get("unavailable") or []:
            notes.append(f"Not in this catalog: {item}.")
        result.profile = profile
        result.warnings += notes
        self._step(result, "validate", "ok", {"profile": profile, "notes": notes}, start)

        if raw.get("on_topic") is False or not profile:
            result.clarification = (
                "I couldn't find any music preferences in that. Try naming a genre, a mood, "
                "or an activity, e.g. 'upbeat pop for a run' or 'quiet piano for sleeping'."
            )
            self._step(result, "clarify", "no usable preferences", {})
            return

        # 4. SCORE & RANK with the Module 3 algorithm
        start = time.perf_counter()
        ranked = recommend_songs(profile, self.songs, k=len(self.songs))
        best = max_possible_score(profile) or 1.0
        candidates = [
            Recommendation(song, score, min(1.0, score / best), reasons.split("; ") if reasons else [])
            for song, score, reasons in ranked
        ]
        picks = candidates[: self.k]
        self._step(result, "score", "module3.recommend_songs",
                   {"max_possible": best, "top": [(r.song["title"], round(r.score, 2)) for r in picks]}, start)

        # 5. SELF-CHECK, refine at most once
        start = time.perf_counter()
        picks, checks = self._self_check(profile, picks, candidates, result)
        self._step(result, "self_check", "refined" if checks["refined"] else "ok", checks, start)
        result.recommendations = picks
        result.confidence = sum(r.confidence for r in picks) / len(picks) if picks else 0.0

        # 6. EXPLAIN
        start = time.perf_counter()
        source = self._explain(result)
        self._step(result, "explain", source, {"summary": result.summary}, start)

    def _plan(self, text: str, result: AgentResult):
        if self.client:
            for attempt in (1, 2):  # one retry, then fall back
                try:
                    return parse_with_llm(self.client, text, self.vocab), f"llm ({self.provider})"
                except LLMError as exc:
                    self._step(result, "plan", f"llm attempt {attempt} failed", {"error": str(exc)})
            result.warnings.append("The AI model was unavailable, so a simpler keyword parser was used.")
        return parse_offline(text, self.vocab), "offline keyword parser"

    def _self_check(self, profile, picks, candidates, result):
        checks: Dict[str, Any] = {"refined": False, "issues": []}

        # Variety: no artist twice in the list (Future Work #3 from the model card)
        artists = [r.song["artist"] for r in picks]
        if self.diversify and len(set(artists)) < len(artists):
            checks["issues"].append("same artist appears twice")
            seen, diverse = set(), []
            for rec in candidates:
                if rec.song["artist"] not in seen:
                    diverse.append(rec)
                    seen.add(rec.song["artist"])
                if len(diverse) == self.k:
                    break
            picks = diverse
            checks["refined"] = True

        # Honesty: did we actually deliver what was asked for?
        genre = profile.get("favorite_genre")
        if genre:
            matches = sum(1 for r in picks if r.song["genre"].lower() == genre)
            checks["genre_matches"] = matches
            if matches < min(2, len(picks)):
                result.warnings.append(
                    f"The catalog only has {matches} '{genre}' song(s) in this list; "
                    "the rest are the closest matches by feel."
                )
        mood = profile.get("favorite_mood")
        if mood and not any(r.song["mood"].lower() == mood for r in picks):
            result.warnings.append(f"No '{mood}' songs made the list; these are the nearest by energy and feel.")

        confidence = sum(r.confidence for r in picks) / len(picks) if picks else 0.0
        checks["avg_confidence"] = round(confidence, 3)
        top = picks[0].confidence if picks else 0.0
        if top < LOW_CONFIDENCE:
            checks["issues"].append("low confidence")
            result.warnings.append(
                f"Low confidence (best match {top:.0%}): nothing in this {len(self.songs)}-song "
                "catalog fits closely, so treat these as loose suggestions."
            )
        elif confidence < LOW_CONFIDENCE - 0.1 and checks.get("genre_matches", 2) >= 2:
            checks["issues"].append("thin catalog")
            result.warnings.append(
                f"Only the first pick is a strong match ({top:.0%}); the catalog has few songs "
                "like it, so the rest are fallbacks."
            )
        return picks, checks

    def _explain(self, result: AgentResult) -> str:
        titles = [r.song["title"] for r in result.recommendations]
        explanations: Dict[str, str] = {}
        source = "template"
        if self.client:
            facts = "\n".join(
                f"- {r.song['title']} by {r.song['artist']} ({r.song['genre']}, {r.song['mood']}, "
                f"energy {r.song['energy']}, tags {', '.join(r.song.get('mood_tags', []))}): "
                f"reasons = {'; '.join(r.reasons)}"
                for r in result.recommendations
            )
            prompt = f"Listener asked: <<<{result.request}>>>\n\nRecommended songs:\n{facts}"
            try:
                data = self.client.complete_json(EXPLAIN_SYSTEM, prompt)
                explanations = check_explanations(data.get("explanations"), titles)
                if isinstance(data.get("summary"), str):
                    result.summary = " ".join(data["summary"].split())[:300]
                source = f"llm ({self.provider})"
                if len(explanations) < len(titles):
                    source += f", {len(titles) - len(explanations)} filled from template"
            except LLMError as exc:
                result.trace.append({"step": "explain", "status": "llm failed", "detail": {"error": str(exc)}})

        for rec in result.recommendations:
            rec.explanation = explanations.get(rec.song["title"]) or self._template(rec)
        if not result.summary:
            top = result.recommendations[0].song if result.recommendations else None
            result.summary = (
                f"Top pick: {top['title']} by {top['artist']}. Overall match {result.confidence:.0%}."
                if top else "No recommendations."
            )
        return source

    @staticmethod
    def _template(rec: Recommendation) -> str:
        song = rec.song

        def points(reason: str) -> float:
            try:
                return float(reason.rsplit("(+", 1)[1].rstrip(")"))
            except (IndexError, ValueError):
                return 0.0

        ranked = sorted(rec.reasons, key=points, reverse=True)
        strongest = [r for r in ranked if points(r) >= 0.25][:3] or ranked[:1]
        why = ", ".join(r.rsplit(" (+", 1)[0] for r in strongest) or "closest overall fit"
        article = "An" if song["mood"][:1].lower() in "aeiou" else "A"
        return f"{article} {song['mood']} {song['genre']} track. Strongest matches: {why}."

    @staticmethod
    def _step(result: AgentResult, name: str, status: str, detail: Dict[str, Any], start: float = None):
        entry = {"step": name, "status": status, "detail": detail}
        if start is not None:
            entry["ms"] = round((time.perf_counter() - start) * 1000, 1)
        result.trace.append(entry)
