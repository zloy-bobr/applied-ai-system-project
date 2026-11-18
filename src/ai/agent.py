"""
The recommendation agent.

    request ─► 1 GUARD ─► 2 PLAN ─► 3 VALIDATE ─► 4 SCORE & RANK ─► 5 SELF-CHECK ─► 6 EXPLAIN
               (input     (LLM or    (guardrails   (Module 3         (confidence,     (LLM or
                checks)    offline)   + catalog)    score_song)       variety, may     template,
                                                                      refine once)     checked)

Every step is written to a trace so you can see *why* the agent did what it did.
The language model only translates words into a profile and writes explanations;
the ranking itself is always the transparent Module 3 scoring rule.
"""
import re
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from src.ai.catalog import CatalogVocab
from src.ai.guardrails import (
    INJECTION_WARNING, GuardrailError, check_explanations, check_request, sanitize_profile, to_bool,
)
from src.ai.llm import LLMClient, LLMError, get_client
from src.ai.logger import log_run
from src.ai.parser import parse_offline, parse_with_llm
from src.core.recommender import load_songs, recommend_songs

LOW_CONFIDENCE = 0.55  # best match below this triggers a warning
WEIGHTS = {  # best points per rule, mirroring score_song in src/core/recommender.py
    "favorite_genre": 2.0, "favorite_mood": 1.0, "target_energy": 1.5, "likes_acoustic": 0.5,
    "target_valence": 1.0, "favorite_decade": 0.75, "target_popularity": 0.5,
    "target_instrumentalness": 0.5, "preferred_language": 0.5,
}
CLARIFY = ("I couldn't find any music preferences in that. Try naming a genre, a mood, or an "
           "activity, e.g. 'upbeat pop for a run' or 'quiet piano for sleeping'.")
EXPLAIN_SYSTEM = """You write short, friendly explanations for music recommendations.
Use ONLY the facts provided for each song (its attributes and the scoring reasons). Do not invent
facts about artists, lyrics, or songs that are not listed. One or two sentences per song.
Return JSON: {"summary": "<one sentence about the whole list>",
              "explanations": {"<exact song title>": "<explanation>", ...}}"""


def max_possible_score(profile: Dict[str, Any]) -> float:
    """Best score any song could get for this profile."""
    total = sum(weight for key, weight in WEIGHTS.items() if profile.get(key) is not None)
    return total + min(1.0, 0.5 * len(profile.get("desired_mood_tags") or []))


def _points(reason: str) -> float:
    """'genre match (+2.0)' -> 2.0"""
    found = re.search(r"\(\+([\d.]+)\)$", reason)
    return float(found.group(1)) if found else 0.0


@dataclass
class Recommendation:
    song: Dict[str, Any]
    score: float
    confidence: float  # score / best possible score, 0-1
    reasons: List[str]
    explanation: str = ""

    def template_explanation(self) -> str:
        """Fallback explanation from the strongest scoring reasons (worth at least 0.25 points)."""
        ranked = sorted(self.reasons, key=_points, reverse=True)
        strongest = [r for r in ranked if _points(r) >= 0.25][:3] or ranked[:1]
        why = ", ".join(r.rsplit(" (+", 1)[0] for r in strongest) or "closest overall fit"
        mood, genre = self.song["mood"], self.song["genre"]
        return f"{'An' if mood[:1].lower() in 'aeiou' else 'A'} {mood} {genre} track. Strongest matches: {why}."


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

    def step(self, name: str, status: str, detail: Dict[str, Any], start: Optional[float] = None) -> None:
        entry = {"step": name, "status": status, "detail": detail}
        if start is not None:
            entry["ms"] = round((time.perf_counter() - start) * 1000, 1)
        self.trace.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["confidence"] = round(self.confidence, 3)
        data["recommendations"] = [
            {**{key: r.song[key] for key in ("id", "title", "artist", "genre")},
             "score": round(r.score, 3), "confidence": round(r.confidence, 3),
             "reasons": r.reasons, "explanation": r.explanation}
            for r in self.recommendations
        ]
        return data


class RecommendationAgent:
    def __init__(self, songs: Optional[List[Dict[str, Any]]] = None, client: Any = "auto", k: int = 5,
                 log_path: Optional[str] = "logs/agent_runs.jsonl", diversify: bool = True):
        self.songs = songs if songs is not None else load_songs("data/songs.csv")
        self.vocab = CatalogVocab.from_songs(self.songs)
        self.client: Optional[LLMClient] = get_client() if client == "auto" else client
        self.k, self.log_path, self.diversify = k, log_path, diversify

    @property
    def provider(self) -> str:
        return self.client.name if self.client else "offline"

    def run(self, request: str) -> AgentResult:
        result = AgentResult(request=str(request), provider=self.provider)
        try:
            self._run_steps(request, result)
        except GuardrailError as exc:
            result.blocked, result.clarification = True, str(exc)
            result.step("guard", "blocked", {"reason": str(exc)})
        if self.log_path:
            log_run(result.to_dict(), self.log_path)
        return result

    def _run_steps(self, request: str, result: AgentResult) -> None:
        # 1. GUARD
        start = time.perf_counter()
        result.request, injection = check_request(request)
        if injection:
            result.warnings.append(INJECTION_WARNING)
        result.step("guard", "ok", {"chars": len(result.request), "injection_suspected": injection}, start)

        # 2. PLAN: words -> raw profile
        start = time.perf_counter()
        raw, source = self._plan(result)
        result.interpretation = str(raw.get("interpretation") or "")
        result.step("plan", source, {"raw_profile": raw}, start)

        # 3. VALIDATE against the catalog
        start = time.perf_counter()
        result.profile, notes = sanitize_profile(raw, self.vocab)
        result.warnings += notes
        result.step("validate", "ok", {"profile": result.profile, "notes": notes}, start)
        if to_bool(raw.get("on_topic")) is False or not result.profile:
            result.clarification = CLARIFY
            result.step("clarify", "no usable preferences", {})
            return

        # 4. SCORE & RANK every song with the Module 3 algorithm
        start = time.perf_counter()
        best = max_possible_score(result.profile) or 1.0
        candidates = [
            Recommendation(song, score, min(1.0, score / best), reasons.split("; ") if reasons else [])
            for song, score, reasons in recommend_songs(result.profile, self.songs, k=len(self.songs))
        ]
        result.step("score", "module3.recommend_songs", {
            "max_possible": best, "top": [(r.song["title"], round(r.score, 2)) for r in candidates[: self.k]],
        }, start)

        # 5. SELF-CHECK, refining at most once
        start = time.perf_counter()
        checks = self._self_check(candidates, result)
        result.step("self_check", "refined" if checks["refined"] else "ok", checks, start)

        # 6. EXPLAIN
        start = time.perf_counter()
        source = self._explain(result)
        result.step("explain", source, {"summary": result.summary}, start)

    def _plan(self, result: AgentResult):
        if self.client:
            for attempt in (1, 2):  # one retry, then fall back
                try:
                    return parse_with_llm(self.client, result.request, self.vocab), f"llm ({self.provider})"
                except LLMError as exc:
                    result.step("plan", f"llm attempt {attempt} failed", {"error": str(exc)})
            result.warnings.append("The AI model was unavailable, so a simpler keyword parser was used.")
        return parse_offline(result.request, self.vocab), "offline keyword parser"

    def _self_check(self, candidates: List[Recommendation], result: AgentResult) -> Dict[str, Any]:
        picks = candidates[: self.k]
        checks: Dict[str, Any] = {"refined": False, "issues": []}

        # Variety: at most one song per artist (Future Work #3 in the model card)
        if self.diversify and len({r.song["artist"] for r in picks}) < len(picks):
            seen, picks = set(), []
            for rec in candidates:
                if rec.song["artist"] not in seen and len(picks) < self.k:
                    seen.add(rec.song["artist"])
                    picks.append(rec)
            checks.update(refined=True, issues=["same artist appears twice"])

        # Honesty: did we deliver what was asked for?
        genre, mood = result.profile.get("favorite_genre"), result.profile.get("favorite_mood")
        genre_warned = False
        if genre:
            matches = checks["genre_matches"] = sum(r.song["genre"].lower() == genre for r in picks)
            in_catalog = sum(s["genre"].lower() == genre for s in self.songs)
            if matches < min(2, len(picks)):
                genre_warned = True
                per_artist = ", and only one song per artist is shown" if in_catalog > matches else ""
                result.warnings.append(
                    f"Only {matches} of these picks {'is' if matches == 1 else 'are'} '{genre}' "
                    f"(the catalog has {in_catalog}{per_artist}); the rest are the closest matches by feel."
                )
        if mood and not any(r.song["mood"].lower() == mood for r in picks):
            result.warnings.append(f"No '{mood}' songs made the list; these are the nearest by energy and feel.")

        result.recommendations = picks
        result.confidence = sum(r.confidence for r in picks) / len(picks) if picks else 0.0
        checks["avg_confidence"] = round(result.confidence, 3)
        top = picks[0].confidence if picks else 0.0
        if top < LOW_CONFIDENCE:
            checks["issues"].append("low confidence")
            result.warnings.append(
                f"Low confidence (best match {top:.0%}): nothing in this {len(self.songs)}-song "
                "catalog fits closely, so treat these as loose suggestions."
            )
        elif result.confidence < LOW_CONFIDENCE - 0.1 and not genre_warned:
            checks["issues"].append("thin catalog")
            result.warnings.append(
                f"Only the first pick is a strong match ({top:.0%}); the catalog has few songs "
                "like it, so the rest are fallbacks."
            )
        return checks

    def _explain(self, result: AgentResult) -> str:
        recs = result.recommendations
        explanations: Dict[str, str] = {}
        source = "template"
        if self.client:
            facts = "\n".join(
                f"- {r.song['title']} by {r.song['artist']} ({r.song['genre']}, {r.song['mood']}, "
                f"energy {r.song['energy']}, tags {', '.join(r.song.get('mood_tags', []))}): "
                f"reasons = {'; '.join(r.reasons)}"
                for r in recs
            )
            try:
                data = self.client.complete_json(
                    EXPLAIN_SYSTEM, f"Listener asked: <<<{result.request}>>>\n\nRecommended songs:\n{facts}")
                explanations = check_explanations(data.get("explanations"), [r.song["title"] for r in recs])
                if isinstance(data.get("summary"), str):
                    result.summary = " ".join(data["summary"].split())[:300]
                missing = len(recs) - len(explanations)
                source = f"llm ({self.provider})" + (f", {missing} filled from template" if missing else "")
            except LLMError as exc:
                result.step("explain", "llm failed", {"error": str(exc)})

        for rec in recs:
            rec.explanation = explanations.get(rec.song["title"]) or rec.template_explanation()
        if not result.summary and recs:
            top = recs[0].song
            result.summary = f"Top pick: {top['title']} by {top['artist']}. Overall match {result.confidence:.0%}."
        return source
