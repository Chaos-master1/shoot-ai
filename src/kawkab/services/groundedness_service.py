"""LLM groundedness gate (transformation Phase D / D4).

The trust layer's second half: any LLM-surfaced claim must cite its
evidence. This service does two things:

1. **Evidence registry** — ``register_match_evidence`` distills a
   match's *stored* events and player attributions into a durable
   evidence record (kind ``event_summary``) whose payload is the
   complete set of checkable facts (counts, scorers, cards, shots).
   Nothing is inferred here: only quantities directly read from the
   event store.

2. **Claim gate** — ``gate_claims`` checks a review text sentence by
   sentence against the evidence corpus:

   - every *number* in a sentence must appear verbatim in the corpus,
     otherwise the sentence is flagged (numbers are the most
     confident-looking lies);
   - content words must overlap the corpus above a threshold, else the
     sentence is ``unsupported`` (a claim about nothing observable);
   - the overall verdict is ``grounded`` / ``partly_grounded`` /
     ``ungrounded`` — a review goes out with its grounding report
     attached, never silently.

Deliberately narrow: this is a lexical gate against fabrication, not
a semantic fact-checker. Its provenance block says exactly that.
"""

from __future__ import annotations

import re
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[\w\u0600-\u06FF']+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!؟?\n])\s+")

# Words too generic to count as evidence anchors (EN + short AR list;
# the gate is lexical, so function words must not dilute coverage).
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "for",
        "without",
        "within",
        "over",
        "under",
        "again",
        "very",
        "really",
        "just",
        "also",
        "too",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "so",
        "as",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "us",
        "who",
        "whom",
        "which",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "further",
        "once",
        "there",
        "here",
        "team",
        "match",
        "game",
        "player",
        "players",
        "minute",
        "minut",
        "minutes",
        "très",
        "bien",
        "très",
        "bien",
        "كان",
        "كانت",
        "يكون",
        "يكون",
        "هذا",
        "هذه",
        "ذلك",
        "التي",
        "الذي",
        "في",
        "من",
        "على",
        "إلى",
        "عن",
        "مع",
        "هل",
        "ما",
        "لا",
        "نحن",
        "هم",
        "هو",
        "هي",
    ]
)

_COVERAGE_FLOOR = 0.5  # fraction of content tokens that must anchor in evidence


class GroundednessService:
    """Registers durable evidence and gates LLM claims against it."""

    def __init__(self, storage_service: Any) -> None:
        self._storage = storage_service

    async def register_match_evidence(self, match_id: int) -> dict[str, Any]:
        """Distill stored match data into one durable evidence record.

        Returns the persisted record. Raises on storage failure (the
        registry must not silently shrink — every claim cites ids that
        must resolve).
        """
        events = await self._storage.get_match_events(match_id, limit=100000)
        players = await self._storage.get_match_players(match_id)

        track_to_name: dict[int, str] = {}
        for p in players:
            try:
                track_to_name[int(p["track_id"])] = p.get("name") or f"track_{p['track_id']}"
            except (TypeError, ValueError, KeyError):
                continue

        event_counts: dict[str, int] = {}
        scorers: list[str] = []
        cards: list[dict[str, str]] = []
        shots: dict[str, int] = {"home": 0, "away": 0}

        match = None
        try:
            matches = await self._storage.get_all_matches()
            match = next((m for m in matches if int(m["id"]) == int(match_id)), None)
        except Exception:
            match = None

        home = (match or {}).get("home_team") or "home"
        away = (match or {}).get("away_team") or "away"

        def _team_name(raw: Any) -> str:
            s = str(raw or "").lower()
            if s in ("home", "h", "1"):
                return home
            if s in ("away", "a", "2"):
                return away
            return str(raw or "unknown")

        for ev in events:
            etype = str(ev.get("event_type") or ev.get("type") or "").lower()
            event_counts[etype] = event_counts.get(etype, 0) + 1
            meta = ev.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    import json

                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            if etype in ("goal", "goal_conceded", "shot_goal", "score"):
                tid = _safe_int(ev.get("from_track_id"))
                scorers.append(track_to_name.get(tid, "unknown") if tid is not None else "unknown")
            elif etype in ("yellow_card", "red_card", "card"):
                card_tid = _safe_int(ev.get("from_track_id"))
                cards.append(
                    {
                        "type": etype,
                        "player": (
                            track_to_name.get(card_tid, "unknown")
                            if card_tid is not None
                            else "unknown"
                        ),
                    }
                )
            elif etype in ("shot", "shot_saved", "shot_off_target", "shot_blocked"):
                side = _team_name(ev.get("team"))
                key = "home" if side == home else ("away" if side == away else "unknown")
                if key in shots:
                    shots[key] += 1

        payload = {
            "match_id": match_id,
            "home_team": home,
            "away_team": away,
            "event_counts": event_counts,
            "total_events": len(events),
            "goals": scorers,
            "goal_count": len(scorers),
            "cards": cards,
            "shots_by_team": shots,
            "source": "stored events and player attributions; no external or inferred data",
        }
        label = f"event summary {home} vs {away} ({len(events)} events)"

        record_id = await self._storage.save_evidence_record(
            kind="event_summary", payload=payload, match_id=match_id, label=label
        )
        return {
            "evidence_id": record_id,
            "kind": "event_summary",
            "label": label,
            "payload": payload,
        }

    async def gate_claims(self, review_text: str, evidence_ids: list[int]) -> dict[str, Any]:
        """Gate a review's claims against the cited evidence records.

        Sentence-level verdicts plus an overall grounding status; every
        number in every sentence is cross-checked verbatim against the
        evidence corpus.
        """
        if not str(review_text or "").strip():
            return {"error": "review_text must not be empty"}
        if not evidence_ids:
            return {
                "status": "ungrounded",
                "note": "no evidence records cited — an LLM claim without cited evidence does not ship",
                "sentences": [],
            }

        records = await self._storage.get_evidence_records(record_ids=evidence_ids)
        found_ids = {int(r["id"]) for r in records}
        missing = [i for i in evidence_ids if int(i) not in found_ids]

        corpus_parts: list[str] = []
        for r in records:
            payload = r.get("payload") or {}
            corpus_parts.append(str(r.get("label") or ""))
            corpus_parts.append(_flatten_json(payload))
        corpus = "\n".join(corpus_parts).lower()
        corpus_numbers = set(_NUMBER_RE.findall(corpus))
        corpus_tokens = {t for t in _WORD_RE.findall(corpus.lower()) if t not in _STOPWORDS}

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(review_text) if s.strip()]
        results: list[dict[str, Any]] = []
        for sent in sentences:
            tokens = [t.lower() for t in _WORD_RE.findall(sent)]
            content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
            numbers = _NUMBER_RE.findall(sent)

            missing_numbers = sorted({n for n in numbers if n not in corpus_numbers})
            if content:
                anchored = sum(1 for t in set(content) if t in corpus_tokens)
                coverage = anchored / len(set(content))
            else:
                coverage = 0.0

            number_ok = not missing_numbers
            if not content:
                verdict = "unsupported"  # a claim about nothing observable
            elif number_ok and coverage >= _COVERAGE_FLOOR:
                verdict = "grounded"
            elif coverage > 0 and number_ok:
                verdict = "partly_grounded"
            else:
                verdict = "unsupported"

            results.append(
                {
                    "text": sent,
                    "verdict": verdict,
                    "coverage": round(coverage, 3),
                    "unverified_numbers": missing_numbers,
                    "unanchored_terms": sorted(set(content) - corpus_tokens)[:8],
                }
            )

        verdicts = [r["verdict"] for r in results]
        if any(v == "unsupported" for v in verdicts):
            status = "ungrounded"
        elif any(v == "partly_grounded" for v in verdicts):
            status = "partly_grounded"
        else:
            status = "grounded"

        return {
            "status": status,
            "sentences": results,
            "evidence_used": [
                {"id": int(r["id"]), "kind": r.get("kind"), "label": r.get("label")}
                for r in records
            ],
            "evidence_ids_missing": missing,
            "gate_provenance": "lexical anchor + verbatim number check against registered evidence; "
            "not a semantic fact-checker",
        }

    async def generate_and_gate_report(
        self, llm_service: Any, match_id: int, language: str, summary: str
    ) -> dict[str, Any]:
        """Production path: generate an LLM report, register evidence, gate it.

        The report is saved exactly as before (the honest artifact), and
        the gate result ships *with* it — the UI renders both, so the
        grounding status is visible wherever the text is.
        """
        evidence = await self.register_match_evidence(match_id)
        report_text = await llm_service.generate_coach_report(summary, language)
        gate = await self.gate_claims(report_text, [evidence["evidence_id"]])
        return {
            "report_text": report_text,
            "grounding": gate,
            "evidence_id": evidence["evidence_id"],
        }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flatten_json(obj: Any) -> str:
    """Flatten a JSON-ish payload into comparable text (keys + values)."""
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(str(k))
            parts.append(_flatten_json(v))
        return " ".join(parts)
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten_json(v) for v in obj)
    return str(obj)
