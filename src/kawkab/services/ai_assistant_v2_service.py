"""AI Coach Assistant v2 — voice interface, tactical suggestions, automated reports, conversation history."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from kawkab.core.logging import get_logger
from kawkab.services.llm_service import LLMConfig, LLMService

logger = get_logger(__name__)


@dataclass
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class Conversation:
    id: str
    match_id: int | None
    title: str
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        self.messages.append(ConversationMessage(role=role, content=content))
        self.updated_at = datetime.now(UTC).isoformat()


TACTICAL_PROMPTS = {
    "formation": "Analyze the formation used by each team in this match. What are the strengths and weaknesses of each formation? Suggest tactical adjustments.",
    "pressing": "Evaluate the pressing structure of both teams. Where are the pressing triggers? Which team was more effective at winning the ball high up the pitch?",
    "transitions": "Analyze the transition moments in this match. Which team was more dangerous on the counter-attack? How did each team react to losing possession?",
    "set_pieces": "Evaluate the set piece performance of both teams. How effective were their corner routines? What about free kicks and throw-ins?",
    "defense": "Analyze the defensive organization of both teams. How did they shape up without the ball? Were there any recurring defensive vulnerabilities?",
    "attack": "Analyze the attacking patterns of both teams. What were their primary build-up methods? Which channels did they target?",
    "possession": "Break down the possession patterns. Which team controlled the tempo? How did each team progress the ball through the thirds?",
    "key_moments": "Identify the key moments that decided this match. What were the critical goals, chances, or defensive actions that shaped the outcome?",
    "substitutions": "Evaluate the impact of substitutions made by both teams. Did the changes improve or disrupt the team's performance?",
    "overall": "Provide a comprehensive tactical analysis of this match covering formations, pressing, transitions, key moments, and recommendations for improvement.",
}

SYSTEM_PROMPT_TACTICAL = """You are an elite football tactical analyst with UEFA Pro License credentials. You analyze match data with the precision of a top-level coach.

STRICT RULES:
1. Always structure your analysis: Summary → Tactical Shape → Key Patterns → Individual Performances → Recommendations
2. Cite specific data points when available (xG, possession %, pass completion, etc.)
3. Use proper football terminology in the user's selected language
4. Be honest about data limitations — if data is from a short clip, state this clearly
5. Never invent statistics or events not present in the data
6. Focus on actionable insights a coach can use in training
7. Consider the match context (scoreline, home/away, competition level)
8. Frame criticisms constructively — suggest solutions, not just problems
9. If the user asks in a specific language, respond in that same language
10. Keep responses concise but thorough — coaches are busy people"""


class AIAssistantV2Service:
    """Enhanced AI Coach Assistant with voice, tactical suggestions, reports, and history."""

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self.llm = llm_service or LLMService(LLMConfig(provider="ollama"))
        self.conversations: dict[str, Conversation] = {}
        self._conversations_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "ai_conversations.json"
        )
        self._load_conversations()

    def _conversations_path(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    def _load_conversations(self) -> None:
        path = self._conversations_path()
        filepath = os.path.join(path, "ai_conversations.json")
        try:
            if os.path.exists(filepath):
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                for conv_data in data:
                    conv = Conversation(
                        id=conv_data["id"],
                        match_id=conv_data.get("match_id"),
                        title=conv_data.get("title", "Chat"),
                        created_at=conv_data.get("created_at", ""),
                        updated_at=conv_data.get("updated_at", ""),
                    )
                    for msg in conv_data.get("messages", []):
                        conv.messages.append(
                            ConversationMessage(
                                role=msg["role"],
                                content=msg["content"],
                                timestamp=msg.get("timestamp", ""),
                            )
                        )
                    self.conversations[conv.id] = conv
        except Exception as e:
            logger.warning(f"Failed to load conversations: {e}")

    def _save_conversations(self) -> None:
        path = self._conversations_path()
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "ai_conversations.json")
        try:
            data = []
            for conv in self.conversations.values():
                data.append(
                    {
                        "id": conv.id,
                        "match_id": conv.match_id,
                        "title": conv.title,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                        "messages": [
                            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                            for m in conv.messages
                        ],
                    }
                )
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save conversations: {e}")

    def create_conversation(self, match_id: int | None, title: str = "New Chat") -> Conversation:
        import uuid

        conv_id = str(uuid.uuid4())[:8]
        conv = Conversation(id=conv_id, match_id=match_id, title=title)
        self.conversations[conv_id] = conv
        self._save_conversations()
        return conv

    def get_conversation(self, conv_id: str) -> Conversation | None:
        return self.conversations.get(conv_id)

    def list_conversations(self, match_id: int | None = None) -> list[dict]:
        results = []
        for conv in self.conversations.values():
            if match_id is not None and conv.match_id != match_id:
                continue
            results.append(
                {
                    "id": conv.id,
                    "match_id": conv.match_id,
                    "title": conv.title,
                    "message_count": len(conv.messages),
                    "updated_at": conv.updated_at,
                }
            )
        results.sort(key=lambda x: str(x["updated_at"]), reverse=True)  # type: ignore[arg-type,return-value]
        return results

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            self._save_conversations()
            return True
        return False

    async def ask(
        self,
        conv_id: str,
        question: str,
        match_context: str = "",
        language: str = "en",
    ) -> str:
        conv = self.conversations.get(conv_id)
        if not conv:
            conv = self.create_conversation(match_id=None, title=question[:50])

        conv.add_message("user", question)

        system = SYSTEM_PROMPT_TACTICAL
        if language == "ar":
            system = system.replace(
                "You are an elite football tactical analyst", "أنت محلل تكتيكي كرة قدم نخبة"
            )

        context_block = f"\n\nMatch Context:\n{match_context}\n\n" if match_context else "\n\n"

        history_block = ""
        if len(conv.messages) > 1:
            history_block = "\n\nConversation History:\n"
            for msg in conv.messages[-6:-1]:
                prefix = "Coach" if msg.role == "user" else "Analyst"
                history_block += f"{prefix}: {msg.content}\n"

        prompt = f"""Question: {question}{context_block}{history_block}

Provide a tactical analysis following the structure: Summary → Tactical Shape → Key Patterns → Individual Performances → Recommendations.
Be specific and actionable."""

        answer = await self.llm.generate(prompt, system)
        conv.add_message("assistant", answer)
        self._save_conversations()
        return answer

    async def generate_automated_report(
        self,
        match_id: int,
        match_data: dict,
        language: str = "en",
    ) -> str:
        match_name = match_data.get("name", f"Match #{match_id}")
        home = match_data.get("home_team", "Home")
        away = match_data.get("away_team", "Away")
        score = f"{match_data.get('home_score', '?')} - {match_data.get('away_score', '?')}"

        system = SYSTEM_PROMPT_TACTICAL
        if language == "ar":
            system = (
                "أنت محلل تكتيكي كرة قدم. مهمتك إنشاء تقرير شامل بعد المباراة "
                "للمدربين. كن دقيقاً وموضوعياً."
            )

        summary = json.dumps(match_data, indent=2, ensure_ascii=False)

        prompt = f"""Generate a comprehensive post-match analysis report for:

Match: {match_name}
Score: {score}
Home: {home} | Away: {away}

Match Data:
{summary}

Structure the report as follows:
1. Executive Summary (2-3 sentences)
2. Tactical Shape & Formation Analysis
3. Attacking Patterns & Chance Creation
4. Defensive Organization & Pressing
5. Transition Moments
6. Set Pieces
7. Key Individual Performances
8. Key Moments That Decided the Match
9. Areas for Improvement (per team)
10. Training Recommendations for Next Week

Use proper football terminology. Be honest about what the data shows and doesn't show.
Focus on actionable insights."""

        report = await self.llm.generate(prompt, system)

        conv = self.create_conversation(match_id=match_id, title=f"Report: {match_name}")
        conv.add_message("user", f"Generate match report for {match_name}")
        conv.add_message("assistant", report)
        self._save_conversations()
        return report

    async def get_tactical_suggestion(
        self,
        topic: str,
        match_context: str = "",
        language: str = "en",
    ) -> str:
        prompt_template = TACTICAL_PROMPTS.get(topic, TACTICAL_PROMPTS["overall"])
        full_prompt = f"{prompt_template}\n\n{match_context}" if match_context else prompt_template
        system = SYSTEM_PROMPT_TACTICAL
        return await self.llm.generate(full_prompt, system)
