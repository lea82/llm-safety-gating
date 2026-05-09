"""
session_tracker.py
------------------
Maintains per-session conversation state for trajectory-aware safety evaluation.

Design philosophy:
  A single-turn classifier sees one message.  A session tracker sees a
  *sequence* of messages and builds a picture of where the conversation
  is heading — before any individual turn trips a hard rule.

Key abstraction: ConversationSession
  - Stores ordered turns (role + content + per-turn risk)
  - Exposes the full turn list to the risk accumulator
  - Is intentionally dumb about scoring — that logic lives in RiskAccumulator

Storage options (design tradeoff documented in TRADEOFFS.md):
  - In-process dict  → fast, no deps, lost on restart (default here)
  - Redis            → survives restarts, horizontal scale, TTL built-in
  - SQLite/Postgres  → audit trail, queryable, heavier
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Turn:
    """One message in a conversation."""
    turn_index: int          # 0-based position in session
    role: str                # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)

    # Populated after single-turn evaluation runs
    single_turn_risk: float = 0.0        # 0.0–1.0, from existing pipeline
    category: Optional[str] = None       # e.g. "illicit_behavior"
    technique: Optional[str] = None      # e.g. "fiction_framing"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "single_turn_risk": self.single_turn_risk,
            "category": self.category,
            "technique": self.technique,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationSession:
    """
    A complete conversation thread with trajectory metadata.

    Design note — why not just a list of dicts?
    We want to enforce turn ordering, attach session-level signals
    (e.g. escalation_triggered, trajectory_score), and make it easy
    to freeze/serialize for audit logging.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    turns: List[Turn] = field(default_factory=list)

    # Trajectory-level signals (set by RiskAccumulator)
    trajectory_score: float = 0.0        # running accumulated score
    escalation_triggered: bool = False
    escalation_reason: Optional[str] = None
    blocked_at_turn: Optional[int] = None

    # Optional metadata for logging/auditing
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Mutation API                                                         #
    # ------------------------------------------------------------------ #

    def add_turn(
        self,
        role: str,
        content: str,
        single_turn_risk: float = 0.0,
        category: Optional[str] = None,
        technique: Optional[str] = None,
        **metadata,
    ) -> Turn:
        """Append a new turn and return it."""
        turn = Turn(
            turn_index=len(self.turns),
            role=role,
            content=content,
            single_turn_risk=single_turn_risk,
            category=category,
            technique=technique,
            metadata=metadata,
        )
        self.turns.append(turn)
        return turn

    def mark_escalated(self, reason: str, at_turn: Optional[int] = None) -> None:
        self.escalation_triggered = True
        self.escalation_reason = reason
        self.blocked_at_turn = at_turn if at_turn is not None else len(self.turns) - 1

    # ------------------------------------------------------------------ #
    #  Query helpers                                                        #
    # ------------------------------------------------------------------ #

    @property
    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def user_turn_count(self) -> int:
        return len(self.user_turns)

    def last_n_user_turns(self, n: int) -> List[Turn]:
        return self.user_turns[-n:]

    def categories_seen(self) -> List[str]:
        return list({t.category for t in self.turns if t.category})

    def has_category(self, category: str) -> bool:
        return category in self.categories_seen()

    def risk_history(self) -> List[float]:
        """Per-turn single_turn_risk for all user turns, in order."""
        return [t.single_turn_risk for t in self.user_turns]

    # ------------------------------------------------------------------ #
    #  Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turns": [t.to_dict() for t in self.turns],
            "trajectory_score": self.trajectory_score,
            "escalation_triggered": self.escalation_triggered,
            "escalation_reason": self.escalation_reason,
            "blocked_at_turn": self.blocked_at_turn,
            "user_id": self.user_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationSession":
        turns = [Turn.from_dict(t) for t in d.pop("turns", [])]
        session = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        session.turns = turns
        return session

    def __repr__(self) -> str:
        return (
            f"<ConversationSession id={self.session_id[:8]}… "
            f"turns={self.turn_count} "
            f"traj_score={self.trajectory_score:.2f} "
            f"escalated={self.escalation_triggered}>"
        )


class SessionStore:
    """
    In-process session registry.

    Production note: replace the internal dict with a Redis client or
    a SQLAlchemy session to get persistence + TTL.  The API is identical —
    callers only use get/put/delete.

    TTL behaviour: sessions idle beyond `ttl_seconds` are garbage-collected
    lazily on next `get` or `purge_expired`.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: Dict[str, ConversationSession] = {}
        self.ttl_seconds = ttl_seconds

    def create(self, user_id: Optional[str] = None) -> ConversationSession:
        """Create and register a new empty session."""
        session = ConversationSession(user_id=user_id)
        self._store[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[ConversationSession]:
        session = self._store.get(session_id)
        if session is None:
            return None
        if time.time() - session.created_at > self.ttl_seconds:
            del self._store[session_id]
            return None
        return session

    def put(self, session: ConversationSession) -> None:
        self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def purge_expired(self) -> int:
        """Remove stale sessions. Returns count removed."""
        now = time.time()
        stale = [
            sid for sid, s in self._store.items()
            if now - s.created_at > self.ttl_seconds
        ]
        for sid in stale:
            del self._store[sid]
        return len(stale)

    def __len__(self) -> int:
        return len(self._store)
