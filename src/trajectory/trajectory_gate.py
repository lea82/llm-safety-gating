"""
trajectory_gate.py
------------------
Integration layer: connects the existing single-turn pipeline
(completions → evaluate → gate) to the new trajectory layer.

This module is the only file the existing cli.py needs to import
to get trajectory-aware evaluation.  Everything else is internal
to src/trajectory/.

Usage in cli.py:
    from src.trajectory.trajectory_gate import TrajectoryGate

    gate = TrajectoryGate()
    session = gate.new_session(user_id="u123")

    for user_message in conversation:
        # 1. Run your existing single-turn evaluator
        single_turn_result = existing_evaluator.evaluate(user_message)

        # 2. Hand the scored turn to the trajectory gate
        decision = gate.process_turn(
            session=session,
            content=user_message,
            single_turn_risk=single_turn_result.risk_score,
            category=single_turn_result.category,
            technique=single_turn_result.technique,
        )

        if decision.action == "BLOCK":
            return block_response(decision.reason)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Design tradeoff: why a wrapper instead of modifying the existing gate?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Option A (chosen): Wrapper / decorator pattern
  → Adds trajectory layer on top of existing single-turn gate
  → Zero changes to existing code
  → Can be toggled off with one flag
  → Easier to A/B test trajectory vs non-trajectory

  Option B: Modify existing aggregator.py
  → Tighter coupling, harder to revert
  → Would require changes to existing tests
  → Better long-term if trajectory becomes the only mode

  We chose A for a ~1-week extension.  Option B is the right refactor
  if this ships to production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .session_tracker import ConversationSession, SessionStore
from .risk_accumulator import RiskAccumulator, AccumulatorConfig, TrajectoryResult


@dataclass
class TurnDecision:
    """
    The gate's per-turn answer: what should the system do right now?
    """
    action: str              # "ALLOW" | "WARN" | "BLOCK"
    reason: str
    trajectory_score: float
    trajectory_status: str
    single_turn_risk: float
    session_id: str
    turn_index: int

    @property
    def should_block(self) -> bool:
        return self.action == "BLOCK"

    @property
    def should_warn(self) -> bool:
        return self.action == "WARN"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class TrajectoryGate:
    """
    Main public interface for Phase 1.

    Wraps SessionStore + RiskAccumulator and exposes a simple
    process_turn() → TurnDecision API.
    """

    def __init__(
        self,
        config: Optional[AccumulatorConfig] = None,
        session_ttl: int = 3600,
    ) -> None:
        self.store = SessionStore(ttl_seconds=session_ttl)
        self.accumulator = RiskAccumulator(config=config)

    # ------------------------------------------------------------------ #
    #  Session management                                                   #
    # ------------------------------------------------------------------ #

    def new_session(self, user_id: Optional[str] = None) -> ConversationSession:
        return self.store.create(user_id=user_id)

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self.store.get(session_id)

    # ------------------------------------------------------------------ #
    #  Core: process one turn                                               #
    # ------------------------------------------------------------------ #

    def process_turn(
        self,
        session: ConversationSession,
        content: str,
        role: str = "user",
        single_turn_risk: float = 0.0,
        category: Optional[str] = None,
        technique: Optional[str] = None,
        irreversible: bool = False,   # Phase 2 hook — see agentic/action_gate.py
    ) -> TurnDecision:
        """
        1. Append the turn to the session.
        2. Run the accumulator over the full session.
        3. Return a TurnDecision.

        The caller is responsible for:
          - Computing single_turn_risk (from existing evaluator)
          - Deciding what to do with action="BLOCK" (surface to user, log, etc.)
        """

        # ── Guard: already-escalated session ─────────────────────────────
        if session.escalation_triggered:
            return TurnDecision(
                action="BLOCK",
                reason=f"Session already escalated: {session.escalation_reason}",
                trajectory_score=session.trajectory_score,
                trajectory_status="ESCALATE",
                single_turn_risk=single_turn_risk,
                session_id=session.session_id,
                turn_index=len(session.turns),
            )

        # ── Add turn ──────────────────────────────────────────────────────
        turn = session.add_turn(
            role=role,
            content=content,
            single_turn_risk=single_turn_risk,
            category=category,
            technique=technique,
            irreversible=irreversible,
        )

        # ── Phase 2 hook: irreversible action override ────────────────────
        if irreversible:
            session.mark_escalated(
                f"Irreversible action at turn {turn.turn_index} with "
                f"trajectory score {session.trajectory_score:.2f}",
                at_turn=turn.turn_index,
            )
            return TurnDecision(
                action="BLOCK",
                reason=session.escalation_reason,
                trajectory_score=session.trajectory_score,
                trajectory_status="ESCALATE",
                single_turn_risk=single_turn_risk,
                session_id=session.session_id,
                turn_index=turn.turn_index,
            )

        # ── Accumulate trajectory risk ────────────────────────────────────
        result: TrajectoryResult = self.accumulator.accumulate(session)
        self.store.put(session)

        # ── Map status → action ───────────────────────────────────────────
        if result.status == "ESCALATE":
            action = "BLOCK"
        elif result.status == "WARN":
            action = "WARN"
        else:
            action = "ALLOW"

        return TurnDecision(
            action=action,
            reason=result.explanation,
            trajectory_score=result.trajectory_score,
            trajectory_status=result.status,
            single_turn_risk=single_turn_risk,
            session_id=session.session_id,
            turn_index=turn.turn_index,
        )
