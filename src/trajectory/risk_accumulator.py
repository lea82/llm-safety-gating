"""
risk_accumulator.py
-------------------
Trajectory risk scoring across a multi-turn conversation.

The core insight this module encodes:

  Single-turn classifiers see one message.
  A trajectory accumulator sees a *sequence* and can detect intent
  that only becomes visible across turns — e.g. three benign questions
  that together constitute a harmful research arc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SCORING MODEL — key design decisions (see TRADEOFFS.md §1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Non-linear accumulation
     Risk is not additive. Turn 4 asking about "precursor chemicals"
     after turns 1-3 about amateur chemistry should spike risk
     non-linearly. We implement a recency-weighted exponential
     accumulator with a configurable decay parameter.

  2. Category coherence bonus
     A session that stays in the same harmful category across turns
     gets a multiplier (default 1.5×).  This catches slow-escalation
     patterns where each individual turn looks borderline.

  3. Technique escalation penalty
     A session that uses different adversarial techniques across turns
     (fiction framing → academic framing → persona override) gets an
     additional penalty.  This catches adversarial probing.

  4. Cooling on benign turns
     If the user sends a clearly benign turn (risk < LOW_RISK_THRESHOLD),
     the trajectory score decays slightly.  This prevents a good-faith
     conversation from being permanently tainted by an early ambiguous
     message.

  5. Irreversibility override (Phase 2 hook)
     Any turn annotated with irreversible=True bypasses normal accumulation
     and triggers immediate escalation.  This is the Phase 2 integration
     point.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .session_tracker import ConversationSession, Turn


# ─────────────────────────────────────────────────────────────────────────────
#  Thresholds (mirror your policies/trajectory_thresholds.yaml values here
#  as Python constants for unit-test convenience)
# ─────────────────────────────────────────────────────────────────────────────

LOW_RISK_THRESHOLD = 0.15       # below this → benign, apply cooling
WARN_THRESHOLD = 0.45           # WARN band
ESCALATE_THRESHOLD = 0.70       # above this → trajectory block
CRITICAL_SINGLE_TURN = 0.85     # single turn this bad → immediate block


@dataclass
class AccumulatorConfig:
    """
    All tuning knobs in one place — load from trajectory_thresholds.yaml
    in production via load_trajectory_config().

    Design tradeoff: why YAML and not just constructor kwargs?
    YAML keeps thresholds in the policy layer, auditable by non-engineers.
    """
    recency_weight_base: float = 1.3       # Each turn is weighted base^(turn_index)
    category_coherence_multiplier: float = 1.5   # Same category repeated
    technique_variety_penalty: float = 0.15      # New adversarial technique each turn
    benign_cooling_factor: float = 0.85          # Score decay on benign turns
    warn_threshold: float = WARN_THRESHOLD
    escalate_threshold: float = ESCALATE_THRESHOLD
    critical_single_turn_threshold: float = CRITICAL_SINGLE_TURN
    low_risk_threshold: float = LOW_RISK_THRESHOLD
    max_score: float = 1.0                 # score is always clamped to [0,1]


@dataclass
class TrajectoryResult:
    """
    Output of accumulate() — attached back to the ConversationSession.
    """
    session_id: str
    trajectory_score: float
    status: str                   # "SAFE" | "WARN" | "ESCALATE"
    turn_count: int
    triggered_rules: List[str] = field(default_factory=list)
    per_turn_scores: List[float] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RiskAccumulator:
    """
    Stateless scorer — takes a ConversationSession, returns a TrajectoryResult.

    Stateless is intentional: the session owns state; the accumulator is
    a pure function (almost — it mutates session.trajectory_score as a side
    effect for convenience).

    Usage:
        accumulator = RiskAccumulator()
        result = accumulator.accumulate(session)
        if result.status == "ESCALATE":
            session.mark_escalated(result.explanation)
    """

    def __init__(self, config: Optional[AccumulatorConfig] = None) -> None:
        self.config = config or AccumulatorConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def accumulate(self, session: ConversationSession) -> TrajectoryResult:
        """
        Score the full trajectory of a session.
        Returns a TrajectoryResult and mutates session.trajectory_score.
        """
        cfg = self.config
        user_turns = session.user_turns

        if not user_turns:
            return TrajectoryResult(
                session_id=session.session_id,
                trajectory_score=0.0,
                status="SAFE",
                turn_count=0,
                explanation="No user turns yet.",
            )

        triggered_rules: List[str] = []
        per_turn_scores: List[float] = []

        # ── 1. Check for immediate single-turn critical hit ───────────────
        for turn in user_turns:
            if turn.single_turn_risk >= cfg.critical_single_turn_threshold:
                score = cfg.max_score
                session.trajectory_score = score
                session.mark_escalated(
                    f"Single turn risk {turn.single_turn_risk:.2f} at turn {turn.turn_index} "
                    f"exceeds critical threshold {cfg.critical_single_turn_threshold}",
                    at_turn=turn.turn_index,
                )
                return TrajectoryResult(
                    session_id=session.session_id,
                    trajectory_score=score,
                    status="ESCALATE",
                    turn_count=session.turn_count,
                    triggered_rules=["CRITICAL_SINGLE_TURN"],
                    per_turn_scores=[t.single_turn_risk for t in user_turns],
                    explanation=session.escalation_reason or "",
                )

        # ── 2. Recency-weighted accumulation ─────────────────────────────
        raw_score = self._recency_weighted_score(user_turns, cfg)
        per_turn_scores = [t.single_turn_risk for t in user_turns]

        # ── 3. Category coherence multiplier ─────────────────────────────
        coherence_mult = self._category_coherence_mult(user_turns, cfg)
        if coherence_mult > 1.0:
            triggered_rules.append(
                f"CATEGORY_COHERENCE(×{coherence_mult:.2f}) — "
                f"repeated harmful category: {self._dominant_category(user_turns)}"
            )
        raw_score = min(raw_score * coherence_mult, cfg.max_score)

        # ── 4. Technique variety penalty ─────────────────────────────────
        technique_count = len({t.technique for t in user_turns if t.technique})
        if technique_count >= 2:
            penalty = cfg.technique_variety_penalty * (technique_count - 1)
            raw_score = min(raw_score + penalty, cfg.max_score)
            triggered_rules.append(
                f"TECHNIQUE_VARIETY(+{penalty:.2f}) — "
                f"{technique_count} distinct adversarial techniques seen"
            )

        # ── 5. Benign cooling ─────────────────────────────────────────────
        benign_count = sum(
            1 for t in user_turns if t.single_turn_risk < cfg.low_risk_threshold
        )
        if benign_count > 0 and raw_score < cfg.warn_threshold:
            cooling = cfg.benign_cooling_factor ** benign_count
            raw_score *= cooling
            triggered_rules.append(
                f"BENIGN_COOLING(×{cooling:.2f}) — "
                f"{benign_count} low-risk turns reduced trajectory score"
            )

        # ── 6. Determine status ───────────────────────────────────────────
        final_score = round(min(raw_score, cfg.max_score), 4)
        status = self._status(final_score, cfg)

        # ── 7. Propagate to session ───────────────────────────────────────
        session.trajectory_score = final_score
        if status == "ESCALATE" and not session.escalation_triggered:
            explanation = (
                f"Trajectory score {final_score:.2f} ≥ escalation threshold "
                f"{cfg.escalate_threshold}. Rules fired: {'; '.join(triggered_rules) or 'weighted accumulation'}"
            )
            session.mark_escalated(explanation)
        else:
            explanation = self._build_explanation(final_score, status, triggered_rules, cfg)

        return TrajectoryResult(
            session_id=session.session_id,
            trajectory_score=final_score,
            status=status,
            turn_count=session.turn_count,
            triggered_rules=triggered_rules,
            per_turn_scores=per_turn_scores,
            explanation=explanation,
        )

    # ------------------------------------------------------------------ #
    #  Incremental API (for streaming / real-time use)                     #
    # ------------------------------------------------------------------ #

    def update(self, session: ConversationSession, new_turn: Turn) -> TrajectoryResult:
        """
        Re-score after a new turn is added.  More efficient than re-scoring
        the whole session from scratch in a streaming context.

        In practice, for sessions < 100 turns, accumulate() is fast enough
        that this optimisation is rarely needed — but it's the right seam
        to add if you move to a streaming pipeline.
        """
        # For now, full re-score.  Optimise later if profiling shows it matters.
        return self.accumulate(session)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _recency_weighted_score(
        self, turns: List[Turn], cfg: AccumulatorConfig
    ) -> float:
        """
        Weighted mean where later turns carry exponentially more weight.

        weight[i] = base ^ i   (i = 0-based index in user turn list)

        Rationale: in a multi-turn escalation, the model should be more
        alarmed by the *latest* message, not a message from 10 turns ago.
        But early context still matters — it's not discarded, just discounted.
        """
        if not turns:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for i, turn in enumerate(turns):
            w = cfg.recency_weight_base ** i
            weighted_sum += turn.single_turn_risk * w
            total_weight += w
        return weighted_sum / total_weight

    def _category_coherence_mult(
        self, turns: List[Turn], cfg: AccumulatorConfig
    ) -> float:
        """
        If the majority of flagged turns share a category, apply multiplier.
        Rationale: topic consistency across turns implies intentional probing,
        not accidental triggering.
        """
        categories = [t.category for t in turns if t.category and t.single_turn_risk > cfg.low_risk_threshold]
        if len(categories) < 2:
            return 1.0
        dominant = max(set(categories), key=categories.count)
        dominant_ratio = categories.count(dominant) / len(categories)
        if dominant_ratio >= 0.6:
            # Scale: ratio 0.6 → 1.0×, ratio 1.0 → full multiplier
            scale = (dominant_ratio - 0.6) / 0.4  # 0–1
            return 1.0 + scale * (cfg.category_coherence_multiplier - 1.0)
        return 1.0

    def _dominant_category(self, turns: List[Turn]) -> Optional[str]:
        categories = [t.category for t in turns if t.category]
        if not categories:
            return None
        return max(set(categories), key=categories.count)

    def _status(self, score: float, cfg: AccumulatorConfig) -> str:
        if score >= cfg.escalate_threshold:
            return "ESCALATE"
        if score >= cfg.warn_threshold:
            return "WARN"
        return "SAFE"

    def _build_explanation(
        self,
        score: float,
        status: str,
        triggered_rules: List[str],
        cfg: AccumulatorConfig,
    ) -> str:
        parts = [f"Trajectory score: {score:.2f} → {status}"]
        if triggered_rules:
            parts.append("Rules: " + "; ".join(triggered_rules))
        parts.append(
            f"Thresholds — WARN≥{cfg.warn_threshold}, ESCALATE≥{cfg.escalate_threshold}"
        )
        return " | ".join(parts)
