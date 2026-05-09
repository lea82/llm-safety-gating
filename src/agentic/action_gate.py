"""
action_gate.py
--------------
Phase 2: Irreversibility-aware safety gate for agentic tool calls.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE INSIGHT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Single-turn safety: "is this text harmful?"
  Phase 1 trajectory: "is this *conversation* trending toward harm?"
  Phase 2 action:     "is this *action* safe to execute, given what
                       we know about the conversation so far?"

  Two new dimensions single-turn safety doesn't have:

    1. Harm probability  — same as before: how likely is this to cause harm?
    2. Reversibility     — can we undo this if we're wrong?

  Gate logic:
    REVERSIBLE  + low harm    → SHIP
    REVERSIBLE  + high harm   → REVIEW (human in the loop)
    IRREVERSIBLE + low harm   → REVIEW (can't undo, need confidence)
    IRREVERSIBLE + high harm  → BLOCK
    IRREVERSIBLE + suspicious trajectory → BLOCK (regardless of individual action)

  The last rule is the critical insight from the previous chat:
  "An agent assembling a harmful capability across 10 tool calls looks safe
  on each individual call."  Trajectory context is what catches it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DESIGN TRADEOFFS (§2 in TRADEOFFS.md)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Pre-execution vs post-execution gating:
    We gate BEFORE execution (pre-execution check).
    Post-execution auditing is a separate concern (audit log).
    Rationale: you can't unsend an email.

  LLM-as-judge vs rule-based harm probability:
    For this test harness: harm_probability comes from the JSONL dataset.
    In production: an LLM judge evaluates the action + context to score it.
    Tradeoff: LLM judge is accurate but adds latency (~500ms) and cost.
              Rule-based is fast but misses novel patterns.
    Recommendation: LLM judge for IRREVERSIBLE actions only.

  Synchronous vs async gating:
    Synchronous here for simplicity.  In a real pipeline, pre-execution
    gating for high-latency actions (e.g. LLM judge call) should be async
    with a timeout + conservative default (REVIEW on timeout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from .action_schema import ToolAction, ReversibilityClass, ActionClassifier
from ..trajectory.session_tracker import ConversationSession


@dataclass
class ActionDecision:
    """
    The gate's verdict on a single tool call.
    """
    action: str                  # "SHIP" | "REVIEW" | "BLOCK"
    reason: str
    tool_name: str
    reversibility: str
    harm_probability: float
    trajectory_score: float
    rules_fired: List[str] = field(default_factory=list)

    @property
    def should_block(self) -> bool:
        return self.action == "BLOCK"

    @property
    def needs_human(self) -> bool:
        return self.action == "REVIEW"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ActionGateConfig:
    """
    All thresholds for the action gate — load from agentic_thresholds.yaml
    in production.

    Design note: why are these separate from trajectory_thresholds.yaml?
    Action gate and trajectory gate have different semantics.
    Keeping them in separate configs makes the policy audit trail cleaner.
    """
    # Harm probability thresholds
    low_harm: float = 0.25           # below → low harm
    high_harm: float = 0.60          # above → high harm

    # Trajectory score at which an irreversible action is auto-blocked
    # even if its individual harm probability looks low.
    # Rationale: if the conversation is suspicious, we can't trust that
    # a "low harm" action is actually low harm in context.
    trajectory_block_threshold: float = 0.45

    # For IRREVERSIBLE actions, minimum confidence (1 - harm_probability)
    # required to SHIP without human review.
    irreversible_ship_confidence: float = 0.80  # i.e. harm_prob < 0.20


class ActionGate:
    """
    Pre-execution safety gate for agentic tool calls.

    Requires a ConversationSession from Phase 1 to evaluate the
    trajectory context of the action.

    Usage:
        gate = ActionGate()
        decision = gate.evaluate(tool_action, session)
        if decision.should_block:
            raise ActionBlocked(decision.reason)
        elif decision.needs_human:
            await human_review_queue.submit(tool_action, decision)
        else:
            result = await tool_executor.run(tool_action)
    """

    def __init__(self, config: Optional[ActionGateConfig] = None) -> None:
        self.config = config or ActionGateConfig()
        self.classifier = ActionClassifier()

    def evaluate(
        self,
        action: ToolAction,
        session: Optional[ConversationSession] = None,
    ) -> ActionDecision:
        """
        Evaluate a tool call against harm probability + reversibility + trajectory.
        """
        cfg = self.config

        # ── 1. Classify action if not already classified ──────────────────
        if action.category.value == "unknown":
            action = self.classifier.classify(action)

        trajectory_score = session.trajectory_score if session else 0.0
        rules_fired: List[str] = []

        # ── 2. Session already escalated → block everything ───────────────
        if session and session.escalation_triggered:
            rules_fired.append("SESSION_ESCALATED")
            return ActionDecision(
                action="BLOCK",
                reason=f"Session already escalated: {session.escalation_reason}",
                tool_name=action.tool_name,
                reversibility=action.reversibility.value,
                harm_probability=action.estimated_harm_probability,
                trajectory_score=trajectory_score,
                rules_fired=rules_fired,
            )

        # ── 3. High harm probability → always block ───────────────────────
        if action.estimated_harm_probability >= cfg.high_harm:
            rules_fired.append(f"HIGH_HARM_PROBABILITY({action.estimated_harm_probability:.2f})")
            return ActionDecision(
                action="BLOCK",
                reason=(
                    f"Harm probability {action.estimated_harm_probability:.2f} "
                    f"exceeds high_harm threshold {cfg.high_harm}"
                ),
                tool_name=action.tool_name,
                reversibility=action.reversibility.value,
                harm_probability=action.estimated_harm_probability,
                trajectory_score=trajectory_score,
                rules_fired=rules_fired,
            )

        # ── 4. Irreversible action + suspicious trajectory → block ─────────
        if (
            action.is_irreversible
            and trajectory_score >= cfg.trajectory_block_threshold
        ):
            rules_fired.append(
                f"IRREVERSIBLE_IN_SUSPICIOUS_TRAJECTORY"
                f"(traj={trajectory_score:.2f} ≥ {cfg.trajectory_block_threshold})"
            )
            # Also signal Phase 1 that this session is now escalated
            if session:
                session.mark_escalated(
                    f"Irreversible action '{action.tool_name}' attempted "
                    f"in suspicious trajectory (score={trajectory_score:.2f})"
                )
            return ActionDecision(
                action="BLOCK",
                reason=(
                    f"Irreversible action '{action.tool_name}' blocked: "
                    f"trajectory score {trajectory_score:.2f} ≥ "
                    f"threshold {cfg.trajectory_block_threshold}"
                ),
                tool_name=action.tool_name,
                reversibility=action.reversibility.value,
                harm_probability=action.estimated_harm_probability,
                trajectory_score=trajectory_score,
                rules_fired=rules_fired,
            )

        # ── 5. Irreversible action + insufficient confidence → review ──────
        if action.is_irreversible:
            required_confidence = cfg.irreversible_ship_confidence
            actual_confidence = 1.0 - action.estimated_harm_probability
            if actual_confidence < required_confidence:
                rules_fired.append(
                    f"IRREVERSIBLE_LOW_CONFIDENCE"
                    f"(conf={actual_confidence:.2f} < {required_confidence:.2f})"
                )
                return ActionDecision(
                    action="REVIEW",
                    reason=(
                        f"Irreversible action '{action.tool_name}' requires "
                        f"confidence ≥ {required_confidence:.0%}; "
                        f"actual = {actual_confidence:.0%}. Human review required."
                    ),
                    tool_name=action.tool_name,
                    reversibility=action.reversibility.value,
                    harm_probability=action.estimated_harm_probability,
                    trajectory_score=trajectory_score,
                    rules_fired=rules_fired,
                )

        # ── 6. Reversible + high harm (but below block threshold) → review ─
        if (
            action.reversibility == ReversibilityClass.REVERSIBLE
            and action.estimated_harm_probability >= cfg.low_harm
        ):
            rules_fired.append(
                f"REVERSIBLE_ELEVATED_HARM({action.estimated_harm_probability:.2f})"
            )
            return ActionDecision(
                action="REVIEW",
                reason=(
                    f"Reversible action '{action.tool_name}' has elevated harm probability "
                    f"{action.estimated_harm_probability:.2f}. Flagged for review."
                ),
                tool_name=action.tool_name,
                reversibility=action.reversibility.value,
                harm_probability=action.estimated_harm_probability,
                trajectory_score=trajectory_score,
                rules_fired=rules_fired,
            )

        # ── 7. Default: SHIP ───────────────────────────────────────────────
        return ActionDecision(
            action="SHIP",
            reason=(
                f"Action '{action.tool_name}' cleared: "
                f"reversibility={action.reversibility.value}, "
                f"harm={action.estimated_harm_probability:.2f}, "
                f"trajectory={trajectory_score:.2f}"
            ),
            tool_name=action.tool_name,
            reversibility=action.reversibility.value,
            harm_probability=action.estimated_harm_probability,
            trajectory_score=trajectory_score,
            rules_fired=[],
        )

    def evaluate_sequence(
        self,
        actions: List[ToolAction],
        session: Optional[ConversationSession] = None,
    ) -> List[ActionDecision]:
        """
        Evaluate a sequence of tool calls.
        Stops at first BLOCK (fail-fast).
        Updates trajectory context after each action.

        Design tradeoff: fail-fast vs evaluate-all
          fail-fast  → stops agent immediately on first block (safer)
          evaluate-all → gives full picture of what would have been blocked
          For production: fail-fast.  For test harness: evaluate-all is
          useful for dataset analysis.
        """
        decisions = []
        for action in actions:
            decision = self.evaluate(action, session)
            decisions.append(decision)
            if decision.should_block:
                break  # fail-fast
        return decisions
