"""
tests/agentic/test_action_gate.py
----------------------------------
Tests for Phase 2: action schema + action gate.
All tests run without API keys.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from src.agentic.action_schema import (
    ToolAction, ReversibilityClass, ActionCategory, ActionClassifier
)
from src.agentic.action_gate import ActionGate, ActionGateConfig
from src.trajectory.session_tracker import ConversationSession


# ─────────────────────────────────────────────────────────────────────────────
#  ActionClassifier Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestActionClassifier:

    def test_known_irreversible_tool(self):
        classifier = ActionClassifier()
        action = ToolAction(tool_name="send_email")
        classifier.classify(action)
        assert action.reversibility == ReversibilityClass.IRREVERSIBLE
        assert action.category == ActionCategory.EMAIL

    def test_known_reversible_tool(self):
        classifier = ActionClassifier()
        action = ToolAction(tool_name="read_file")
        classifier.classify(action)
        assert action.reversibility == ReversibilityClass.REVERSIBLE

    def test_unknown_tool_defaults_to_partial(self):
        classifier = ActionClassifier()
        action = ToolAction(tool_name="custom_enterprise_tool_v2")
        classifier.classify(action)
        assert action.reversibility == ReversibilityClass.PARTIAL
        assert action.category == ActionCategory.UNKNOWN

    def test_financial_tools_are_irreversible(self):
        classifier = ActionClassifier()
        action = ToolAction(tool_name="transfer_funds")
        classifier.classify(action)
        assert action.reversibility == ReversibilityClass.IRREVERSIBLE
        assert action.category == ActionCategory.FINANCIAL


# ─────────────────────────────────────────────────────────────────────────────
#  ActionGate Tests
# ─────────────────────────────────────────────────────────────────────────────

def make_action(tool_name, harm=0.05, reversibility=None):
    action = ToolAction(
        tool_name=tool_name,
        estimated_harm_probability=harm,
    )
    if reversibility:
        action.reversibility = reversibility
    else:
        ActionClassifier().classify(action)
    return action


def make_session(trajectory_score=0.0, escalated=False):
    session = ConversationSession()
    session.trajectory_score = trajectory_score
    if escalated:
        session.mark_escalated("test escalation")
    return session


class TestActionGate:

    def test_safe_reversible_ships(self):
        gate = ActionGate()
        action = make_action("read_file", harm=0.02)
        decision = gate.evaluate(action, make_session(0.05))
        assert decision.action == "SHIP"

    def test_high_harm_always_blocks(self):
        gate = ActionGate()
        action = make_action("delete_file", harm=0.75)
        decision = gate.evaluate(action, make_session(0.05))
        assert decision.action == "BLOCK"
        assert "HIGH_HARM_PROBABILITY" in " ".join(decision.rules_fired)

    def test_irreversible_low_harm_ships(self):
        """send_email with harm=0.05 → confidence 0.95 ≥ 0.80 → SHIP"""
        gate = ActionGate()
        action = make_action("send_email", harm=0.05)
        decision = gate.evaluate(action, make_session(0.10))
        assert decision.action == "SHIP"

    def test_irreversible_moderate_harm_reviews(self):
        """send_email with harm=0.35 → confidence 0.65 < 0.80 → REVIEW"""
        gate = ActionGate()
        action = make_action("send_email", harm=0.35)
        decision = gate.evaluate(action, make_session(0.10))
        assert decision.action == "REVIEW"
        assert "IRREVERSIBLE_LOW_CONFIDENCE" in " ".join(decision.rules_fired)

    def test_irreversible_in_suspicious_trajectory_blocks(self):
        """send_email in session with traj score 0.60 ≥ 0.45 → BLOCK"""
        gate = ActionGate()
        action = make_action("send_email", harm=0.10)  # individually looks fine
        decision = gate.evaluate(action, make_session(0.60))
        assert decision.action == "BLOCK"
        assert "IRREVERSIBLE_IN_SUSPICIOUS_TRAJECTORY" in " ".join(decision.rules_fired)

    def test_escalated_session_blocks_all(self):
        gate = ActionGate()
        action = make_action("read_file", harm=0.01)  # benign action
        session = make_session(escalated=True)
        decision = gate.evaluate(action, session)
        assert decision.action == "BLOCK"
        assert "SESSION_ESCALATED" in " ".join(decision.rules_fired)

    def test_reversible_elevated_harm_reviews(self):
        gate = ActionGate()
        action = make_action("write_file", harm=0.30)
        action.reversibility = ReversibilityClass.REVERSIBLE  # force reversible
        decision = gate.evaluate(action, make_session(0.05))
        assert decision.action == "REVIEW"

    def test_financial_transfer_edge_case(self):
        """transfer_funds harm=0.15 → confidence=0.85 ≥ 0.80 → SHIP"""
        gate = ActionGate()
        action = make_action("transfer_funds", harm=0.15)
        decision = gate.evaluate(action, make_session(0.05))
        assert decision.action == "SHIP", (
            "transfer_funds with harm=0.15 should just pass the confidence threshold"
        )

    def test_evaluate_sequence_fails_fast(self):
        gate = ActionGate()
        actions = [
            make_action("read_file", harm=0.02),
            make_action("send_email", harm=0.80),   # BLOCK
            make_action("write_file", harm=0.02),    # should not be reached
        ]
        session = make_session(0.05)
        decisions = gate.evaluate_sequence(actions, session)
        assert len(decisions) == 2  # stopped at BLOCK
        assert decisions[1].action == "BLOCK"

    def test_evaluate_sequence_all_ship(self):
        gate = ActionGate()
        actions = [make_action("read_file", harm=0.02) for _ in range(3)]
        decisions = gate.evaluate_sequence(actions, make_session(0.05))
        assert all(d.action == "SHIP" for d in decisions)

    def test_no_session_still_evaluates(self):
        """Gate should work without a session (no trajectory context)."""
        gate = ActionGate()
        action = make_action("send_email", harm=0.05)
        decision = gate.evaluate(action, session=None)
        assert decision.action == "SHIP"

    def test_decision_to_dict(self):
        gate = ActionGate()
        action = make_action("read_file", harm=0.01)
        decision = gate.evaluate(action, make_session(0.05))
        d = decision.to_dict()
        assert "action" in d
        assert "trajectory_score" in d


# ─────────────────────────────────────────────────────────────────────────────
#  Integration: Phase 1 → Phase 2
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase1Phase2Integration:
    """
    Simulate a full agentic interaction where trajectory state from Phase 1
    gates Phase 2 action execution.
    """

    def test_suspicious_conversation_blocks_irreversible_action(self):
        from src.trajectory.trajectory_gate import TrajectoryGate

        # Phase 1: build a suspicious session
        tgate = TrajectoryGate()
        session = tgate.new_session()
        tgate.process_turn(session, "msg1", single_turn_risk=0.40, category="illicit_behavior")
        tgate.process_turn(session, "msg2", single_turn_risk=0.45, category="illicit_behavior")
        tgate.process_turn(session, "msg3", single_turn_risk=0.50, category="illicit_behavior")

        # Phase 2: agent tries to send an email
        agate = ActionGate()
        action = make_action("send_email", harm=0.15)
        decision = agate.evaluate(action, session)

        # Should be BLOCK because trajectory_score is elevated
        assert decision.action in ("BLOCK", "REVIEW"), (
            f"Expected BLOCK or REVIEW for send_email in suspicious trajectory, "
            f"got {decision.action} (traj_score={session.trajectory_score:.2f})"
        )

    def test_clean_conversation_allows_irreversible_action(self):
        from src.trajectory.trajectory_gate import TrajectoryGate

        tgate = TrajectoryGate()
        session = tgate.new_session()
        for _ in range(3):
            tgate.process_turn(session, "help me with my report",
                               single_turn_risk=0.03, category="benign_control")

        agate = ActionGate()
        action = make_action("send_email", harm=0.05)
        decision = agate.evaluate(action, session)
        assert decision.action == "SHIP"
