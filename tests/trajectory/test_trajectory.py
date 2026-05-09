"""
tests/trajectory/test_trajectory.py
------------------------------------
Tests for Phase 1: session tracker + risk accumulator + trajectory gate.
All tests run without API keys.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from src.trajectory.session_tracker import ConversationSession, SessionStore, Turn
from src.trajectory.risk_accumulator import RiskAccumulator, AccumulatorConfig
from src.trajectory.trajectory_gate import TrajectoryGate


# ─────────────────────────────────────────────────────────────────────────────
#  Session Tracker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationSession:

    def test_add_turn_increments_index(self):
        session = ConversationSession()
        t1 = session.add_turn("user", "hello", single_turn_risk=0.1)
        t2 = session.add_turn("user", "world", single_turn_risk=0.2)
        assert t1.turn_index == 0
        assert t2.turn_index == 1

    def test_user_turns_filters_by_role(self):
        session = ConversationSession()
        session.add_turn("user", "msg1")
        session.add_turn("assistant", "reply")
        session.add_turn("user", "msg2")
        assert len(session.user_turns) == 2

    def test_risk_history_returns_user_turn_risks(self):
        session = ConversationSession()
        session.add_turn("user", "a", single_turn_risk=0.1)
        session.add_turn("assistant", "b", single_turn_risk=0.0)
        session.add_turn("user", "c", single_turn_risk=0.5)
        assert session.risk_history() == [0.1, 0.5]

    def test_mark_escalated(self):
        session = ConversationSession()
        session.add_turn("user", "bad msg", single_turn_risk=0.9)
        session.mark_escalated("test reason", at_turn=0)
        assert session.escalation_triggered is True
        assert session.escalation_reason == "test reason"
        assert session.blocked_at_turn == 0

    def test_categories_seen(self):
        session = ConversationSession()
        session.add_turn("user", "a", category="illicit_behavior")
        session.add_turn("user", "b", category="jailbreak_attempts")
        session.add_turn("user", "c", category="illicit_behavior")
        cats = session.categories_seen()
        assert "illicit_behavior" in cats
        assert "jailbreak_attempts" in cats

    def test_serialization_roundtrip(self):
        session = ConversationSession()
        session.add_turn("user", "hello", single_turn_risk=0.3, category="benign_control")
        d = session.to_dict()
        restored = ConversationSession.from_dict(d)
        assert restored.session_id == session.session_id
        assert len(restored.turns) == 1
        assert restored.turns[0].content == "hello"


class TestSessionStore:

    def test_create_and_get(self):
        store = SessionStore()
        session = store.create(user_id="u1")
        retrieved = store.get(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_nonexistent_returns_none(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = SessionStore()
        session = store.create()
        store.delete(session.session_id)
        assert store.get(session.session_id) is None

    def test_purge_expired(self):
        store = SessionStore(ttl_seconds=0)  # expires immediately
        store.create()
        store.create()
        import time; time.sleep(0.01)
        removed = store.purge_expired()
        assert removed == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Risk Accumulator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskAccumulator:

    def _session_with_risks(self, risks, category=None, technique=None):
        session = ConversationSession()
        for r in risks:
            session.add_turn("user", f"msg_{r}", single_turn_risk=r,
                             category=category, technique=technique)
        return session

    def test_empty_session_returns_safe(self):
        acc = RiskAccumulator()
        session = ConversationSession()
        result = acc.accumulate(session)
        assert result.status == "SAFE"
        assert result.trajectory_score == 0.0

    def test_low_risks_stay_safe(self):
        acc = RiskAccumulator()
        session = self._session_with_risks([0.05, 0.08, 0.10])
        result = acc.accumulate(session)
        assert result.status == "SAFE"

    def test_high_risks_escalate(self):
        acc = RiskAccumulator()
        session = self._session_with_risks([0.5, 0.6, 0.7, 0.75])
        result = acc.accumulate(session)
        assert result.status in ("WARN", "ESCALATE")

    def test_critical_single_turn_immediate_block(self):
        acc = RiskAccumulator()
        session = self._session_with_risks([0.1, 0.2, 0.90])
        result = acc.accumulate(session)
        assert result.status == "ESCALATE"
        assert "CRITICAL_SINGLE_TURN" in result.triggered_rules

    def test_category_coherence_fires_for_repeated_category(self):
        acc = RiskAccumulator()
        session = ConversationSession()
        for _ in range(4):
            session.add_turn("user", "msg", single_turn_risk=0.4,
                             category="illicit_behavior")
        result = acc.accumulate(session)
        assert any("CATEGORY_COHERENCE" in r for r in result.triggered_rules)

    def test_technique_variety_penalty_fires(self):
        acc = RiskAccumulator()
        session = ConversationSession()
        techniques = ["fiction_framing", "academic_framing", "persona_override"]
        for tech in techniques:
            session.add_turn("user", f"msg_{tech}", single_turn_risk=0.35,
                             category="illicit_behavior", technique=tech)
        result = acc.accumulate(session)
        assert any("TECHNIQUE_VARIETY" in r for r in result.triggered_rules)

    def test_benign_cooling_reduces_score(self):
        acc = RiskAccumulator()
        # One borderline turn followed by many benign turns
        session = ConversationSession()
        session.add_turn("user", "slightly risky", single_turn_risk=0.40)
        for _ in range(5):
            session.add_turn("user", "totally benign", single_turn_risk=0.05)
        result = acc.accumulate(session)
        assert result.status == "SAFE", (
            f"Expected SAFE after benign cooling, got {result.status} "
            f"score={result.trajectory_score:.3f}"
        )

    def test_recency_weight_amplifies_late_risk(self):
        """A late high-risk turn should score higher than an early one."""
        acc = RiskAccumulator()
        # Session A: risk spike early
        session_a = ConversationSession()
        session_a.add_turn("user", "risky", single_turn_risk=0.70)
        session_a.add_turn("user", "benign", single_turn_risk=0.05)
        session_a.add_turn("user", "benign", single_turn_risk=0.05)
        result_a = acc.accumulate(session_a)

        # Session B: risk spike late
        session_b = ConversationSession()
        session_b.add_turn("user", "benign", single_turn_risk=0.05)
        session_b.add_turn("user", "benign", single_turn_risk=0.05)
        session_b.add_turn("user", "risky", single_turn_risk=0.70)
        result_b = acc.accumulate(session_b)

        assert result_b.trajectory_score > result_a.trajectory_score, (
            "Late-turn risk should produce higher trajectory score (recency weighting)"
        )

    def test_score_clamped_to_max(self):
        acc = RiskAccumulator()
        session = self._session_with_risks([1.0, 1.0, 1.0])
        result = acc.accumulate(session)
        assert result.trajectory_score <= 1.0

    def test_session_mutation(self):
        acc = RiskAccumulator()
        session = ConversationSession()
        session.add_turn("user", "msg", single_turn_risk=0.8)
        acc.accumulate(session)
        assert session.trajectory_score > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory Gate Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryGate:

    def test_safe_conversation_allows_all_turns(self):
        gate = TrajectoryGate()
        session = gate.new_session()
        for _ in range(5):
            decision = gate.process_turn(session, "tell me about sourdough",
                                         single_turn_risk=0.02)
            assert decision.action == "ALLOW"

    def test_critical_turn_blocks_immediately(self):
        gate = TrajectoryGate()
        session = gate.new_session()
        decision = gate.process_turn(session, "extremely harmful content",
                                     single_turn_risk=0.90)
        assert decision.action == "BLOCK"

    def test_post_escalation_turns_are_blocked(self):
        gate = TrajectoryGate()
        session = gate.new_session()
        gate.process_turn(session, "harmful", single_turn_risk=0.90)
        # Next turn in same session should also be blocked
        decision = gate.process_turn(session, "benign followup",
                                     single_turn_risk=0.01)
        assert decision.action == "BLOCK"

    def test_warn_band_returns_warn(self):
        cfg = AccumulatorConfig(warn_threshold=0.30, escalate_threshold=0.70)
        gate = TrajectoryGate(config=cfg)
        session = gate.new_session()
        for _ in range(3):
            decision = gate.process_turn(session, "borderline msg",
                                         single_turn_risk=0.35,
                                         category="illicit_behavior")
        # Should be WARN or BLOCK (depends on coherence multiplier)
        assert decision.action in ("WARN", "BLOCK")

    def test_irreversible_flag_blocks_immediately(self):
        gate = TrajectoryGate()
        session = gate.new_session()
        decision = gate.process_turn(
            session, "send email to everyone",
            single_turn_risk=0.10,
            irreversible=True,
        )
        assert decision.action == "BLOCK"
        assert "irreversible" in decision.reason.lower()

    def test_session_retrieval(self):
        gate = TrajectoryGate()
        session = gate.new_session(user_id="test_user")
        gate.process_turn(session, "msg", single_turn_risk=0.1)
        retrieved = gate.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.user_id == "test_user"
        assert retrieved.user_turn_count == 1
