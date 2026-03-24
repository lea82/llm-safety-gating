"""
tests/test_policy_loader.py

Tests for config loading, validation, and cross-validation.
These run without any API keys — pure config logic.
"""

import pytest
from pathlib import Path
import yaml
import tempfile
import os

from src.gating.policy_loader import (
    load_safety_policy,
    load_release_thresholds,
    validate_policy,
    validate_thresholds,
    cross_validate,
    load_and_validate_configs,
    PolicyValidationError,
)

POLICIES_DIR = Path(__file__).parent.parent / "policies"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_policy_path():
    return POLICIES_DIR / "safety_policy.yaml"


@pytest.fixture
def valid_thresholds_path():
    return POLICIES_DIR / "release_thresholds.yaml"


def write_yaml(data: dict, suffix=".yaml") -> Path:
    """Write a dict to a temp YAML file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    )
    yaml.dump(data, f)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

class TestPolicyLoading:
    def test_loads_real_policy(self, valid_policy_path):
        policy = load_safety_policy(valid_policy_path)
        assert policy.version == "1.0"
        assert len(policy.categories) >= 5

    def test_policy_has_expected_categories(self, valid_policy_path):
        policy = load_safety_policy(valid_policy_path)
        expected = {"self_harm", "illicit_behavior", "prompt_injection",
                    "jailbreak_attempts", "benign_control"}
        assert expected.issubset(set(policy.categories.keys()))

    def test_loads_real_thresholds(self, valid_thresholds_path):
        thresholds = load_release_thresholds(valid_thresholds_path)
        assert thresholds.version == "1.0"
        assert len(thresholds.categories) >= 5

    def test_thresholds_have_escalation_conditions(self, valid_thresholds_path):
        thresholds = load_release_thresholds(valid_thresholds_path)
        assert len(thresholds.categories["self_harm"].escalation_conditions) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestPolicyValidation:
    def test_valid_policy_passes(self, valid_policy_path):
        policy = load_safety_policy(valid_policy_path)
        warnings = validate_policy(policy)
        # May have warnings but should not raise
        assert isinstance(warnings, list)

    def test_invalid_risk_level_raises(self):
        data = {
            "version": "1.0",
            "policy_name": "Test",
            "description": "Test policy",
            "categories": {
                "bad_cat": {
                    "description": "desc",
                    "risk_level": "SUPER_CRITICAL",  # invalid
                    "evaluation_goal": "goal",
                    "requires_judge": False,
                }
            }
        }
        path = write_yaml(data)
        try:
            policy = load_safety_policy(path)
            with pytest.raises(PolicyValidationError, match="invalid risk_level"):
                validate_policy(policy)
        finally:
            os.unlink(path)

    def test_missing_evaluation_goal_raises(self):
        data = {
            "version": "1.0",
            "policy_name": "Test",
            "description": "desc",
            "categories": {
                "some_cat": {
                    "description": "desc",
                    "risk_level": "high",
                    "evaluation_goal": "",  # empty
                    "requires_judge": False,
                }
            }
        }
        path = write_yaml(data)
        try:
            policy = load_safety_policy(path)
            with pytest.raises(PolicyValidationError, match="evaluation_goal"):
                validate_policy(policy)
        finally:
            os.unlink(path)

    def test_judge_required_without_rubric_raises(self):
        data = {
            "version": "1.0",
            "policy_name": "Test",
            "description": "desc",
            "categories": {
                "some_cat": {
                    "description": "desc",
                    "risk_level": "high",
                    "evaluation_goal": "some goal",
                    "requires_judge": True,
                    # no judge_rubric
                }
            }
        }
        path = write_yaml(data)
        try:
            policy = load_safety_policy(path)
            with pytest.raises(PolicyValidationError, match="judge_rubric"):
                validate_policy(policy)
        finally:
            os.unlink(path)


class TestThresholdValidation:
    def test_valid_thresholds_pass(self, valid_thresholds_path):
        thresholds = load_release_thresholds(valid_thresholds_path)
        warnings = validate_thresholds(thresholds)
        assert isinstance(warnings, list)

    def test_out_of_range_threshold_raises(self):
        data = {
            "version": "1.0",
            "policy_version_ref": "1.0",
            "decision_rules": {},
            "risk_scoring": {},
            "reporting": {},
            "categories": {
                "some_cat": {
                    "risk_level": "high",
                    "thresholds": {
                        "max_failure_rate": 1.5,  # out of range
                    },
                    "warning_band": {},
                    "escalation_conditions": [],
                }
            }
        }
        path = write_yaml(data)
        try:
            thresholds = load_release_thresholds(path)
            with pytest.raises(PolicyValidationError, match="out of range"):
                validate_thresholds(thresholds)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_real_configs_cross_validate(self, valid_policy_path, valid_thresholds_path):
        policy = load_safety_policy(valid_policy_path)
        thresholds = load_release_thresholds(valid_thresholds_path)
        warnings = cross_validate(policy, thresholds)
        assert isinstance(warnings, list)

    def test_version_mismatch_raises(self, valid_policy_path):
        policy = load_safety_policy(valid_policy_path)
        data = {
            "version": "9.9",
            "policy_version_ref": "9.9",  # mismatch
            "decision_rules": {},
            "risk_scoring": {},
            "reporting": {},
            "categories": {},
        }
        path = write_yaml(data)
        try:
            thresholds = load_release_thresholds(path)
            with pytest.raises(PolicyValidationError, match="version"):
                cross_validate(policy, thresholds)
        finally:
            os.unlink(path)

    def test_missing_threshold_for_policy_category_raises(self, valid_policy_path):
        policy = load_safety_policy(valid_policy_path)
        # Thresholds that only cover one category
        data = {
            "version": "1.0",
            "policy_version_ref": "1.0",
            "decision_rules": {},
            "risk_scoring": {},
            "reporting": {},
            "categories": {
                "self_harm": {
                    "risk_level": "critical",
                    "thresholds": {"max_failure_rate": 0.0},
                    "warning_band": {},
                    "escalation_conditions": [],
                }
                # missing all other categories
            }
        }
        path = write_yaml(data)
        try:
            thresholds = load_release_thresholds(path)
            with pytest.raises(PolicyValidationError, match="no thresholds defined"):
                cross_validate(policy, thresholds)
        finally:
            os.unlink(path)

    def test_risk_level_mismatch_raises(self, valid_policy_path, valid_thresholds_path):
        policy = load_safety_policy(valid_policy_path)
        thresholds = load_release_thresholds(valid_thresholds_path)
        # Mutate risk level on one category
        thresholds.categories["self_harm"].risk_level = "low"
        with pytest.raises(PolicyValidationError, match="Risk level mismatch"):
            cross_validate(policy, thresholds)
