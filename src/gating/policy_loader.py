"""
src/gating/policy_loader.py

Loads, validates, and cross-references safety_policy.yaml and
release_thresholds.yaml. Surfaces mismatches early so config errors are
caught before any eval run begins.

Design intent: these configs are the source of truth for what the program
considers "safe enough to ship." Validation here is a program management
concern, not just a software concern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_RISK_LEVELS = {"critical", "high", "medium", "low"}
VALID_DECISIONS = {"ship", "conditional_ship", "block"}


# ---------------------------------------------------------------------------
# Data classes — typed representations of the YAML configs
# ---------------------------------------------------------------------------


@dataclass
class PolicyCategory:
    name: str
    description: str
    risk_level: str
    evaluation_goal: str
    requires_judge: bool
    judge_rubric: str | None
    example_failure_modes: list[str] = field(default_factory=list)


@dataclass
class SafetyPolicy:
    version: str
    policy_name: str
    description: str
    categories: dict[str, PolicyCategory]


@dataclass
class CategoryThreshold:
    name: str
    risk_level: str
    thresholds: dict[str, float]
    warning_band: dict[str, float]
    escalation_conditions: list[dict[str, str]]
    notes: str = ""


@dataclass
class ReleaseThresholds:
    version: str
    policy_version_ref: str
    decision_rules: dict[str, Any]
    categories: dict[str, CategoryThreshold]
    risk_scoring: dict[str, Any]
    reporting: dict[str, Any]


@dataclass
class LoadedConfig:
    policy: SafetyPolicy
    thresholds: ReleaseThresholds


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_safety_policy(path: Path) -> SafetyPolicy:
    raw = _load_yaml(path)
    categories = {}
    for name, cat in raw.get("categories", {}).items():
        categories[name] = PolicyCategory(
            name=name,
            description=cat["description"].strip(),
            risk_level=cat["risk_level"],
            evaluation_goal=cat["evaluation_goal"].strip(),
            requires_judge=cat.get("requires_judge", False),
            judge_rubric=cat.get("judge_rubric"),
            example_failure_modes=cat.get("example_failure_modes", []),
        )
    return SafetyPolicy(
        version=raw["version"],
        policy_name=raw["policy_name"],
        description=raw["description"].strip(),
        categories=categories,
    )


def load_release_thresholds(path: Path) -> ReleaseThresholds:
    raw = _load_yaml(path)
    categories = {}
    for name, cat in raw.get("categories", {}).items():
        categories[name] = CategoryThreshold(
            name=name,
            risk_level=cat["risk_level"],
            thresholds=cat.get("thresholds", {}),
            warning_band=cat.get("warning_band", {}),
            escalation_conditions=cat.get("escalation_conditions", []),
            notes=cat.get("notes", "").strip(),
        )
    return ReleaseThresholds(
        version=raw["version"],
        policy_version_ref=raw["policy_version_ref"],
        decision_rules=raw.get("decision_rules", {}),
        categories=categories,
        risk_scoring=raw.get("risk_scoring", {}),
        reporting=raw.get("reporting", {}),
    )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class PolicyValidationError(Exception):
    """Raised when a config file fails validation."""
    pass


def validate_policy(policy: SafetyPolicy) -> list[str]:
    """Returns a list of validation warnings (not errors)."""
    issues = []
    for name, cat in policy.categories.items():
        if cat.risk_level not in VALID_RISK_LEVELS:
            raise PolicyValidationError(
                f"Category '{name}' has invalid risk_level '{cat.risk_level}'. "
                f"Must be one of: {VALID_RISK_LEVELS}"
            )
        if not cat.description:
            issues.append(f"Category '{name}' has no description.")
        if not cat.evaluation_goal:
            raise PolicyValidationError(
                f"Category '{name}' is missing evaluation_goal — "
                "every category must have a measurable goal."
            )
        if cat.requires_judge and not cat.judge_rubric:
            raise PolicyValidationError(
                f"Category '{name}' requires_judge=true but has no judge_rubric."
            )
    return issues


def validate_thresholds(thresholds: ReleaseThresholds) -> list[str]:
    """Returns a list of validation warnings (not errors)."""
    issues = []
    for name, cat in thresholds.categories.items():
        if cat.risk_level not in VALID_RISK_LEVELS:
            raise PolicyValidationError(
                f"Threshold category '{name}' has invalid risk_level '{cat.risk_level}'."
            )
        if not cat.thresholds:
            raise PolicyValidationError(
                f"Threshold category '{name}' has no thresholds defined."
            )
        # Validate numeric threshold values are in [0, 1]
        for key, val in cat.thresholds.items():
            if not isinstance(val, (int, float)):
                raise PolicyValidationError(
                    f"Threshold '{name}.{key}' must be numeric, got {type(val).__name__}."
                )
            if not (0.0 <= float(val) <= 1.0):
                raise PolicyValidationError(
                    f"Threshold '{name}.{key}' = {val} is out of range [0.0, 1.0]."
                )
    return issues


def cross_validate(policy: SafetyPolicy, thresholds: ReleaseThresholds) -> list[str]:
    """
    Checks that policy and thresholds are internally consistent:
    - version refs match
    - same categories defined in both
    - risk levels agree
    """
    issues = []

    # Version cross-reference
    if thresholds.policy_version_ref != policy.version:
        raise PolicyValidationError(
            f"release_thresholds.yaml references policy version "
            f"'{thresholds.policy_version_ref}' but safety_policy.yaml "
            f"is version '{policy.version}'. These must match."
        )

    policy_cats = set(policy.categories.keys())
    threshold_cats = set(thresholds.categories.keys())

    # Categories in policy but not in thresholds
    missing_thresholds = policy_cats - threshold_cats
    if missing_thresholds:
        raise PolicyValidationError(
            f"Policy categories {missing_thresholds} have no thresholds defined. "
            "Every policy category must have corresponding thresholds."
        )

    # Categories in thresholds but not in policy
    orphan_thresholds = threshold_cats - policy_cats
    if orphan_thresholds:
        issues.append(
            f"Threshold categories {orphan_thresholds} have no matching policy entry. "
            "These thresholds will be evaluated but have no documented policy rationale."
        )

    # Risk level consistency
    for cat_name in policy_cats & threshold_cats:
        p_risk = policy.categories[cat_name].risk_level
        t_risk = thresholds.categories[cat_name].risk_level
        if p_risk != t_risk:
            raise PolicyValidationError(
                f"Risk level mismatch for '{cat_name}': "
                f"policy says '{p_risk}', thresholds say '{t_risk}'. "
                "These must agree."
            )

    return issues


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def load_and_validate_configs(
    policy_path: Path,
    thresholds_path: Path,
) -> LoadedConfig:
    """
    Load both config files, validate each independently, then cross-validate.
    Raises PolicyValidationError on any hard failure.
    Returns LoadedConfig on success; logs warnings for soft issues.
    """
    logger.info("Loading safety policy from %s", policy_path)
    policy = load_safety_policy(policy_path)

    logger.info("Loading release thresholds from %s", thresholds_path)
    thresholds = load_release_thresholds(thresholds_path)

    logger.info("Validating safety policy...")
    policy_warnings = validate_policy(policy)

    logger.info("Validating release thresholds...")
    threshold_warnings = validate_thresholds(thresholds)

    logger.info("Cross-validating policy and thresholds...")
    cross_warnings = cross_validate(policy, thresholds)

    all_warnings = policy_warnings + threshold_warnings + cross_warnings
    for w in all_warnings:
        logger.warning("Config warning: %s", w)

    if all_warnings:
        logger.warning("%d config warning(s) found. Review before production use.", len(all_warnings))
    else:
        logger.info("All config validations passed.")

    return LoadedConfig(policy=policy, thresholds=thresholds)
