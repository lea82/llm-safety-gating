"""
action_schema.py
----------------
Structured representation of an agent tool call.

Why model actions as structured data?
  Text output safety is well-understood. Action safety is different —
  the harm isn't in the words, it's in the side effect.
  Sending an email is not dangerous text; it's a real-world action
  that cannot be taken back. Structured schemas let us reason about
  reversibility, blast radius, and authorization independently of content.

  This is the same insight behind Anthropic's "minimal footprint" principle:
  "request only necessary permissions, prefer reversible over irreversible actions."

Action taxonomy (see TRADEOFFS.md §2):
  reversible   → file write to scratch dir, read operation, in-memory state
  irreversible → send_email, post_to_api, delete_record, transfer_funds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReversibilityClass(str, Enum):
    """
    Three-level reversibility taxonomy.

    Design decision: why not binary reversible/irreversible?
    PARTIAL captures actions that are difficult but not impossible to undo
    (e.g. creating a calendar event — you can delete it, but attendees
    already received the invite).  This gives the gate a middle option
    besides ALLOW and BLOCK.
    """
    REVERSIBLE = "reversible"       # Can be undone with no external side effects
    PARTIAL = "partial"             # Can be undone but with some external side effects
    IRREVERSIBLE = "irreversible"   # Cannot be undone (sends, deletes, financial)


class ActionCategory(str, Enum):
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    EMAIL = "email"
    CALENDAR = "calendar"
    DATABASE = "database"
    CODE_EXECUTION = "code_execution"
    FINANCIAL = "financial"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ToolAction:
    """
    A structured representation of one agent tool call.

    In a real agentic system this comes from parsing the model's tool_use
    response block.  In this test harness it comes from the JSONL datasets
    in datasets/agentic/.
    """
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Classification (can be auto-inferred via ActionClassifier or set manually)
    reversibility: ReversibilityClass = ReversibilityClass.REVERSIBLE
    category: ActionCategory = ActionCategory.UNKNOWN

    # Risk signals
    estimated_harm_probability: float = 0.0    # 0.0–1.0, from LLM judge
    blast_radius: str = "local"                 # "local" | "user" | "org" | "external"

    # Metadata
    action_id: Optional[str] = None
    description: Optional[str] = None          # Human-readable summary

    @property
    def is_irreversible(self) -> bool:
        return self.reversibility == ReversibilityClass.IRREVERSIBLE

    @property
    def is_high_risk(self) -> bool:
        return self.estimated_harm_probability >= 0.5 or self.is_irreversible

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reversibility"] = self.reversibility.value
        d["category"] = self.category.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in reversibility catalogue
#  (extend this or load from a YAML config for production)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REVERSIBILITY_CATALOGUE: Dict[str, Dict] = {
    # FILE SYSTEM
    "read_file":             {"reversibility": ReversibilityClass.REVERSIBLE,   "category": ActionCategory.FILE_SYSTEM,    "blast_radius": "local"},
    "write_file":            {"reversibility": ReversibilityClass.REVERSIBLE,   "category": ActionCategory.FILE_SYSTEM,    "blast_radius": "local"},
    "delete_file":           {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.FILE_SYSTEM,    "blast_radius": "local"},
    "create_directory":      {"reversibility": ReversibilityClass.REVERSIBLE,   "category": ActionCategory.FILE_SYSTEM,    "blast_radius": "local"},

    # NETWORK / API
    "http_get":              {"reversibility": ReversibilityClass.REVERSIBLE,   "category": ActionCategory.NETWORK,        "blast_radius": "external"},
    "http_post":             {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.NETWORK,        "blast_radius": "external"},
    "api_call":              {"reversibility": ReversibilityClass.PARTIAL,      "category": ActionCategory.NETWORK,        "blast_radius": "external"},

    # EMAIL
    "send_email":            {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.EMAIL,          "blast_radius": "external"},
    "draft_email":           {"reversibility": ReversibilityClass.REVERSIBLE,   "category": ActionCategory.EMAIL,          "blast_radius": "local"},
    "delete_email":          {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.EMAIL,          "blast_radius": "user"},

    # CALENDAR
    "create_event":          {"reversibility": ReversibilityClass.PARTIAL,      "category": ActionCategory.CALENDAR,       "blast_radius": "user"},
    "delete_event":          {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.CALENDAR,       "blast_radius": "user"},
    "update_event":          {"reversibility": ReversibilityClass.PARTIAL,      "category": ActionCategory.CALENDAR,       "blast_radius": "user"},

    # DATABASE
    "db_read":               {"reversibility": ReversibilityClass.REVERSIBLE,   "category": ActionCategory.DATABASE,       "blast_radius": "local"},
    "db_write":              {"reversibility": ReversibilityClass.PARTIAL,      "category": ActionCategory.DATABASE,       "blast_radius": "org"},
    "db_delete":             {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.DATABASE,       "blast_radius": "org"},

    # CODE EXECUTION
    "run_python":            {"reversibility": ReversibilityClass.PARTIAL,      "category": ActionCategory.CODE_EXECUTION, "blast_radius": "local"},
    "run_shell":             {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.CODE_EXECUTION, "blast_radius": "local"},

    # FINANCIAL
    "transfer_funds":        {"reversibility": ReversibilityClass.IRREVERSIBLE, "category": ActionCategory.FINANCIAL,      "blast_radius": "external"},
    "create_invoice":        {"reversibility": ReversibilityClass.PARTIAL,      "category": ActionCategory.FINANCIAL,      "blast_radius": "external"},
}


class ActionClassifier:
    """
    Infer reversibility + category from tool_name.
    Falls back to UNKNOWN / PARTIAL if not in catalogue.
    """

    def classify(self, action: ToolAction) -> ToolAction:
        entry = TOOL_REVERSIBILITY_CATALOGUE.get(action.tool_name)
        if entry:
            action.reversibility = entry["reversibility"]
            action.category = entry["category"]
            action.blast_radius = entry.get("blast_radius", "unknown")
        else:
            # Conservative default: unknown tools are treated as partially reversible
            action.reversibility = ReversibilityClass.PARTIAL
            action.category = ActionCategory.UNKNOWN
        return action
