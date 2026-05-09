# Design & Tech Tradeoffs — Phases 1 & 2

This document captures the key design decisions made in Phase 1 (trajectory risk
accumulator) and Phase 2 (action gate), along with the alternatives that were
considered and why they were rejected.

For an interview, each section here is a talking point: *what problem were you
solving, what options existed, what did you choose, and what did you trade away?*

---

## §1 — Phase 1: Trajectory Risk Accumulator

### 1.1 Scoring model: weighted accumulation vs. per-turn threshold

**Problem**: A sequence of borderline turns should escalate even if no individual
turn exceeds a single-turn block threshold.

**Options considered**:

| Option | How it works | Pro | Con |
|--------|-------------|-----|-----|
| **Per-turn max** | Block if any turn exceeds threshold | Simple, predictable | Misses slow-escalation patterns entirely |
| **Simple average** | Mean of all turn risks | Easy to reason about | Recent malicious turns diluted by early benign turns |
| **Recency-weighted average** *(chosen)* | Exponential weight: later turns carry more weight | Catches late-escalation; early benign turns still count | Recency bias could raise false positive if a user asks something risky then backtracks |
| **LLM-as-trajectory-judge** | Feed full conversation to LLM, ask "is this escalating?" | Most accurate | ~500ms latency per turn, expensive, hard to explain decisions |

**Chosen**: Recency-weighted average with configurable `recency_weight_base`.
Weight of turn i = `base^i`. Default `base=1.3` means turn 5 has 3.7× the weight
of turn 1. This is tunable in `trajectory_thresholds.yaml` without code changes.

**What we traded away**: Interpretability. A weighted score is harder to explain
to a PM than "this specific turn was blocked." Mitigation: `triggered_rules` in
`TrajectoryResult` surfaces which rules fired.

---

### 1.2 Category coherence: multiplier vs. separate counter

**Problem**: A session where every message is about the same harmful category
(e.g. illicit_behavior repeated 4 times) is more suspicious than a session
with one illicit_behavior message and three benign ones.

**Options**:

| Option | Pro | Con |
|--------|-----|-----|
| **Separate counter threshold** | "3+ same-category turns → block" — very explicit | Binary, can't capture degrees of concern |
| **Multiplier on score** *(chosen)* | Scales continuously with coherence ratio | Adds complexity; multiplier value is non-obvious to tune |
| **No coherence signal** | Simpler | Misses a real escalation pattern |

**Chosen**: Multiplier that scales from 1.0× (at 60% category coherence) to
`category_coherence_multiplier` (1.5×, at 100% coherence). The 60% floor prevents
the multiplier from firing on sessions that are only slightly dominated by a category.

---

### 1.3 Session storage: in-process dict vs. Redis vs. DB

**Problem**: Sessions need to persist across multiple turns. In a production
multi-server deployment, a session started on server A might continue on server B.

| Option | Persistence | Scale | Latency | Complexity |
|--------|------------|-------|---------|------------|
| **In-process dict** *(implemented)* | None (lost on restart) | Single process | Microseconds | Zero |
| **Redis** | TTL-based | Horizontal | ~1ms | Medium (one dependency) |
| **Postgres / SQLite** | Full audit trail | Horizontal | ~5–10ms | High |

**Chosen for this project**: In-process dict. Zero dependencies, sufficient for
a test harness where each test run is a fresh process.

**Production recommendation**: Redis with TTL. Rationale: sessions rarely need to
be queried historically (that's what the results/run_XXX logs are for). Redis
TTL handles expiry automatically. The `SessionStore` API is identical — swap the
dict for a Redis client in `session_tracker.py` without changing any callers.

**What we traded away**: In-process means sessions don't survive a restart or
scale horizontally. Acceptable for a portfolio project; unacceptable for production.

---

### 1.4 Stateless scorer vs. stateful scorer

**Problem**: Should the `RiskAccumulator` maintain running state, or should it
re-score the full session each time?

**Chosen**: Stateless. The accumulator is a pure function over the session object.
The session owns all state.

**Why**: Stateless is easier to test, parallelize, and reason about. It also means
you can change the scoring model (e.g., update `recency_weight_base`) and re-score
historical sessions without losing data. The cost is that re-scoring a 50-turn
session is O(n) — but at < 1ms per turn, this is never the bottleneck.

**Alternative**: Maintain a running score that's updated incrementally. Faster
for very long sessions, but introduces state mutation bugs and makes replaying
historical sessions harder.

---

### 1.5 Benign cooling: should good behaviour "undo" suspicion?

**Problem**: A user asks something risky in turn 2, then the next 8 turns are
completely benign. Should we still block them?

**Two views**:

- *Conservative*: "Suspicious once, suspicious forever." A serious safety system
  should have long memory. An attacker could probe, get told to back off, wait,
  then probe again.
- *Liberal*: "Context matters." A nurse asking about medication overdoses will
  get a follow-up question like "I'm asking because of my job." That should
  reduce the score.

**Chosen**: Partial cooling. Benign turns reduce the score by `benign_cooling_factor^n`
(default 0.85 per benign turn), BUT cooling is only applied when `trajectory_score < warn_threshold`.
Once a session enters WARN, cooling is disabled — it would require explicit human review
or session close to clear.

**What we traded away**: Stricter systems would never cool. Looser systems would cool
more aggressively. This is a policy decision, not a technical one — document it in
`trajectory_thresholds.yaml` so policy owners can change it.

---

## §2 — Phase 2: Action Gate

### 2.1 Two-dimensional gating: why reversibility + harm, not just harm?

**The key insight**: For text safety, harm probability is the only variable that
matters. For action safety, you also need to know *can we undo this if we're wrong?*

A file write with 30% harm probability → REVIEW. You can delete the file.
A `send_email` with 10% harm probability → REVIEW. You cannot unsend it.
A `transfer_funds` with 5% harm probability → borderline SHIP. Low harm, but irreversible.

This is directly from Anthropic's ["minimal footprint" principle](https://www.anthropic.com/research/agentic-safety):
"prefer reversible over irreversible actions."

**Why not just lower the harm threshold for irreversible actions?**
You could set `harm_block_threshold=0.30` for irreversible vs `0.60` for reversible.
Problem: this requires knowing a priori which tools are irreversible — and "just lower
the threshold" doesn't capture the key insight that an irreversible action in a
suspicious trajectory should be blocked even with low individual harm probability.

---

### 2.2 Reversibility catalogue: static YAML vs. dynamic inference

**Problem**: How do we know if a tool call is reversible?

| Option | Pro | Con |
|--------|-----|-----|
| **Static catalogue** *(chosen)* | Fast, transparent, no LLM call | Requires maintenance; new tools default to PARTIAL |
| **LLM inference** | Handles novel tools | Adds latency and cost to every action; LLM can be wrong |
| **Schema annotation** | Tool authors annotate their tools | Requires upstream coordination |

**Chosen**: Static catalogue in `action_schema.py`, with conservative default
(PARTIAL) for unknown tools. The catalogue is intentionally small — 20 tools —
to show the pattern without exhaustive coverage.

**What we traded away**: Unknown tools are treated as PARTIAL, not REVERSIBLE.
This is conservative: better to REVIEW a safe action than to SHIP a harmful one.

---

### 2.3 Pre-execution vs. post-execution gating

**Chosen**: Pre-execution only.

**Why**: You cannot unsend an email. Post-execution auditing is useful for
learning and detection, but it doesn't prevent harm. The action gate must run
*before* tool execution, not after.

**What this misses**: Post-execution anomaly detection (e.g., did the file
write contain sensitive data patterns?). This is a separate concern — add it
as an audit layer alongside the gate, not instead of it.

---

### 2.4 Fail-fast vs. evaluate-all in `evaluate_sequence`

**Problem**: When evaluating a sequence of tool calls, do you stop at the first BLOCK
or evaluate all of them?

**Chosen**: Fail-fast in `evaluate_sequence` (production mode). The sequence stops
at the first BLOCK.

**Why**: An agent that has been BLOCKED should not continue executing subsequent
actions. In a real pipeline, the agent process is suspended or terminated; there
is no "continue past a block."

**Test harness note**: The test dataset (`action_sequences.jsonl`) expects
fail-fast behaviour — sequences are designed to show where the first BLOCK fires.
For dataset analysis, you could disable fail-fast to see what *would* have been
blocked at each step.

---

### 2.5 Phase 1 → Phase 2 integration: why trajectory context matters for actions

**The critical insight** (from the interview prep chat):
> "An agent assembling a harmful capability across 10 tool calls looks safe on each
> individual call. You need trajectory-aware evaluation, not just per-action gating."

The `IRREVERSIBLE_IN_SUSPICIOUS_TRAJECTORY` rule in `action_gate.py` is the
infrastructure implementation of this insight. When:

1. The conversation trajectory score is ≥ `trajectory_block_threshold` (0.45), **and**
2. The agent attempts an irreversible action

→ BLOCK, regardless of the action's individual harm probability.

This catches the data exfiltration pattern in `ag_006`: each individual action
(db_read, write_file, http_post) looks borderline, but together in a suspicious
conversation they constitute a harmful sequence.

---

## §3 — What's not here (Phase 3+ ideas)

These are explicitly out of scope for the ~2-week build, but worth knowing for
an interview discussion of where the system could go:

| Feature | Why it matters | Complexity |
|---------|---------------|------------|
| **Minimum sample size / confidence intervals** | Release decisions on 5 samples are meaningless | Medium |
| **LLM judge for trajectory** | "Is this conversation escalating toward harm?" as an explicit judge call | Medium |
| **Human-in-the-loop queue** | REVIEW decisions need an actual review workflow | High |
| **Cross-session signals** | Same user/IP across multiple sessions = pattern | High |
| **Streaming / async gate** | Real-time gating in a streaming model response | High |
| **Differential privacy for audit logs** | Sessions contain PII; audit logs need privacy guarantees | High |
