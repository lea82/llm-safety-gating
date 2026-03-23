#!/bin/bash
# ============================================================
# Push llm-safety-gating to GitHub
# Run these commands from the project root directory.
# ============================================================

# ── Step 1: Create the repo on GitHub first ──────────────────
# Go to https://github.com/new
# Name: llm-safety-gating
# Description: Safety evaluation and release gating system for LLMs.
#              Translates policy into measurable thresholds and SHIP/BLOCK decisions.
# Visibility: Public
# Do NOT initialize with README (we have our own)
# Click "Create repository"

# ── Step 2: Initialize and push ──────────────────────────────

cd /path/to/llm-safety-gating-v2    # replace with your actual path

git init
git add .
git commit -m "Initial release: LLM safety evaluation and release gating system

- 62-prompt JSONL dataset across 5 policy categories
- LLM judge evaluator: policy_compliance / refusal_quality / harmfulness
- Heuristic fallback evaluator (no API calls required)
- Release gating: SHIP / CONDITIONAL_SHIP / BLOCK with threshold config
- Markdown + JSON release reports designed for PM review
- Provider abstraction: OpenAI (default), Anthropic (plug-in ready)
- 157 tests, all passing without API keys
- Sample reports for all three decision states committed to results/
"

git branch -M main
git remote add origin https://github.com/lea82/llm-safety-gating.git
git push -u origin main

# ── Step 3: Add repo metadata on GitHub ──────────────────────
# After pushing, go to your repo → Settings (gear icon next to About)
# Add:
#   Description: Safety evaluation and release gating for LLMs —
#                policy-aligned thresholds, LLM judge scoring, SHIP/BLOCK decisions
#   Website: https://lea82.github.io/leayanhuili.github.io/projects.html
#   Topics: llm, safety, evaluation, policy, release-gating, openai, anthropic

# ── Step 4: Update your website ──────────────────────────────
# In your leayanhuili.github.io repo:
# Open projects.html
# Paste the card from project_card.html after the LLM Evaluation Framework card
# Then push:

cd /path/to/leayanhuili.github.io    # replace with your actual path

git add projects.html
git commit -m "Add LLM Safety Gating project card"
git push
