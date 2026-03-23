# Quickstart

Get from zero to a release report in under 5 minutes.

---

## 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/llm-safety-gating.git
cd llm-safety-gating
pip install -r requirements.txt
```

## 2. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

Or copy `.env.example` → `.env` and fill it in, then `source .env`.

## 3. Validate the config

```bash
python cli.py validate-config
```

Expected output: `✅ Config valid. Policy v1.0 | Thresholds v1.0`

## 4. Run the full pipeline

```bash
python cli.py run-all \
  --provider openai \
  --model gpt-4o \
  --model-id gpt-4o-2024-11-20 \
  --mode heuristic \
  --run-id my_first_run
```

This runs: **completions → heuristic evaluation → release report**

No extra API calls for the evaluator in heuristic mode — good for a fast first run.

Output:
```
results/my_first_run/
  completions.jsonl      # raw model responses
  evaluations.jsonl      # scored records
  release_decision.json  # machine-readable gate result
  release_report.md      # human-readable report
```

## 5. Upgrade to LLM judge evaluation

```bash
python cli.py run-all \
  --provider openai \
  --model gpt-4o \
  --model-id gpt-4o-2024-11-20 \
  --mode llm_judge \
  --run-id judge_run_001
```

The LLM judge scores each response on `policy_compliance`, `refusal_quality`,
and `harmfulness` using a structured rubric and your safety policy context.

---

## Step-by-step mode

```bash
# 1. Run model completions
python cli.py completions \
  --provider openai \
  --model gpt-4o \
  --run-id run_001

# 2. Evaluate responses (heuristic = no extra API calls)
python cli.py evaluate \
  --run-id run_001 \
  --mode heuristic

# 3. Generate report
python cli.py report \
  --run-id run_001 \
  --model-id gpt-4o-2024-11-20
```

---

## Single category debug run

```bash
python cli.py completions \
  --provider openai \
  --model gpt-4o \
  --category self_harm \
  --max-samples 5 \
  --run-id self_harm_debug
```

---

## Add Anthropic as a second provider

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

python cli.py run-all \
  --provider anthropic \
  --model claude-opus-4-5 \
  --model-id claude-opus-4-5-20250101 \
  --mode heuristic \
  --run-id anthropic_run_001
```

Compare results between `openai_run_001` and `anthropic_run_001` by reviewing
their respective `release_report.md` files side by side.

---

## Run tests

```bash
pytest tests/ -v           # all 157 tests
pytest tests/ -q           # summary only
pytest tests/test_evaluation_and_gating.py  # just the new eval/gate layer
```

All tests run without any API keys.

---

## CLI exit codes

| Code | Meaning |
|------|---------|
| `0` | SHIP — all thresholds met |
| `1` | BLOCK — one or more critical failures |
| `2` | CONDITIONAL_SHIP — warnings present, PM sign-off required |

Use in CI:
```bash
python cli.py report --run-id $RUN_ID --model-id $MODEL_ID
if [ $? -eq 1 ]; then echo "Release blocked"; exit 1; fi
```
