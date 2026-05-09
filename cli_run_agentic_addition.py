# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS BLOCK TO cli.py
#
# Find the section where other commands are defined (e.g. near "run-all")
# and add this new command alongside them.
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("run-agentic")
@click.option("--provider", default="openai", help="Model provider")
@click.option("--model", default="gpt-4o", help="Model name")
@click.option("--model-id", required=True, help="Full model version ID")
@click.option("--run-id", default="run_003", help="Run identifier for output folder")
@click.option("--dataset", default=None, help="Path to agentic sequences JSONL (optional)")
def run_agentic(provider, model, model_id, run_id, dataset):
    """
    Run Phase 2 agentic evaluation: multi-turn tool-call sequences
    evaluated by the trajectory gate (Phase 1) + action gate (Phase 2).

    Requires OPENAI_API_KEY in environment.

    Output: results/{run_id}/agentic_evaluations.jsonl
            results/{run_id}/agentic_summary.json

    Example:
        python cli.py run-agentic \\
            --provider openai \\
            --model gpt-4o \\
            --model-id gpt-4o-2024-08-06 \\
            --run-id run_003
    """
    from src.runners.agentic_runner import AgenticRunner
    from pathlib import Path

    runner = AgenticRunner(
        model=model,
        model_id=model_id,
        provider=provider,
        run_id=run_id,
    )

    dataset_path = Path(dataset) if dataset else None
    runner.run(dataset_path=dataset_path)
