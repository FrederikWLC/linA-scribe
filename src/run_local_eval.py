from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent


def _prepare_local_python_context() -> None:
    if not (SRC_DIR / "evaluation").exists():
        raise RuntimeError(f"Missing evaluation directory: {SRC_DIR / 'evaluation'}")

    # Keep relative data/ paths aligned with existing scripts.
    os.chdir(SRC_DIR)

    # Ensure imports like evaluation.* and data.* resolve from local host runs.
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def run_local(task: str, n_trials: int) -> None:
    _prepare_local_python_context()

    if task in {"tune", "all"}:
        from evaluation.tune_them import perform_tuning

        perform_tuning(n_trials=n_trials)

    if task in {"evaluate", "all"}:
        from evaluation.evaluate_them import run_default_evaluation

        run_default_evaluation()

    if task in {"ablate", "all"}:
        from evaluation.ablate_SAM import run_ablation

        run_ablation()

    if task in {"compare", "all"}:
        from evaluation.compare_them import run_default_comparison

        run_default_comparison()

    if task in {"autoprompt", "all"}:
        from evaluation.save_autoprompt_displays import run_autoprompt_export

        run_autoprompt_export()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local evaluation workflows outside Docker.")
    parser.add_argument(
        "task",
        choices=["tune", "evaluate", "ablate", "compare", "autoprompt", "all"],
        help="Which local workflow to run.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for tune (only used for tune/all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_local(task=args.task, n_trials=args.n_trials)


main()
