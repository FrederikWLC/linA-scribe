import sys
from argparse import ArgumentParser
from pathlib import Path

# Allow running this file directly from the src directory while using src.* imports.
if __package__ in (None, ""):
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from evaluation.tool.ui.controller import Controller
from evaluation.tool.ui.main_window import MainWindow


def run_gui():
    app = QApplication(sys.argv)
    controller = Controller()
    window = MainWindow(controller)
    window.show()
    sys.exit(app.exec())


def run(argv=None):
    parser = ArgumentParser(description="linA-scribe task runner")
    parser.add_argument(
        "task",
        nargs="?",
        default="gui",
        choices=["gui", "tune", "evaluate", "ablate", "compare", "autoprompt"],
    )
    args = parser.parse_args(argv)

    if args.task == "gui":
        run_gui()
        return

    if args.task == "tune":
        from evaluation.tune_them import perform_tuning

        perform_tuning()
        return

    if args.task == "evaluate":
        from evaluation.evaluate_them import run_default_evaluation

        run_default_evaluation()
        return

    if args.task == "ablate":
        from evaluation.ablate_SAM import run_ablation

        run_ablation()
        return

    if args.task == "compare":
        from evaluation.compare_them import run_default_comparison

        run_default_comparison()
        return

    if args.task == "autoprompt":
        from evaluation.save_autoprompt_displays import run_autoprompt_export

        run_autoprompt_export()
        return
