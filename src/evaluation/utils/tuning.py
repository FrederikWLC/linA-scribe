import pandas as pd
from pathlib import Path
from config import config
from scribe.tunable import Tunable

def save_tuned_hyperparameters(model: Tunable, metric_name: str, metric_value: float, hyperparameters: dict, n_trials: int):
    base_columns = [metric_name] + list(model.hyperparameters) + ["n_trials"]
    df = pd.DataFrame(
        [{**{metric_name: metric_value}, **hyperparameters, "n_trials": n_trials}],
        columns=base_columns
    )
    output_path = config.DATA_DIR / "results" / "tuning" / model.name / "best.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def set_tuned_hyperparameters(model: Tunable):
    df = pd.read_csv(config.DATA_DIR / "results" / "tuning" / model.name / "best.csv")
    values = df.iloc[0].to_dict()
    model.set_hyperparameters_from(**values)

def set_all_tuned_hyperparameters(models):
    for model in models:
        if is_tunable(model):
            set_tuned_hyperparameters(model)

def is_tunable(model) -> bool:
    return isinstance(model, Tunable) and model.is_tunable()
