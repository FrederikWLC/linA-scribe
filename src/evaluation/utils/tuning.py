import pandas as pd
from pathlib import Path
from scribe.tunable import Tunable

def save_tuned_hyperparameters(model: Tunable, metric_name: str, metric_value: float, hyperparameters: dict, n_trials: int):
    base_columns = [metric_name] + list(model.hyperparameters) + ["n_trials"]
    df = pd.DataFrame(
        [{**{metric_name: metric_value}, **hyperparameters, "n_trials": n_trials}],
        columns=base_columns
    )
    output_path = Path(f"data/results/tuning/{model.name}/best.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Stored tuned hyperparameters for {model.name} at {output_path}")

def set_tuned_hyperparameters(model: Tunable):
    df = pd.read_csv(f"data/results/tuning/{model.name}/best.csv")
    values = df.iloc[0].to_dict()
    print(values)
    hyperparameters = {k: values[k] for k in model.configuration.hyperparameters}
    print(hyperparameters)
    model.set_hyperparameters(**hyperparameters)

def set_all_tuned_hyperparameters(models):
    for model in models:
        if is_tunable(model):
            set_tuned_hyperparameters(model)

def is_tunable(model) -> bool:
    return isinstance(model, Tunable) and model.is_tunable()
