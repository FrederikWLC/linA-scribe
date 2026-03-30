import pandas as pd
from model.scribe import Tunable

def set_tuned_hyperparameters(baseline):
    df = pd.read_csv(f"data/tuning-{baseline.name}.csv")
    values = df.iloc[0].to_dict()
    hyperparameters = {k: values[k] for k in baseline.hyperparameters.keys()}
    print(hyperparameters)
    baseline.set_hyperparameters(**hyperparameters)

def set_all_tuned_hyperparameters(baselines):
    for baseline in baselines:
        if isinstance(baseline, (Tunable)):
            set_tuned_hyperparameters(baseline)