import pandas as pd
from model.scribe import Tunable

def set_tuned_hyperparameters(model: Tunable):
    df = pd.read_csv(f"data/tuning-{model.name}.csv")
    values = df.iloc[0].to_dict()
    print(values)
    hyperparameters = {k: values[k] for k in model.hyperparameters.keys()}
    print(hyperparameters)
    model.set_hyperparameters(**hyperparameters)

def set_all_tuned_hyperparameters(models):
    for model in models:
        if isinstance(model, (Tunable)):
            set_tuned_hyperparameters(model)