import pandas as pd
from scribe.tunable import Tunable

def set_tuned_hyperparameters(model: Tunable,):
    df = pd.read_csv(f"data/results/tuning/{model.name}/best.csv")
    values = df.iloc[0].to_dict()
    print(values)
    hyperparameters = {k: values[k] for k in model.hyperparameters.keys()}
    print(hyperparameters)
    model.set_hyperparameters(**hyperparameters)

def set_all_tuned_hyperparameters(models):
    for model in models:
        if isinstance(model, (Tunable)):
            if len(model.hyperparameters) > 0:
                set_tuned_hyperparameters(model)
