from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from data.split import get_training_data
from scribe.baselines.canny_fill import CannyFill
from scribe.baselines.gaussian import Gaussian
from scribe.baselines.grabcut import GrabCutAutoBrush
from scribe.baselines.otsu import Otsu
from scribe.baselines.sam import (
    MobileSAMv2AutoPointBilateralFilter,
    MobileSAMv2AutoPointBilateralFilterBestOfThree,
    MobileSAMv2AutoPointNoFilter,
    MobileSAMv2AutoPointNoFilterBestOfThree,
    MobileSAMv2BilateralFilter,
    MobileSAMv2BilateralFilterBestOfThree,
    MobileSAMv2NoFilterBestOfThree,
    MobileSAMv2NoFilter
)
from fatesam2d_api.ModalFATESAM2D import ModalFATESAM2DAutoPoint
from evaluation.utils.metrics import BinaryDiceScore, evaluate_model


METRIC = {"Dice": BinaryDiceScore} # only choose one metric pls
METRIC_NAME = list(METRIC.keys())[0]
MODELS = [
    #CannyFill(),
    #Gaussian(),
    #Otsu(),
    #MobileSAMv2AutoPointBilateralFilter(),
    #MobileSAMv2AutoPointBilateralFilterBestOfThree(),
    #MobileSAMv2AutoPointNoFilter(),
    #MobileSAMv2AutoPointNoFilterBestOfThree(),
    #MobileSAMv2BilateralFilter(),
    #MobileSAMv2BilateralFilterBestOfThree(),
    #MobileSAMv2NoFilterBestOfThree(),
    #MobileSAMv2NoFilter(),
    ModalFATESAM2DAutoPoint(),
    GrabCutAutoBrush(),

]


def evaluation_trial(trial, baseline, images, ground_truths):
    # set hyperparameters accoridng to trial's suggestions
    baseline.set_hyperparameters(**baseline.hyperparameter_ranges(trial)) 
    _, resume = evaluate_model(baseline,X=images,Y=ground_truths,metrics=METRIC) # evaluation resume
    score = resume[METRIC_NAME+"_mean"] # mean Dice score from the resume
    return score

def perform_tuning(n_trials=100):
    X,Y,_ = get_training_data(seed=42) # get training data for tuning, seed for reproducibility
    output_dir = Path("data/results/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        model_output_dir = output_dir / model.name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTuning {model.name}...")
        # Hyper optimization with Optuna using random search (seeded for reproducibility)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42)) # 42 the meaning of life, why not?
        study.optimize(lambda trial: evaluation_trial(trial, model, X, Y), n_trials=n_trials)
        
        # display results
        print(f"Best {METRIC_NAME}: {study.best_value}")
        print(f"Best parameters: {study.best_params}")

        # Store study results in a csv for documentation
        df_trials = study.trials_dataframe()
        df_trials.to_csv(model_output_dir / "trials.csv", index=False)

        # Store best results in a csv for documentation
        base_columns = [METRIC_NAME] + list(model.hyperparameters.keys()) + ["n_trials"]
        df_best = pd.DataFrame([{**{METRIC_NAME: study.best_value}, **study.best_params, "n_trials": n_trials}], columns=base_columns)
        df_best.to_csv(model_output_dir / "best.csv", index=False)




