from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from data.split import DIFFICULTIES, get_test_data_by_difficulty
from evaluation.baselines.canny_fill import CannyFill
from evaluation.baselines.gaussian import Gaussian
from evaluation.baselines.grabcut import GrabCutAutoBrush
from evaluation.baselines.otsu import Otsu
from evaluation.baselines.sam import (
    MobileSAMv2AutoPointBilateralFilter,
    MobileSAMv2AutoPointBilateralFilterBestOfThree,
    MobileSAMv2AutoPointNoFilter,
    MobileSAMv2AutoPointNoFilterBestOfThree,
    MobileSAMv2BilateralFilter,
    MobileSAMv2BilateralFilterBestOfThree,
    MobileSAMv2NoFilterBestOfThree,
    MobileSAMv2NoFilter
)
from evaluation.utils.metrics import BinaryDiceScore, evaluate_model


METRIC = {"Dice": BinaryDiceScore} # only choose one metric pls
METRIC_NAME = list(METRIC.keys())[0]
MODELS = [
    CannyFill(),
    Gaussian(),
    Otsu(),
    GrabCutAutoBrush(),
    MobileSAMv2AutoPointBilateralFilter(),
    MobileSAMv2AutoPointBilateralFilterBestOfThree(),
    MobileSAMv2AutoPointNoFilter(),
    MobileSAMv2AutoPointNoFilterBestOfThree(),
    MobileSAMv2BilateralFilter(),
    MobileSAMv2BilateralFilterBestOfThree(),
    MobileSAMv2NoFilterBestOfThree(),
    MobileSAMv2NoFilter()
]

def _load_dataset(seed: int = 42):
    evaluation_data = get_test_data_by_difficulty(seed=seed)
    images = []
    ground_truths = []
    labels = []

    for difficulty in DIFFICULTIES:
        if difficulty not in evaluation_data:
            continue
        split = evaluation_data[difficulty]
        images.extend(split["images"])
        ground_truths.extend(split["ground_truths"])
        labels.extend(split["labels"])

    return images, ground_truths, labels

def evaluation_trial(trial, baseline, images, ground_truths):
    # set hyperparameters accoridng to trial's suggestions
    baseline.set_hyperparameters(**baseline.hyperparameter_ranges(trial)) 
    _, resume = evaluate_model(baseline,X=images,Y=ground_truths,metrics=METRIC) # evaluation resume
    score = resume[METRIC_NAME+"_mean"] # mean Dice score from the resume
    return score

def perform_tuning(n_trials=100):
    X,Y,_ = _load_dataset()
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




