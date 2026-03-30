from pathlib import Path

import cv2
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from model.baselines.canny_fill import CannyFill
from model.baselines.gaussian import Gaussian
from model.baselines.grabcut import GrabCutAutoBrush
# OTSU HAS NO HYPERPARAMETERS, NOT INCLUDED IN TUNING
from model.sam import MobileSAMv2AutoPoint
from utils.evaluation import BinaryDiceScore, evaluate_model



METRIC = {"Dice": BinaryDiceScore} # only choose one metric pls
METRIC_NAME = list(METRIC.keys())[0]
# the baselines that can be tuned
BASELINES = [
    #CannyFill(), 
    #Gaussian(), 
    #GrabCutAutoBrush() 
    MobileSAMv2AutoPoint()
    ]
DIFFICULTIES = ("easy", "medium", "hard")
RAW_ROOT = Path("data/raw")
GROUND_TRUTH_ROOT = Path("data/ground_truth/registered")

def _load_dataset():
    image_paths = [path for difficulty in DIFFICULTIES for path in sorted((RAW_ROOT / difficulty).glob("*.jpg"))]
    images = [cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE) for path in image_paths]
    ground_truths = [
        cv2.imread((GROUND_TRUTH_ROOT / path.name).as_posix(), cv2.IMREAD_GRAYSCALE)
        for path in image_paths
    ]
    labels = [path.stem for path in image_paths]
    return images, ground_truths, labels

def evaluation_trial(trial, baseline, images, ground_truths):
    # set hyperparameters accoridng to trial's suggestions
    baseline.set_hyperparameters(**baseline.hyperparameter_ranges(trial)) 
    _, resume = evaluate_model(baseline,X=images,Y=ground_truths,metrics=METRIC) # evaluation resume
    score = resume[METRIC_NAME+"_mean"] # mean Dice score from the resume
    return score

def perform_tuning(n_trials=100):
    X,Y,_ = _load_dataset()
    for baseline in BASELINES:
        print(f"\nTuning {baseline.name}...")
        # Hyper optimization with Optuna using random search (seeded for reproducibility)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42)) # 42 the meaning of life, why not?
        study.optimize(lambda trial: evaluation_trial(trial, baseline, X, Y), n_trials=n_trials)
        
        # display results
        print(f"Best {METRIC_NAME}: {study.best_value}")
        print(f"Best parameters: {study.best_params}")

        # Store study results in a csv for documentation
        df_trials = study.trials_dataframe()
        df_trials.to_csv(f"data/tuning-{baseline.name}-trials.csv", index=False)

        # Store best results in a csv for documentation
        BASE_COLUMNS = [METRIC_NAME] + list(baseline.hyperparameters.keys()) + ["n_trials"]
        df_best = pd.DataFrame([{**{METRIC_NAME: study.best_value}, **study.best_params, "n_trials": n_trials}],columns=BASE_COLUMNS)
        df_best.to_csv(f"data/tuning-{baseline.name}.csv", index=False)

        
if __name__ == "__main__":
    perform_tuning()


