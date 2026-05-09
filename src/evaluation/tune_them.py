from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from data.split import get_val_data
from fatesam2d_api.ModalFATESAM2D import build_all_tunable_modal_fatesam2d_variants
from scribe.baselines.canny_fill import build_cannyfill
from scribe.baselines.gaussian import build_gaussian
from scribe.baselines.otsu import build_otsu
from evaluation.utils.metrics import BinaryDiceScore, evaluate_model
from evaluation.utils.tuning import save_tuned_hyperparameters
from sam_api.modal_sam import build_all_tunable_modal_sam_variants

METRIC = {"Dice": BinaryDiceScore} # only choose one metric pls
METRIC_NAME = list(METRIC.keys())[0]

def get_models_to_be_tuned():
    return [
        #build_cannyfill(),
        #build_gaussian(),
        #build_otsu(),
    ] + build_all_tunable_modal_sam_variants() + build_all_tunable_modal_fatesam2d_variants()

# NOTE: WRITE IT BACK BEFORE COMMIT!

def evaluation_trial(trial, model, images, ground_truths):
    # set hyperparameters according to trial's suggestions
    model.set_hyperparameters(**model.hyperparameter_ranges(trial))
    _, resume = evaluate_model(model, X=images, Y=ground_truths, metrics=METRIC) # evaluation resume
    score = resume[METRIC_NAME+"_mean"] # mean Dice score from the resume
    return score

def perform_tuning(n_trials=100):
    models = get_models_to_be_tuned()
    X,Y,_ = get_val_data() # get validation data for tuning
    output_dir = Path("data/results/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        model_output_dir = output_dir / model.name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTuning {model.name}...")
        # Hyperparameter optimization with Optuna using random search (seeded for reproducibility)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42)) # 42 the meaning of life, why not?
        study.optimize(lambda trial: evaluation_trial(trial, model, X, Y), n_trials=n_trials)
        
        # display results
        print(f"Best {METRIC_NAME}: {study.best_value}")
        print(f"Best parameters: {study.best_params}")

        # Store study results in csv for documentation
        df_trials = study.trials_dataframe()
        df_trials.to_csv(model_output_dir / "trials.csv", index=False)

        # Store best results in csv for documentation
        save_tuned_hyperparameters(
            model=model,
            metric_name=METRIC_NAME,
            metric_value=study.best_value,
            hyperparameters=study.best_params,
            n_trials=n_trials,
        )



