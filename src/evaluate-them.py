from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, probplot, shapiro, ttest_rel, wilcoxon
from model.baselines.canny_fill import CannyFill
from model.baselines.gaussian import Gaussian
from model.baselines.grabcut import GrabCutAutoBrush
from model.baselines.otsu import Otsu
from model.sam import MobileSAMv2, MobileSAMv2AutoPoint
from utils.evaluation import BinaryDiceScore, evaluate_model, summarize_results
from utils.tuning import set_all_tuned_hyperparameters

METRICS = {"Dice": BinaryDiceScore}
BASELINES = [
    Otsu(),
    Gaussian(),
    CannyFill(),
    GrabCutAutoBrush(), 
    MobileSAMv2(),
    MobileSAMv2AutoPoint()
    ]

DIFFICULTIES = ("easy", "medium", "hard")
RAW_ROOT = Path("data/raw")
GROUND_TRUTH_ROOT = Path("data/ground_truth/registered")

BASE_COLUMNS = ["difficulty","model"]
RESUME_COLUMNS = BASE_COLUMNS + [
    f"{metric}_{key}" for metric in METRICS.keys() for key in ["median","mean", "std", "std_error", "n"]
]
RAW_COLUMNS = BASE_COLUMNS.copy()
RAW_COLUMNS.insert(1, "label")
RAW_COLUMNS.extend(METRICS.keys())


def _variant_path(csv_path: str, suffix: str) -> Path:
    return Path(f"{csv_path[:-4]}-{suffix}.csv")


def _safe_read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        if path.stat().st_size > 0:
            return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def _load_difficulty_set(difficulty: str):
    image_paths = sorted((RAW_ROOT / difficulty).glob("*.jpg"))
    images = [cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE) for path in image_paths]
    ground_truths = [
        cv2.imread((GROUND_TRUTH_ROOT / path.name).as_posix(), cv2.IMREAD_GRAYSCALE)
        for path in image_paths
    ]
    labels = [path.stem for path in image_paths]
    return images, ground_truths, labels


def _build_raw_pivots(df_raw: pd.DataFrame, csv_path: str) -> None:
    if df_raw.empty:
        return
    for metric in METRICS.keys():
        pivot_df = (
            df_raw.pivot(index=["difficulty", "label"], columns="model", values=metric).reset_index()
        )
        pivot_df.to_csv(_variant_path(csv_path, f"raw-pivot-{metric}"), index=False)


def perform_evaluation(
    raw_images,
    ground_truths,
    baselines,
    difficulty,
    labels,
    csv_path: str = "data/evaluation.csv",
):
    df_raw = _safe_read_csv(_variant_path(csv_path, "raw"), RAW_COLUMNS)

    for baseline in baselines:
        model_name = baseline.name
        results, resume = evaluate_model(baseline, raw_images, ground_truths, metrics=METRICS)
        print(f"{model_name} resume: {resume}")

        raw_sub_dataframe = pd.DataFrame(results)
        raw_sub_dataframe.insert(0, "difficulty", difficulty)
        raw_sub_dataframe.insert(1, "label", labels)
        raw_sub_dataframe.insert(2, "model", model_name)

        df_raw = pd.concat([df_raw, raw_sub_dataframe], ignore_index=True)
        df_raw.drop_duplicates(subset=["difficulty", "label","model"], keep="last", inplace=True)
        df_raw.sort_values(by=["difficulty", "label","model"], inplace=True)

    df_raw.to_csv(_variant_path(csv_path, "raw"), index=False)
    _build_raw_pivots(df_raw, csv_path)


def do_resume(csv_path: str = "data/evaluation.csv"):
    df_raw = _safe_read_csv(_variant_path(csv_path, "raw"), RAW_COLUMNS)
    if df_raw.empty:
        return

    rows = []
    for baseline in BASELINES:
        model_name = baseline.name
        model_df = df_raw[df_raw["model"] == model_name]

        for difficulty in [*DIFFICULTIES, "all"]:
            subset = model_df if difficulty == "all" else model_df[model_df["difficulty"] == difficulty]
            if subset.empty:
                continue

            metric_values = {metric: subset[metric].dropna().tolist() for metric in METRICS.keys()}
            resume_row = {"model": model_name, "difficulty": difficulty}
            resume_row.update(summarize_results(metric_values))
            rows.append(resume_row)

    df_resume = pd.DataFrame(rows, columns=RESUME_COLUMNS)
    df_resume.drop_duplicates(subset=["model", "difficulty"], keep="last", inplace=True)
    df_resume.sort_values(by=["difficulty", "model"], inplace=True)
    df_resume.to_csv(_variant_path(csv_path, "resume"), index=False)


def do_statistical_tests(csv_path: str = "data/evaluation.csv", alpha: float = 0.05):
    df_raw = _safe_read_csv(_variant_path(csv_path, "raw"), RAW_COLUMNS)
    if df_raw.empty:
        return

    pairwise_cols = ["metric", "model1", "model2", "statistic", "p_value", "significant"]
    df_friedman_tests = _safe_read_csv(
        _variant_path(csv_path, "friedman-tests"),
        ["metric", "statistic", "p_value", "significant"],
    )
    df_wilcoxon_tests = _safe_read_csv(_variant_path(csv_path, "wilcoxon-tests"), pairwise_cols)
    df_paired_t_tests = _safe_read_csv(_variant_path(csv_path, "paired-t-tests"), pairwise_cols)
    df_shapiro_tests = _safe_read_csv(_variant_path(csv_path, "shapiro-tests"), pairwise_cols)
    
    metric_names = [col for col in df_raw.columns if col not in ["model", "difficulty", "label"]]

    for metric in metric_names:
        models = df_raw["model"].unique().tolist()

        groups = [df_raw[df_raw["model"] == model][metric].dropna().to_numpy() for model in models]
        friedman_statistic, friedman_p_value = friedmanchisquare(*groups)
        friedman_significant = bool(friedman_p_value < alpha) if not np.isnan(friedman_p_value) else False

        df_friedman_tests = pd.concat(
            [
                df_friedman_tests,
                pd.DataFrame(
                    [
                        {
                            "metric": metric,
                            "statistic": friedman_statistic,
                            "p_value": friedman_p_value,
                            "significant": friedman_significant,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        df_friedman_tests.drop_duplicates(subset=["metric"], keep="last", inplace=True)

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                data1 = df_raw[df_raw["model"] == model1][metric].dropna().to_numpy()
                data2 = df_raw[df_raw["model"] == model2][metric].dropna().to_numpy()

                wilcoxon_statistic, wilcoxon_p_value = wilcoxon(data1, data2)
                wilcoxon_significant = bool(wilcoxon_p_value < alpha)

                residuals = data1 - data2
                do_qqplot(residuals, metric, model1, model2)

                shapiro_statistic, shapiro_p_value = shapiro(residuals)
                shapiro_significant = bool(shapiro_p_value < alpha)

                paired_t_statistic, paired_t_p_value = ttest_rel(data1, data2)
                paired_t_significant = bool(paired_t_p_value < alpha)

                pair_rows = {
                    "metric": metric,
                    "model1": model1,
                    "model2": model2,
                }

                df_wilcoxon_tests = pd.concat(
                    [
                        df_wilcoxon_tests,
                        pd.DataFrame(
                            [
                                {
                                    **pair_rows,
                                    "statistic": wilcoxon_statistic,
                                    "p_value": wilcoxon_p_value,
                                    "significant": wilcoxon_significant,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                df_shapiro_tests = pd.concat(
                    [
                        df_shapiro_tests,
                        pd.DataFrame(
                            [
                                {
                                    **pair_rows,
                                    "statistic": shapiro_statistic,
                                    "p_value": shapiro_p_value,
                                    "significant": shapiro_significant,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                df_paired_t_tests = pd.concat(
                    [
                        df_paired_t_tests,
                        pd.DataFrame(
                            [
                                {
                                    **pair_rows,
                                    "statistic": paired_t_statistic,
                                    "p_value": paired_t_p_value,
                                    "significant": paired_t_significant,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            subset_cols = ["metric", "model1", "model2"]
            df_wilcoxon_tests.drop_duplicates(subset=subset_cols, keep="last", inplace=True)
            df_paired_t_tests.drop_duplicates(subset=subset_cols, keep="last", inplace=True)
            df_shapiro_tests.drop_duplicates(subset=subset_cols, keep="last", inplace=True)

    df_friedman_tests.to_csv(_variant_path(csv_path, "friedman-tests"), index=False)
    df_wilcoxon_tests.to_csv(_variant_path(csv_path, "wilcoxon-tests"), index=False)
    df_paired_t_tests.to_csv(_variant_path(csv_path, "paired-t-tests"), index=False)
    df_shapiro_tests.to_csv(_variant_path(csv_path, "shapiro-tests"), index=False)


def do_qqplot(residuals, metric, model1, model2):
    Path("data/qqplots").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot of {metric} Residuals for {model1} vs {model2}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Residuals")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/qqplots/{metric}_{model1}_vs_{model2}.png")
    plt.close()


def _series_by_model(df_resume: pd.DataFrame, difficulty: str, metric_field: str, model_names: list[str]) -> list[float]:
    subset = df_resume[df_resume["difficulty"] == difficulty]
    values_by_model = subset.set_index("model")[metric_field].to_dict()
    return np.array([values_by_model.get(model, 0.0) for model in model_names])


def do_barplots(csv_path: str = "data/evaluation.csv"):
    df_resume = _safe_read_csv(_variant_path(csv_path, "resume"), RESUME_COLUMNS)
    if df_resume.empty:
        return
    
    model_names = np.array([baseline.name for baseline in BASELINES])

    for metric in METRICS.keys():

        metric_mean = _series_by_model(df_resume, "all", f"{metric}_mean", model_names)
        easy_mean = _series_by_model(df_resume, "easy", f"{metric}_mean", model_names)
        medium_mean = _series_by_model(df_resume, "medium", f"{metric}_mean", model_names)
        hard_mean = _series_by_model(df_resume, "hard", f"{metric}_mean", model_names)

        metric_std_error = _series_by_model(df_resume, "all", f"{metric}_std_error", model_names)
        easy_std_error = _series_by_model(df_resume, "easy", f"{metric}_std_error", model_names)
        medium_std_error = _series_by_model(df_resume, "medium", f"{metric}_std_error", model_names)
        hard_std_error = _series_by_model(df_resume, "hard", f"{metric}_std_error", model_names)

        order = np.argsort(-metric_mean)

        model_names = model_names[order]
        easy_mean = easy_mean[order]
        medium_mean = medium_mean[order]
        hard_mean = hard_mean[order]
        metric_mean = metric_mean[order]
        metric_std_error = metric_std_error[order]

        x = np.arange(len(model_names))
        width = 0.15

        x_metric = x - 1.5 * width
        x_easy = x - 0.5 * width
        x_medium = x + 0.5 * width
        x_hard = x + 1.5 * width

        _, ax = plt.subplots(figsize=(8, 5))

        # Mean Metric (background bars)
        ax.bar(x_metric, metric_mean, width, yerr=metric_std_error, capsize=4, 
            alpha=1, label='Mean ' + metric.capitalize() + ' (All Difficulties)', color='lightblue', edgecolor='none',
            error_kw=dict(
                ecolor='deepskyblue',
                elinewidth=1.5,
                capthick=1.5
            ))

        # Difficulty component bars
        ax.bar(x_easy, easy_mean, width, yerr=easy_std_error, capsize=4, label='Easy',color='green',
               alpha=1,
               error_kw=dict(
                ecolor='darkgreen',
                elinewidth=1.5,
                capthick=1.5
            ))
        ax.bar(x_medium, medium_mean, width, yerr=medium_std_error, capsize=4, label='Medium',color='orange',
               alpha=1,
               error_kw=dict(
                ecolor='darkorange',
                elinewidth=1.5,
                capthick=1.5
            ))
        ax.bar(x_hard, hard_mean, width, yerr=hard_std_error, capsize=4, label='Hard',color='red',
               alpha=1,
               error_kw=dict(
                ecolor='darkred',
                elinewidth=1.5,
                capthick=1.5
            ))

        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Score per Model with Difficulty Breakdown")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

        Path("data/plots").mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"data/plots/{metric}_score_comparison.png")
        plt.close()

def do_boxplots(csv_path: str = "data/evaluation.csv"):
    df_resume = _safe_read_csv(_variant_path(csv_path, "resume"), RESUME_COLUMNS)
    if df_resume.empty:
        return
    
    base_model_names = np.array([baseline.name for baseline in BASELINES])

    for metric in METRICS.keys():

        df_raw_pivot = _safe_read_csv(_variant_path(csv_path, f"raw-pivot-{metric}"), RAW_COLUMNS)
        if df_raw_pivot.empty:
            continue

        metric_median = _series_by_model(df_resume, "all", f"{metric}_median", base_model_names)
        order = np.argsort(-metric_median)

        model_names = base_model_names[order]

        data = [df_raw_pivot[m].values for m in model_names]

        x = np.arange(len(model_names))

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        ax.boxplot(
            data,
            positions=x,
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=1),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black')
        )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_title(f"{metric.capitalize()} Distribution per Model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.capitalize())

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        Path("data/plots").mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"data/plots/{metric}_boxplot.png")
        plt.close()


def run_full_evaluation(csv_path: str = "data/evaluation.csv"):
    set_all_tuned_hyperparameters(BASELINES) # ensure all baselines have their tuned hyperparameters set before starting evaluation
    do_boxplots(csv_path=csv_path) # ensure boxplots are up to date before starting evaluation
    do_barplots(csv_path=csv_path) # ensure barplots are up to date before starting evaluation
    do_statistical_tests(csv_path=csv_path) # ensure statistical tests are up to date before starting evaluation
    for difficulty in DIFFICULTIES:
        print(f"Evaluating on {difficulty} images...")
        raw_images, ground_truths, labels = _load_difficulty_set(difficulty)
        perform_evaluation(
            raw_images,
            ground_truths,
            BASELINES,
            difficulty=difficulty,
            labels=labels,
            csv_path=csv_path,
        )
        do_resume(csv_path=csv_path)
        do_statistical_tests(csv_path=csv_path)

    do_resume(csv_path=csv_path)
    do_barplots(csv_path=csv_path)
    do_boxplots(csv_path=csv_path)
    do_statistical_tests(csv_path=csv_path)

if __name__ == "__main__":
    print("Starting full evaluation...")
    run_full_evaluation()