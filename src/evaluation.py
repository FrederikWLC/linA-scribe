import cv2
from pathlib import Path
from model.sam import Sam
from model.baselines.canny_fill import CannyFill
from model.baselines.gaussian import Gaussian
from model.baselines.otsu import Otsu
from model.baselines.grabcut import GrabCutAutoBrush
import pandas as pd
from utils.evaluation import evaluate_model, BinaryF1Score, BinaryIoU
from scipy.stats import friedmanchisquare, wilcoxon

metrics = {"f1": BinaryF1Score, "iou": BinaryIoU}

baselines = [
    Otsu(),
    Gaussian(),
    CannyFill(),
    GrabCutAutoBrush(),
    Sam()
    ]

raw_folder = Path("data/raw")
easy_raw_folder = raw_folder / "easy"
medium_raw_folder = raw_folder / "medium"
hard_raw_folder = raw_folder / "hard"
ground_truth_folder = Path("data/ground_truth/registered")
output_folder = Path("data")


easy_raw_image_paths = list(easy_raw_folder.glob("*.jpg"))
medium_raw_image_paths = list(medium_raw_folder.glob("*.jpg"))
hard_raw_image_paths = list(hard_raw_folder.glob("*.jpg"))

easy_raw_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in easy_raw_image_paths]
medium_raw_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in medium_raw_image_paths]
hard_raw_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in hard_raw_image_paths]

easy_ground_truths = [cv2.imread(ground_truth_folder / img_path.name, cv2.IMREAD_GRAYSCALE) for img_path in easy_raw_image_paths]
medium_ground_truths = [cv2.imread(ground_truth_folder / img_path.name, cv2.IMREAD_GRAYSCALE) for img_path in medium_raw_image_paths]
hard_ground_truths = [cv2.imread(ground_truth_folder / img_path.name, cv2.IMREAD_GRAYSCALE) for img_path in hard_raw_image_paths]

raw_columns = resume_columns = base_columns = ["model","difficulty"]
resume_columns += [f"{m}_{key}" for m in metrics.keys() for key in ["mean", "std", "std_error","n"]]
raw_columns += metrics.keys()

def perform_evaluation(raw_images, ground_truths, baselines, difficulty, labels, csv_path="data/evaluation.csv"):
    try:
        df_raw = pd.read_csv(csv_path[:-4]+"-raw.csv")
        df_resume = pd.read_csv(csv_path[:-4]+"-resume.csv")
    except:
        df_raw = pd.DataFrame(columns=raw_columns)
        df_resume = pd.DataFrame(columns=resume_columns)

    for baseline in baselines:
        model = baseline.name
        results, resume = evaluate_model(baseline, raw_images, ground_truths, metrics=metrics)
        print(f"{model} resume: {resume}")
        
        # do raw dataframe ----------------------------
        raw_sub_dataframe = pd.DataFrame(results)
        raw_sub_dataframe.insert(0, "model", model)
        raw_sub_dataframe.insert(1, "difficulty", difficulty)
        raw_sub_dataframe.insert(2, "label", labels)
        df_raw = pd.concat([df_raw, raw_sub_dataframe], ignore_index=True)
        df_raw.drop_duplicates(subset=["model","difficulty","label"], keep="last", inplace=True) # keeps only new changs in case of duplicates 

        # do resume dataframe ----------------------------
        resume_row = {
            "model": model,
            "difficulty": difficulty
        }
        resume_row.update(resume)
        df_resume = pd.concat([df_resume, pd.DataFrame([resume_row])], ignore_index=True)
        df_resume.drop_duplicates(subset=["model","difficulty"], keep="last", inplace=True) # keeps only new changs in case of duplicates

        # save both files
        df_raw.to_csv(csv_path[:-4]+"-raw.csv", index=False)
        df_resume.to_csv(csv_path[:-4]+"-resume.csv", index=False)

def do_statistical_tests(csv_path="data/evaluation.csv", alpha=0.05):
    df_raw = pd.read_csv(csv_path[:-4]+"-raw.csv")
    try:
        df_friedman_tests = pd.read_csv(csv_path[:-4]+"-friedman-tests.csv")
        df_wilcoxon_tests = pd.read_csv(csv_path[:-4]+"-wilcoxon-tests.csv")
    except:
        df_friedman_tests = pd.DataFrame(columns=["metric", "difficulty", "models", "statistic", "p_value", "significant"])
        df_wilcoxon_tests = pd.DataFrame(columns=["metric", "difficulty", "models", "statistic", "p_value", "significant"])
    metrics = [col for col in df_raw.columns if col not in ["model","difficulty","label"]]

    # perform friedman test for each metric and difficulty level (including all together)
    for metric in metrics:
        difficulties = df_raw["difficulty"].unique().tolist() + ["all"]
        for difficulty in difficulties:
            subset = df_raw.copy()
            if difficulty != "all":
                subset = df_raw[df_raw["difficulty"] == difficulty]

            groups = [subset[subset["model"] == model][metric].tolist() for model in subset["model"].unique()]
            statistic, p_value = friedmanchisquare(*groups)
            significant = p_value < alpha
            df_friedman_tests = pd.concat([df_friedman_tests, pd.DataFrame([{
                "metric": metric,
                "difficulty": difficulty,
                "statistic": statistic,
                "p_value": p_value,
                "significant": significant
            }])], ignore_index=True)
            
            # DO REPLACEMENT OF ROWS WITH SAME MODEL AND DIFFICULTY (keep only last one)
            df_friedman_tests.drop_duplicates(subset=["metric", "difficulty"], keep="last", inplace=True)

            models = subset["model"].unique()
            for i in range(len(models)):
                for j in range(i+1, len(models)): # avoid repeating pairs and testing same models
                    model1, model2 = models[i], models[j]
                    data1 = subset[subset["model"] == model1][metric]
                    data2 = subset[subset["model"] == model2][metric]
                    statistic, p_value = wilcoxon(data1, data2)
                    significant = p_value < alpha
                    df_wilcoxon_tests = pd.concat([df_wilcoxon_tests, pd.DataFrame([{
                        "metric": metric,
                        "difficulty": difficulty,
                        "model1": model1,
                        "model2": model2,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": significant
                    }])], ignore_index=True)
            df_wilcoxon_tests.drop_duplicates(subset=["metric","difficulty","model1","model2"],keep="last",inplace=True)

        
    df_friedman_tests.to_csv(csv_path[:-4]+"-friedman-tests.csv", index=False)       
    df_wilcoxon_tests.to_csv(csv_path[:-4]+"-wilcoxon-tests.csv", index=False)


do_statistical_tests()
print("Evaluating on easy images...")
perform_evaluation(easy_raw_images, easy_ground_truths, baselines, difficulty="easy",labels=[path.name[:-4] for path in easy_raw_image_paths])
do_statistical_tests()
print("\nEvaluating on medium images...")
perform_evaluation(medium_raw_images, medium_ground_truths, baselines, difficulty="medium", labels=[path.name[:-4] for path in medium_raw_image_paths])
do_statistical_tests()
print("\nEvaluating on hard images...")
perform_evaluation(hard_raw_images, hard_ground_truths, baselines, difficulty="hard", labels=[path.name[:-4] for path in hard_raw_image_paths])
do_statistical_tests()