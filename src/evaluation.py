import cv2
from pathlib import Path
from model.sam import Sam, SamAutoBox
from model.sam import Sam, SamAutoBox
from model.baselines.canny_fill import CannyFill
from model.baselines.gaussian import Gaussian
from model.baselines.otsu import Otsu
from model.baselines.grabcut import GrabCutAutoBox, GrabCutAutoBrush
import pandas as pd

baselines = [
    Otsu(),
    Gaussian(),
    CannyFill(),
    GrabCutAutoBox(),
    GrabCutAutoBrush(),
    Sam(),
    SamAutoBox()
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
easy_ground_truths = [cv2.imread(ground_truth_folder / img_path.name, cv2.IMREAD_GRAYSCALE) for img_path in easy_raw_image_paths]
medium_raw_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in medium_raw_image_paths]
medium_ground_truths = [cv2.imread(ground_truth_folder / img_path.name, cv2.IMREAD_GRAYSCALE) for img_path in medium_raw_image_paths]
hard_raw_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in hard_raw_image_paths]
hard_ground_truths = [cv2.imread(ground_truth_folder / img_path.name, cv2.IMREAD_GRAYSCALE) for img_path in hard_raw_image_paths]


def perform_evaluation(raw_images, ground_truths, baselines, difficulty, csv_path="data/evaluation.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "model","difficulty","accuracy","precision","recall",
            "specificity","f1","iou"
        ])
        
    for baseline in baselines:
        model = baseline.name
        metrics = baseline.evaluate(raw_images, ground_truths, tolerance=5) # Assuming ground_truth is defined
        print(f"{model} metrics: {metrics}")
        
        row = {
            "model": model,
            "difficulty": difficulty,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "iou": metrics["iou"]
        }

        # get existing model, difficulty row if exists
        mask = (df["model"] == model) & (df["difficulty"] == difficulty)

        if mask.any():
            df.loc[mask] = list(row.values()) # overwrite
        else:
            df.loc[len(df)] = row   # append new
        
        df.to_csv(csv_path, index=False)
                

print("Evaluating on easy images...")
perform_evaluation(easy_raw_images, easy_ground_truths, baselines, difficulty="easy")
print("\nEvaluating on medium images...")
perform_evaluation(medium_raw_images, medium_ground_truths, baselines, difficulty="medium")
print("\nEvaluating on hard images...")
perform_evaluation(hard_raw_images, hard_ground_truths, baselines, difficulty="hard")