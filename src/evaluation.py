import cv2
from pathlib import Path
from model.sam import Sam, SamAutoBox
from model.baselines import *

baselines = [
    CannyFill(),
    Gaussian(),
    Otsu(),
    GrabCut(),
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


def perform_evaluation(raw_images, ground_truths, baselines):
    for baseline in baselines:
        metrics = baseline.evaluate(raw_images, ground_truths, tolerance=5) # Assuming ground_truth is defined
        print(f"{baseline.__class__.__name__} metrics: {metrics}")

print("Evaluating on easy images...")
perform_evaluation(easy_raw_images, easy_ground_truths, baselines)
print("\nEvaluating on medium images...")
perform_evaluation(medium_raw_images, medium_ground_truths, baselines)
print("\nEvaluating on hard images...")
perform_evaluation(hard_raw_images, hard_ground_truths, baselines)