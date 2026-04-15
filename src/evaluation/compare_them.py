import cv2
from pathlib import Path
from data.split import DIFFICULTIES, get_evaluation_data
from evaluation.baselines.sam import BestMobileSAMv2Implementation
from evaluation.baselines.canny_fill import CannyFill
from evaluation.baselines.gaussian import Gaussian
from evaluation.baselines.otsu import Otsu
from evaluation.baselines.grabcut import GrabCutAutoBrush
from scribe.base import predict
from evaluation.utils.tuning import set_all_tuned_hyperparameters
from data.split import get_evaluation_data


# Get evaluation data, from data split module
evaluation_data = get_evaluation_data(seed=42)

output_folder = Path("data/results/scribed")

# Define baselines (ScribeSAM included by default)
baselines = [
    CannyFill(),
    Gaussian(),
    Otsu(),
    GrabCutAutoBrush(),
    BestMobileSAMv2Implementation()
]

def perform_comparison(raw_images, labels, baselines):
    set_all_tuned_hyperparameters(baselines)
    for img, label in zip(raw_images, labels):
        img_name = f"{label}.jpg"
        print(f"\nSegmenting {img_name}...")

        for baseline in baselines:
            try:
                prediction = predict(baseline, img)
                image = prediction.to_image()
                output_path = output_folder / f'{img_name[:-4]}-{baseline.name}.jpg'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), image)
                print(f"Done for {baseline.name}!")
            except Exception as exc:
                print(f"Skipped {baseline.name} due to error: {exc}")


def run_full_comparison(evaluation_data, baselines):
    for difficulty in DIFFICULTIES:
        if difficulty not in evaluation_data:
            continue

        dataset = evaluation_data[difficulty]
        print(f"\nPerforming comparison on {difficulty} images...")
        perform_comparison(dataset["images"], dataset["labels"], baselines)

output_folder.mkdir(parents=True, exist_ok=True)

def run_default_comparison():
    evaluation_data = get_evaluation_data()
    run_full_comparison(evaluation_data=evaluation_data, baselines=baselines)
