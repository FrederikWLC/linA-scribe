import os
import cv2
import torch
from pathlib import Path
from data.split import DIFFICULTIES, get_evaluation_data
from model.baselines.sam import BestMobileSAMv2Implementation
from model.baselines.canny_fill import CannyFill
from model.baselines.gaussian import Gaussian
from model.baselines.otsu import Otsu
from model.baselines.grabcut import GrabCutAutoBrush
from model.scribe_sam import ScribeSAM
from model.scribe import predict
from utils.tuning import set_all_tuned_hyperparameters
from data.split import get_training_data, get_evaluation_data


# Get evaluation data, from data split module
training_images, training_ground_truths, training_labels = get_training_data(seed=42)
evaluation_data = get_evaluation_data(seed=42)



def _build_baselines():
    baselines = [
        CannyFill(),
        Gaussian(),
        Otsu(),
        GrabCutAutoBrush(),
        BestMobileSAMv2Implementation(),
    ]

    include_scribe = os.getenv("INCLUDE_SCRIBE_SAM", "0").strip().lower() in {"1", "true", "yes", "on"}
    if include_scribe:
        if not torch.cuda.is_available():
            print("Skipping ScribeSAM: CUDA not available on this machine.")
        else:
            baselines.append(
                ScribeSAM(
                    support_images=training_images,
                    support_labels=training_ground_truths,
                    top_n=1,
                    image_size=512,
                )
            )
    else:
        print("ScribeSAM is disabled. Set INCLUDE_SCRIBE_SAM=1 to enable it.")

    return baselines


baselines = _build_baselines()

output_folder = Path("data/scribed")

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

if __name__ == "__main__":
    evaluation_data = get_evaluation_data()
    run_full_comparison(evaluation_data=evaluation_data, baselines=baselines)