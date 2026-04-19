import cv2
from pathlib import Path
from data.split import DIFFICULTIES, get_test_data_by_difficulty, get_training_data
from evaluation.baselines.sam import BestMobileSAMv2Implementation
from evaluation.baselines.canny_fill import CannyFill
from evaluation.baselines.gaussian import Gaussian
from evaluation.baselines.otsu import Otsu
from evaluation.baselines.grabcut import GrabCutAutoBrush
from fatesam_api.model.modal_scribe_sam import ModalScribeSAM
from fatesam_api.model.scribe_sam import ScribeSAM
from scribe.base import predict
from evaluation.utils.tuning import set_all_tuned_hyperparameters


output_folder = Path("data/results/scribed")
#support_images, support_labels, _ = get_training_data(seed=42)

MODELS = [
    CannyFill(),
    Gaussian(),
    Otsu(),
    GrabCutAutoBrush(),
    BestMobileSAMv2Implementation(),
    ModalScribeSAM()
]

def perform_comparison(raw_images, labels, MODELS):
    set_all_tuned_hyperparameters(MODELS)
    for img, label in zip(raw_images, labels):
        img_name = f"{label}.jpg"
        print(f"\nSegmenting {img_name}...")

        for baseline in MODELS:
            prediction = predict(baseline, img)
            image = prediction.to_image()
            output_path = output_folder / f'{img_name[:-4]}-{baseline.name}.jpg'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            print(f"Done for {baseline.name}!")

def run_full_comparison(evaluation_data, MODELS):
    for difficulty in DIFFICULTIES:
        if difficulty not in evaluation_data:
            continue

        dataset = evaluation_data[difficulty]
        print(f"\nPerforming comparison on {difficulty} images...")
        perform_comparison(dataset["images"], dataset["labels"], MODELS)

output_folder.mkdir(parents=True, exist_ok=True)

def run_default_comparison():
    evaluation_data = get_test_data_by_difficulty(seed=42)
    run_full_comparison(evaluation_data=evaluation_data, MODELS=MODELS)
