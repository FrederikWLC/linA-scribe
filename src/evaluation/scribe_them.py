import cv2
from pathlib import Path
from data.split import DIFFICULTIES, get_test_data_by_difficulty
from scribe.baselines.sam import BestMobileSAMv2Implementation
from scribe.baselines.canny_fill import CannyFill
from scribe.baselines.gaussian import Gaussian
from scribe.baselines.otsu import Otsu
from scribe.baselines.grabcut import GrabCutAutoBrush
from gfsam_api.ModalGFSAM import ModalGFSAM
from fatesam2d_api.ModalFATESAM2D import ModalFATESAM2D, ModalFATESAM2DAutoPoint, ModalFATESAM2DBlank
from scribe.base import predict
from evaluation.utils.tuning import set_all_tuned_hyperparameters


output_folder = Path("data/results/scribed")

MODELS = [
    #CannyFill(),
    #Gaussian(),
    #Otsu(),
    #GrabCutAutoBrush(),
    #BestMobileSAMv2Implementation(),
    #ModalGFSAM(),
    #ModalFATESAM2D(),
    #ModalFATESAM2DAutoPoint()
    ModalFATESAM2DBlank()
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
    evaluation_data = get_test_data_by_difficulty()
    run_full_comparison(evaluation_data=evaluation_data, MODELS=MODELS)
