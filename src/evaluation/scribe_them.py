import cv2
from pathlib import Path
from data.split import DIFFICULTIES, get_test_data_by_difficulty
from scribe.baselines.canny_fill import build_cannyfill
from scribe.baselines.gaussian import build_gaussian
from scribe.baselines.otsu import build_otsu    
from gfsam_api.ModalGFSAM import build_modal_gfsam
from fatesam2d_api.ModalFATESAM2D import build_default_modal_fatesam2d
from scribe.base import predict
from evaluation.utils.tuning import set_all_tuned_hyperparameters
from sam_api.modal_sam import build_best_modal_sam_variant

output_folder = Path("data/results/scribed")

def get_models_to_be_scribed():
    return [
        build_cannyfill(),
        build_gaussian(),
        build_otsu(),
        build_default_modal_fatesam2d(),
        build_best_modal_sam_variant(),
    ] 

def perform_comparison(models, raw_images, labels):
    set_all_tuned_hyperparameters(models)
    for img, label in zip(raw_images, labels):
        img_name = f"{label}.jpg"
        print(f"\nSegmenting {img_name}...")

        for model in models:
            prediction = predict(model, img)
            image = prediction.to_image()
            output_path = output_folder / f'{img_name[:-4]}-{model.name}.jpg'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            print(f"Done for {model.name}!")

def run_full_comparison(models, evaluation_data):
    for difficulty in DIFFICULTIES:
        if difficulty not in evaluation_data:
            continue

        dataset = evaluation_data[difficulty]
        print(f"\nPerforming comparison on {difficulty} images...")
        perform_comparison(models, dataset["images"], dataset["labels"])

output_folder.mkdir(parents=True, exist_ok=True)

def run_default_comparison():
    models = get_models_to_be_scribed()
    evaluation_data = get_test_data_by_difficulty()
    run_full_comparison(models=models, evaluation_data=evaluation_data)