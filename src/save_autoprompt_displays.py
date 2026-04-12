import cv2
from data.split import DIFFICULTIES, get_evaluation_data
from model.baselines.sam import MobileSAMv2AutoPoint
from model.baselines.grabcut import GrabCutAutoBrush
from utils.tuning import set_all_tuned_hyperparameters
from pathlib import Path

baselines = [
    GrabCutAutoBrush(),
    MobileSAMv2AutoPoint()
    ]

output_folder = Path("data/autoprompt_displays")

def perform_comparison(images, labels, baselines):
    set_all_tuned_hyperparameters(baselines)

    for img, label in zip(images, labels):
        img_name = f"{label}.jpg"
        print(f"Autoprompting {img_name}...")
        if img is None:
            continue

        for baseline in baselines:
            prompt = baseline.autoprompt(img)
            image = baseline.draw_prompt(img, prompt) if prompt is not None else img
            output_path = output_folder / f'{img_name[:-4]}-{baseline.name}.jpg'
            cv2.imwrite(str(output_path), image)
            print(f"Done for {baseline.name}!")

if __name__ == "__main__":
    output_folder.mkdir(parents=True, exist_ok=True)
    evaluation_data = get_evaluation_data(seed=42)

    for difficulty in DIFFICULTIES:
        if difficulty not in evaluation_data:
            continue
        print(f"Performing comparison on {difficulty} images...")
        split = evaluation_data[difficulty]
        perform_comparison(split["images"], split["labels"], baselines)