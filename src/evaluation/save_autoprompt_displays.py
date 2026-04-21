import cv2
from data.split import DIFFICULTIES, get_test_data_by_difficulty
from scribe.baselines.sam import MobileSAMv2AutoPoint
from scribe.baselines.grabcut import GrabCutAutoBrush
from evaluation.utils.tuning import set_all_tuned_hyperparameters
from pathlib import Path

output_folder = Path("data/results/autoseed_displays")

baselines = [
    GrabCutAutoBrush(),
    MobileSAMv2AutoPoint()
]

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


def run_autoprompt_export():
    output_folder.mkdir(parents=True, exist_ok=True)
    evaluation_data = get_test_data_by_difficulty(seed=42)

    for difficulty in DIFFICULTIES:
        if difficulty not in evaluation_data:
            continue
        print(f"Performing comparison on {difficulty} images...")
        split = evaluation_data[difficulty]
        perform_comparison(split["images"], split["labels"], baselines)
