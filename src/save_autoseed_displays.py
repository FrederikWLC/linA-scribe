import cv2
from pathlib import Path
from model.sam import MobileSAMv2AutoBox, MobileSAMv2AutoPoint
from model.baselines.grabcut import GrabCutAutoBrush

baselines = [
    GrabCutAutoBrush(),
    MobileSAMv2AutoBox(),
    MobileSAMv2AutoPoint()
    ]

raw_folder = Path("data/raw")
easy_raw_folder = raw_folder / "easy"
medium_raw_folder = raw_folder / "medium"
hard_raw_folder = raw_folder / "hard"
output_folder = Path("data/autoseed_displays")

easy_raw_image_paths = easy_raw_folder.glob("*.jpg")
medium_raw_image_paths = medium_raw_folder.glob("*.jpg")
hard_raw_image_paths = hard_raw_folder.glob("*.jpg")

def perform_comparison(raw_image_paths, baselines):
    for img_path in raw_image_paths:
        img_name = img_path.name
        print(f"\Autoseeding {img_name}...")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for baseline in baselines:
            autoseeds = baseline.autoseed(img)
            image = baseline.draw_seeds(img, autoseeds) if autoseeds else img
            output_path = output_folder / f'{img_name[:-4]}-{baseline.name}.jpg'
            cv2.imwrite(str(output_path), image)
            print(f"Done for {baseline.name}!")

print("Performing comparison on easy images...")
perform_comparison(easy_raw_image_paths, baselines)
print("\nPerforming comparison on medium images...")
perform_comparison(medium_raw_image_paths, baselines)
print("\nPerforming comparison on hard images...")
perform_comparison(hard_raw_image_paths, baselines)