import sys
from pathlib import Path
import cv2
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from evaluation.utils.metrics import compute_metrics
from scribe.binary_mask import BinaryMask
WD=Path(__file__).resolve().parent
GT_ROOT=ROOT/"data"/"ground_truth"/"registered2"

def load_tool_output(p):
    image = cv2.imread(p.as_posix(), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3 and image.shape[2] == 4:
        return BinaryMask(image[:, :, 3] > 0)
    return BinaryMask.from_image(cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE))

def load_ground_truth(p):
    return BinaryMask.from_image(cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE))

find_gt=lambda p:GT_ROOT/f"{p.stem.split()[0]}.png"
tool_paths=sorted((WD/"tool output").glob("*.png"))
dice_scores={p.stem:score for p,score in zip(tool_paths, compute_metrics([load_tool_output(p) for p in tool_paths],[load_ground_truth(find_gt(p)) for p in tool_paths]).get("dice",[]))}
CSV_PATH=WD/"dice_scores.csv"
if __name__=="__main__":
    pd.DataFrame(list(dice_scores.items()), columns=["tool_output", "dice"]).to_csv(CSV_PATH, index=False)
    print(f"wrote {CSV_PATH}")
