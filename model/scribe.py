from config import config
from pathlib import Path
import torch 
import cv2

if config.SAM_BACKEND == "mobile":
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
else:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[config.SAM_MODEL_TYPE](
    checkpoint=str(config.SAM_CHECKPOINT_PATH)
)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

class Scribe:
    @staticmethod
    def generate_mask(image_path: str | Path) -> list[dict]:
        path = Path(image_path)
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        return masks