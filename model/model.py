from config import config
from pathlib import Path
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT_PATH)
mask_generator = SamAutomaticMaskGenerator(sam)

class Scribe:
    @staticmethod
    def generate_mask(image_path: str | Path) -> list[dict]:
        path = Path(image_path)
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        return masks