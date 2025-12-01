from config import config
from pathlib import Path
from utils import Mask, Image
from utils.auto_prompts import auto_boxes
import numpy as np
import torch 
import cv2

if config.SAM_BACKEND == "mobile":
    from mobile_sam import sam_model_registry, SamPredictor
else:
    from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"


class Scribe:

    def __init__(self):
        sam = sam_model_registry[config.SAM_MODEL_TYPE](
            checkpoint=str(config.SAM_CHECKPOINT_PATH)
        )
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def generate_masks(
        self,
        image: Image | np.ndarray,
        box: list[int] | tuple[int, int, int, int] = None,
        points: list[list[int]] = None,
        labels: list[int] = None
    ) -> list[dict]:

        image = Image(image)
        
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            box=box,
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )

        return [
            Mask(
                mask=m,
                score=float(s),
                logit=l,
                source_image=image
            )
            for m, s, l in zip(masks, scores, logits)
        ]

    def scribe(self, image: Image | np.ndarray):
        boxes = auto_boxes(image)
        all_masks = []
        for box in boxes:
            all_masks.append(self.generate_masks(image,box))
        return np.concatenate(all_masks).tolist()