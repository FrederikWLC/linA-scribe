import cv2
from config import config
from utils.auto_prompts import auto_boxes
from model.scribe import SeedableScribe, Seed
import numpy as np
import torch 
from utils.seeds import Seed, BoxSeed, get_boxes, get_points_and_labels
from utils.binary_mask import BinaryMask
# the SAM implementation class
class Sam(SeedableScribe):

    def __init__(self):
        
        if config.SAM_BACKEND == "mobile":
            from mobile_sam import sam_model_registry, SamPredictor
        else:
            from segment_anything import sam_model_registry, SamPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[config.SAM_MODEL_TYPE](
            checkpoint=str(config.SAM_CHECKPOINT_PATH)
        )
        sam.to(device)
        self.predictor = SamPredictor(sam)

    # generate the masks with seeds
    def generate_masks(
        self,
        image: np.ndarray,
        points: list[list[int]] = None,
        labels: list[int] = None,
        boxes: list[int] | tuple[int, int, int, int] = None,
    ) -> list[BinaryMask]:
        
        self.predictor.set_image(image)

        masks = []
        if boxes:
            for box in boxes:
                mask, score, logit = self.predictor.predict(
                    box = box,
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
                masks.append(np.squeeze(mask))
        else:
            mask, score, logit = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
            masks.append(BinaryMask(np.squeeze(mask)))
        return masks
    
    def segment(self, image: np.ndarray, seeds: list[Seed] | None = None) -> BinaryMask:
        points, labels = get_points_and_labels(seeds) if seeds else (None, None)
        boxes = get_boxes(seeds) if seeds else []

        # convert to rgb for sam
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        masks = self.generate_masks(image,points,labels,boxes)
        if len(masks) > 1: # combine masks
            # return intersection of all masks
            mask = BinaryMask.from_intersection(*masks)
        else:
            mask = BinaryMask(masks[0])
        return mask
    
    @property
    def name(self):
        return "mSAM"

class SamAutoBox(Sam):
    def autoseed(self, image: np.ndarray) -> list[BoxSeed]:       
        seeds = [BoxSeed(x1, y1, x2, y2) for [[x1, y1], [x2, y2]] in auto_boxes(image)]
        return seeds
        
    @property
    def name(self):
        return "mSAMautobox"