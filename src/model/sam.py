import cv2

from config import config
from model.baselines.gaussian import Gaussian
from model.scribe import SeedableScribe, Seed
from utils import Mask, Image
from utils.auto_prompts import auto_boxes
import numpy as np
import torch 
from utils.seeds import Seed, BoxSeed, get_boxes, get_points_and_labels

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

    def generate_masks(
        self,
        image: Image | np.ndarray,
        points: list[list[int]] = None,
        labels: list[int] = None,
        boxes: list[int] | tuple[int, int, int, int] = None,
    ) -> list[dict]:

        image = Image(image)
        
        self.predictor.set_image(image)

        masks = []
        for box in boxes:
            mask, score, logit = self.predictor.predict(
                box = box,
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
            masks.append(np.squeeze(mask))
        return masks

    def scribe(self, image: Image | np.ndarray, seeds: list[Seed] = None) -> np.ndarray:
        autoseed = True if seeds is None else False
        return super().scribe(image,seeds,autoseed)

    def segment(self, image: Image | np.ndarray, seeds: list[Seed] = None): 
        points, labels = get_points_and_labels(seeds) if seeds else (None, None)
        boxes = get_boxes(seeds) if seeds else None
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to rgb for sam
        #print("SAM input shape:", image.shape, image.dtype, image.min(), image.max())
        #cv2.imwrite("debug_sam_input.jpg", image)
        masks = self.generate_masks(image,points,labels,boxes)
        combined = (255 - np.maximum.reduce(masks) * 255).astype("uint8")
        return combined
    
    def autoseed(self, image: Image | np.ndarray) -> list[BoxSeed]:        
        return []

    def preprocess(self, image: Image | np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.bilateralFilter(image, 15, 75, 75), (5,5), 0)

class SamAutoBox(Sam):
     def autoseed(self, image: Image | np.ndarray) -> list[BoxSeed]:       
        seeds = [BoxSeed(x1, y1, x2, y2) for [[x1, y1], [x2, y2]] in auto_boxes(image)]
        #print("Autoseeding generated boxes:", len(seeds))
        return seeds