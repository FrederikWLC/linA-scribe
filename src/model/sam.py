import cv2
from config import config
from utils.auto_prompts import auto_box, auto_boxes, auto_points
from model.scribe import SeedableScribe
import numpy as np
import torch 
from utils.seeds import PointSeed, Seed, BoxSeed, get_boxes, get_points_and_labels
from utils.binary_mask import BinaryMask
from model.baselines.gaussian import Gaussian
# the SAM implementation class

class SAM(SeedableScribe):
    def __init__(self,display_seeds: bool = False, sam_backend: str = config.SAM_BACKEND, sam_model_type: str = config.SAM_MODEL_TYPE, sam_checkpoint_path: str = config.SAM_CHECKPOINT_PATH):
        super().__init__(display_seeds)
        
        if sam_backend == "mobile":
            from mobile_sam import sam_model_registry, SamPredictor
        else:
            from segment_anything import sam_model_registry, SamPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[sam_model_type](
            checkpoint=str(sam_checkpoint_path)
        )
        sam.to(device)
        self.predictor = SamPredictor(sam)

    # generate the masks with seeds
    def generate_masks(
        self,
        image: np.ndarray,
        points: list[list[int]] = None,
        labels: list[int] = None,
        boxes: list[int] | None = None,
    ) -> list[BinaryMask]:
        
        self.predictor.set_image(image)

        masks = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                mask, score, logit = self.predictor.predict(
                    box = box,
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
                masks.append(BinaryMask(np.squeeze(mask)))
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
        return "mSAMv2"

class MobileSAMv2(SAM):

    def __init__(self, display_seeds: bool = False):
        super().__init__(display_seeds, sam_backend=config.MOBILE_SAM_BACKEND, sam_model_type=config.MOBILESAM_MODEL_TYPE, sam_checkpoint_path=config.MOBILESAM_CHECKPOINT_PATH)

    
    @property
    def name(self):
        return "mSAM"

class MobileSAMv2AutoBox(MobileSAMv2):
    
    def autoseed(self, image: np.ndarray) -> list[BoxSeed]:
        thresh = Gaussian().predict(image)    
        box = auto_box(thresh) 
        return [box]
        
    @property
    def name(self):
        return "mSAM+box"
    
class MobileSAMv2AutoPoint(MobileSAMv2):
    
    def autoseed(self, image: np.ndarray) -> list[PointSeed]:
        thresh = Gaussian().predict(image)
        box = auto_box(thresh) 
        points = auto_points(thresh,num_fgd_points=1000,num_bgd_points=1000,erosion_iter=1)
        return points
        
    @property
    def name(self):
        return "mSAM+pts"