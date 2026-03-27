import cv2
from optuna import Trial
from config import config
from utils.auto_prompts import auto_points
from model.scribe import SeedableScribe, Tunable
import numpy as np
import torch 
from utils.seeds import PointSeed, Seed, get_boxes, get_points_and_labels
from utils.binary_mask import BinaryMask
from model.baselines.gaussian import Gaussian
# the SAM implementation class

class SAM(SeedableScribe):
    def __init__(self, sam_backend: str = config.SAM_BACKEND, sam_model_type: str = config.SAM_MODEL_TYPE, sam_checkpoint_path: str = config.SAM_CHECKPOINT_PATH):
        
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

    def __init__(self):
        super().__init__(sam_backend=config.MOBILE_SAM_BACKEND, sam_model_type=config.MOBILESAM_MODEL_TYPE, sam_checkpoint_path=config.MOBILESAM_CHECKPOINT_PATH)

    
    @property
    def name(self):
        return "mSAM"

""""
class MobileSAMv2AutoBox(MobileSAMv2):
    
    def autoseed(self, image: np.ndarray) -> list[BoxSeed]:
        thresh = Gaussian().predict(image)    
        box = auto_box(thresh) 
        return [box]
        
    @property
    def name(self):
        return "mSAM+box"
"""
    
class MobileSAMv2AutoPoint(MobileSAMv2,Tunable):

    def __init__(self, d_bilateral=15, sigma=75, C=5, d_gaussian=19, n_fgd_points=1000, n_bgd_points=1000, d_fgd_erosion=3, d_bgd_erosion=3):
        super().__init__()
        self.d_bilateral = d_bilateral
        self.sigma = sigma
        self.C = C
        self.d_gaussian = d_gaussian
        self.n_fgd_points = n_fgd_points
        self.n_bgd_points = n_bgd_points
        self.d_fgd_erosion = d_fgd_erosion
        self.d_bgd_erosion = d_bgd_erosion

    def autoseed(self, image: np.ndarray) -> list[PointSeed]:
        
        d_bilateral = self.d_bilateral
        sigma = self.sigma
        C = self.C
        d_gaussian = self.d_gaussian
        n_fgd_points = self.n_fgd_points
        n_bgd_points = self.n_bgd_points
        d_fgd_erosion = self.d_fgd_erosion
        d_bgd_erosion = self.d_bgd_erosion

        thresh = Gaussian(C,d_gaussian,d_bilateral,sigma).predict(image)
        points = auto_points(thresh,n_fgd_points,n_bgd_points,d_fgd_erosion,d_bgd_erosion)
        return points
        
    @property
    def name(self):
        return "mSAM+pts"
    
    @property
    def hyperparameters(self) -> dict:
        return {
            # General bilateral filter hyperparameters for Gaussian preprocessing
            "d_bilateral":self.d_bilateral,
            "sigma":self.sigma,
            # General Gaussian hyperparameters
            "C":self.C,
            # Specific Gaussian hyperparameters for probable foreground and sure foreground (used for autoseeding of brushes)
            "d_gaussian":self.d_gaussian,
            # Seed generation hyperparameters
            "n_fgd_points":self.n_fgd_points,
            "n_bgd_points":self.n_bgd_points,
            "d_fgd_erosion":self.d_fgd_erosion,
            "d_bgd_erosion":self.d_bgd_erosion
        }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return {
            "d_bilateral":trial.suggest_categorical("d_bilateral", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
            "sigma":trial.suggest_int("sigma", 0, 150),
            "C":trial.suggest_int("C", 0, 10),
            "d_gaussian":trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,16)]), # odd integers from 3 to 31
            "n_fgd_points":trial.suggest_int("n_fgd_points", 1, 2000),
            "n_bgd_points":trial.suggest_int("n_bgd_points", 1, 2000),
            "d_fgd_erosion":trial.suggest_categorical("d_fgd_erosion", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
            "d_bgd_erosion":trial.suggest_categorical("d_bgd_erosion", [i * 2 + 1 for i in range(1,11)]) # odd integers from 3 to 21
        }