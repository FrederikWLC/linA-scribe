import cv2
from optuna import Trial
from config import config
from utils.auto_prompts import auto_points
from model.scribe import PointScribe, Tunable
import numpy as np
import torch 
from utils.seeds import MaskSeed, PointSeed, get_points_and_labels, get_mask_prompt
from utils.binary_mask import BinaryMask
from model.baselines.gaussian import Gaussian
from model.baselines.grabcut import GrabCutAutoBrush
# the SAM implementation class

class SAM(PointScribe):
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
    def generate_mask(
        self,
        image: np.ndarray,
        points: list[PointSeed] = None,
        labels: list[int] = None,
    ) -> list[BinaryMask]:
        
        # runs the ViT for the image and saves the image embedding
        self.predictor.set_image(image)
        
        # runs the mask decoder
        mask, _score, _logit = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
        return BinaryMask(np.squeeze(mask))
    
    def segment(self, image: np.ndarray, seeds: list[PointSeed] | None = None) -> BinaryMask:
        points, labels = get_points_and_labels(seeds) if (not seeds is None) else (None, None)
        # convert to rgb for sam
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask = self.generate_mask(image,points,labels)
        
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

    def __init__(self, d_bilateral=15, sigma=75, C=5, d_gaussian=19, n_fgd_points=1000, n_bgd_points=1000, d_gap_erosion=3):
        super().__init__()
        self.d_bilateral = d_bilateral
        self.sigma = sigma
        self.C = C
        self.d_gaussian = d_gaussian
        self.n_fgd_points = n_fgd_points
        self.n_bgd_points = n_bgd_points
        self.d_gap_erosion = d_gap_erosion

    def autoseed(self, image: np.ndarray) -> list[PointSeed]:
        
        d_bilateral = int(self.d_bilateral)
        sigma = int(self.sigma)
        C = int(self.C)
        d_gaussian = int(self.d_gaussian)
        n_fgd_points = int(self.n_fgd_points)
        n_bgd_points = int(self.n_bgd_points)
        d_gap_erosion = int(self.d_gap_erosion)

        thresh = Gaussian(C,d_gaussian,d_bilateral,sigma).predict(image)
        points = auto_points(thresh,n_fgd_points,n_bgd_points,d_gap_erosion)
        return points
        
    @property
    def hyperparameters(self) -> dict:
        return {
            # General bilateral filter hyperparameters for Gaussian preprocessing
            "d_bilateral":int(self.d_bilateral),
            "sigma":int(self.sigma),
            # General Gaussian hyperparameters
            "C":int(self.C),
            "d_gaussian":int(self.d_gaussian),
            # Seed generation hyperparameters
            "n_fgd_points":int(self.n_fgd_points),
            "n_bgd_points":int(self.n_bgd_points),
            "d_gap_erosion":int(self.d_gap_erosion),
        }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return {
            "d_bilateral":trial.suggest_categorical("d_bilateral", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
            "sigma":trial.suggest_int("sigma", 0, 150),
            "C":trial.suggest_int("C", 0, 10),
            "d_gaussian":trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,16)]), # odd integers from 3 to 31
            "n_fgd_points":trial.suggest_int("n_fgd_points", 1, 2000),
            "n_bgd_points":trial.suggest_int("n_bgd_points", 1, 2000),
            "d_gap_erosion":trial.suggest_categorical("d_gap_erosion", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
        }
    
    @property
    def name(self):
        return "mSAM+pts"