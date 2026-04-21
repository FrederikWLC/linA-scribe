import cv2
from optuna import Trial
from config import config
from scribe.auto_prompts import auto_points
from scribe.base import PointScribe
from scribe.tunable import BilateralTunable
import numpy as np
import torch 
from scribe.prompts import PointPrompt, get_point_prompts_and_labels
from scribe.binary_mask import BinaryMask
from scribe.baselines.gaussian import Gaussian
# the SAM implementation class

class SAM(PointScribe):
    def __init__(self, sam_backend: str = config.SAM_BACKEND, sam_model_type: str = config.SAM_MODEL_TYPE, sam_checkpoint_path: str = config.SAM_CHECKPOINT_PATH, use_best_of_three: bool = False, **kwargs):
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
        self._use_best_of_three = use_best_of_three
        super().__init__(**kwargs)

    # generate the masks with prompts
    def generate_mask(
        self,
        image: np.ndarray,
        points: list[PointPrompt] = None,
        labels: list[int] = None
    ) -> list[BinaryMask]:
        
        # runs the ViT for the image and saves the image embedding
        self.predictor.set_image(image)
        
        # now, running the mask decoder...
        if self._use_best_of_three: # if this flag is set, all three output tokens are processed, and best predicted iou mask is returned
            masks, scores, _logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True)
            best_mask = masks[np.argmax(scores)]
            return BinaryMask(np.squeeze(best_mask))
        else: # else, only the fourth unambigous output token is processed and returned as mask
            mask, _score, _logit = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
            return BinaryMask(np.squeeze(mask))
    
    def segment(self, image: np.ndarray, prompts: list[PointPrompt] | None = None) -> BinaryMask:
        points, labels = get_point_prompts_and_labels(prompts) if (not prompts is None) else (None, None)
        # convert to rgb for sam
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = self.generate_mask(image,points,labels)
        
        return mask
    
    @property
    def name(self):
        return "SAM"

class MobileSAMv2(SAM, BilateralTunable):
    """Mobile SAM with optional bilateral filtering and best-of-three output."""
    USE_BILATERAL_FILTER = False
    

    def __init__(self, use_bilateral_filter: bool = False, use_best_of_three: bool = False,
                 d_bilateral: int = 15, sigma_bilateral: int = 75):
        self.use_bilateral_filter = bool(use_bilateral_filter)
        super().__init__(sam_backend=config.MOBILE_SAM_BACKEND,
                         sam_model_type=config.MOBILESAM_MODEL_TYPE,
                         sam_checkpoint_path=config.MOBILESAM_CHECKPOINT_PATH,
                         use_best_of_three=use_best_of_three,
                         d_bilateral=d_bilateral,
                         sigma_bilateral=sigma_bilateral)
        self.filter_str = "bilateral-filter" if self.use_bilateral_filter else "no-filter"
        self.output_str = "best-of-three" if self._use_best_of_three else "single"

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.use_bilateral_filter:
            return super().preprocess(image)
        else:
            return image

    @property
    def hyperparameters(self) -> dict:
        if self.use_bilateral_filter:
            return super().hyperparameters
        return {}

    @classmethod
    def hyperparameter_ranges(cls, trial: Trial) -> dict:
        if cls.USE_BILATERAL_FILTER:
            return super().hyperparameter_ranges(trial)
        return {}

    @property
    def name(self):
        return f"mSAM-{self.filter_str}-{self.output_str}"
    
    @property
    def short_name(self):
        return f"{self.filter_str[0]}-{self.output_str[0]}"


class MobileSAMv2AutoPoint(MobileSAMv2):
    """Mobile SAM with automatic point seeding, optional bilateral filtering, and best-of-three output."""
    USE_BILATERAL_FILTER = True
    

    def __init__(self, use_bilateral_filter=True, use_best_of_three=False,
                 d_bilateral=15, sigma_bilateral=75, C=5, d_gaussian=19, 
                 n_fgd_points=1000, n_bgd_points=1000, d_gap_erosion=3):


        super().__init__(use_bilateral_filter=use_bilateral_filter,
                         use_best_of_three=use_best_of_three,
                         d_bilateral=d_bilateral,
                         sigma_bilateral=sigma_bilateral)
        
        self.C = int(C)
        self.d_gaussian = int(d_gaussian)
        self.n_fgd_points = int(n_fgd_points)
        self.n_bgd_points = int(n_bgd_points)
        self.d_gap_erosion = int(d_gap_erosion)

    def autoprompt(self, image: np.ndarray) -> list[PointPrompt]:
        d_bilateral = int(self.d_bilateral)
        sigma_bilateral = int(self.sigma_bilateral)
        C = int(self.C)
        d_gaussian = int(self.d_gaussian)
        n_fgd_points = int(self.n_fgd_points)
        n_bgd_points = int(self.n_bgd_points)
        d_gap_erosion = int(self.d_gap_erosion)

        thresh = Gaussian(C, d_gaussian, d_bilateral, sigma_bilateral=sigma_bilateral).predict(image)
        points = auto_points(thresh, n_fgd_points, n_bgd_points, d_gap_erosion)
        return points

    @property
    def hyperparameters(self) -> dict:
        return super().hyperparameters | {
            "C": int(self.C),
            "d_gaussian": int(self.d_gaussian),
            "n_fgd_points": int(self.n_fgd_points),
            "n_bgd_points": int(self.n_bgd_points),
            "d_gap_erosion": int(self.d_gap_erosion),
        }

    @classmethod
    def hyperparameter_ranges(cls, trial: Trial) -> dict:
        return super().hyperparameter_ranges(trial) | {
            "C": trial.suggest_int("C", 0, 10),
            "d_gaussian": trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1, 16)]),
            "n_fgd_points": trial.suggest_int("n_fgd_points", 1, 2000),
            "n_bgd_points": trial.suggest_int("n_bgd_points", 1, 2000),
            "d_gap_erosion": trial.suggest_categorical("d_gap_erosion", [i * 2 + 1 for i in range(1, 11)]),
        }

    @property
    def name(self):
        return f"mSAM+pts-{self.filter_str}-{self.output_str}"
    
    @property
    def short_name(self):
        return f"pts-{self.filter_str[0]}-{self.output_str[0]}"


# ============================================================================
# The 8 Different Mobile SAM variants used
# Combinations of:
# - Bilateral filter vs no filter
# - Best-of-three vs single output token
# - Autoseeding vs no autoseeding (i.e. manual point prompts)
# ============================================================================

class MobileSAMv2NoFilter(MobileSAMv2):
    """Legacy: Mobile SAM without filtering."""
    USE_BILATERAL_FILTER = False

    def __init__(self):
        super().__init__(use_bilateral_filter=False, use_best_of_three=False)

class MobileSAMv2BilateralFilter(MobileSAMv2):
    """Legacy: Mobile SAM with bilateral filter."""
    USE_BILATERAL_FILTER = True

    def __init__(self, d_bilateral=15, sigma_bilateral=75):
        super().__init__(use_bilateral_filter=True, use_best_of_three=False,
                         d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral)

class MobileSAMv2BilateralFilterBestOfThree(MobileSAMv2):
    """Legacy: Mobile SAM with bilateral filter and best-of-three output."""
    USE_BILATERAL_FILTER = True

    def __init__(self, d_bilateral=15, sigma_bilateral=75):
        super().__init__(use_bilateral_filter=True, use_best_of_three=True,
                         d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral)

class MobileSAMv2NoFilterBestOfThree(MobileSAMv2):
    """Legacy: Mobile SAM without filter but best-of-three output."""
    USE_BILATERAL_FILTER = False

    def __init__(self):
        super().__init__(use_bilateral_filter=False, use_best_of_three=True)

class MobileSAMv2AutoPointBilateralFilter(MobileSAMv2AutoPoint):
    """Legacy: Mobile SAM with autoseeding and bilateral filter."""
    def __init__(self, d_bilateral=15, sigma_bilateral=75, C=5, d_gaussian=19, 
                 n_fgd_points=1000, n_bgd_points=1000, d_gap_erosion=3):
        super().__init__(use_bilateral_filter=True, use_best_of_three=False,
                         d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral,
                         C=C, d_gaussian=d_gaussian, n_fgd_points=n_fgd_points,
                         n_bgd_points=n_bgd_points, d_gap_erosion=d_gap_erosion)

class MobileSAMv2AutoPointBilateralFilterBestOfThree(MobileSAMv2AutoPoint):
    """Legacy: Mobile SAM with autoseeding, bilateral filter, and best-of-three."""
    def __init__(self, d_bilateral=15, sigma_bilateral=75, C=5, d_gaussian=19,
                 n_fgd_points=1000, n_bgd_points=1000, d_gap_erosion=3):
        super().__init__(use_bilateral_filter=True, use_best_of_three=True,
                         d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral,
                         C=C, d_gaussian=d_gaussian, n_fgd_points=n_fgd_points,
                         n_bgd_points=n_bgd_points, d_gap_erosion=d_gap_erosion)

class MobileSAMv2AutoPointNoFilter(MobileSAMv2AutoPoint):
    """Legacy: Mobile SAM with autoseeding but no filter."""
    USE_BILATERAL_FILTER = False

    def __init__(self, C=5, d_gaussian=19, n_fgd_points=1000, n_bgd_points=1000, 
                 d_gap_erosion=3):
        super().__init__(use_bilateral_filter=False, use_best_of_three=False,
                         C=C, d_gaussian=d_gaussian, n_fgd_points=n_fgd_points,
                         n_bgd_points=n_bgd_points, d_gap_erosion=d_gap_erosion)

class MobileSAMv2AutoPointNoFilterBestOfThree(MobileSAMv2AutoPoint):
    """Legacy: Mobile SAM with autoseeding, no filter, and best-of-three."""
    USE_BILATERAL_FILTER = False

    def __init__(self, C=5, d_gaussian=19, n_fgd_points=1000, n_bgd_points=1000,
                 d_gap_erosion=3):
        super().__init__(use_bilateral_filter=False, use_best_of_three=True,
                         C=C, d_gaussian=d_gaussian, n_fgd_points=n_fgd_points,
                         n_bgd_points=n_bgd_points, d_gap_erosion=d_gap_erosion)
        
# ============================================================================
# The Best Mobile SAM variant used in the main evaluation (MobileSAMv2AutoPointBilateralFilter)
# ============================================================================

class BestMobileSAMv2Implementation(MobileSAMv2AutoPointBilateralFilter):
    
    @property
    def short_name(self):
        return f"mSAM+pts"

