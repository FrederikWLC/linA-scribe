import cv2
import numpy as np
from model.scribe import SeedableScribe
from utils.auto_prompts import auto_boxes, auto_brushes
from utils.binary_mask import BinaryMask
from utils.seeds import Seed, BoxSeed, BrushSeed, get_boxseeds, get_brushseeds
from model.baselines.gaussian import Gaussian

# implementation of GrabCut
class GrabCut(SeedableScribe):
    def __init__(self, display_seeds: bool = False, iters=10):
        super().__init__(display_seeds)
        self.iters = iters

    def segment(self, image: np.ndarray, seeds=None) -> BinaryMask:
        if len(image.shape) == 2:  # convert to bgr
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(image.shape[:2],np.uint8)
        boxseeds = get_boxseeds(seeds) if seeds else []
        
        mask[True] = cv2.GC_PR_BGD # default is probable background
        if boxseeds:
            mask[True] = cv2.GC_BGD
            for box in boxseeds:
                mask[box.y1:box.y2, box.x1:box.x2] = cv2.GC_PR_FGD
        brushseeds = get_brushseeds(seeds)
        if brushseeds:
            for brush in brushseeds:
                xs, ys = zip(*brush.pixels)
                mask[ys, xs] = brush.label

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,self.iters,cv2.GC_INIT_WITH_MASK)
        
        # end result is union of sure foreground and probable foreground
        is_fgd = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        return BinaryMask.from_bool(is_fgd)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (3, 3), 0)

# implementation of GrabCut with automatic brushes given as seeds
class GrabCutAutoBrush(GrabCut):
    def autoseed(self, image: np.ndarray) -> list[BrushSeed]:
        thresh = Gaussian().predict(image)       
        brushes = auto_brushes(thresh,2)
        return brushes

    @property
    def name(self):
        return "GC+brush"
