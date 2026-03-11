import cv2
import numpy as np
from model.scribe import SeedableScribe
from utils.auto_prompts import auto_boxes, auto_brushes
from utils.seeds import Seed, BoxSeed, BrushSeed, get_boxseeds, get_brushseeds

class GrabCut(SeedableScribe):
    def __init__(self, iters=5):
        self.iters = iters

    def scribe(self, image: np.ndarray, seeds: list[Seed] = None) -> np.ndarray:
        autoseed = True if seeds is None else False
        return super().scribe(image,seeds,autoseed)

    def segment(self, image, seeds=None):
        if len(image.shape) == 2:  # convert to bgr
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(image.shape[:2],np.uint8)
        boxseeds = get_boxseeds(seeds)
        
        mask[True] = cv2.GC_PR_BGD # default is probable background
        if boxseeds:
            mask[True] = cv2.GC_BGD
            for box in boxseeds:
                mask[box.y1:box.y2, box.x1:box.x2] = cv2.GC_PR_FGD
        brushseeds = get_brushseeds(seeds)
        if brushseeds:
            for brush in brushseeds:
                xs, ys = zip(*brush.pixels)
                if brush.label == 1:
                    if brush.label == 1:
                        mask[ys, xs] = cv2.GC_FGD
                    else:
                        mask[ys, xs] = cv2.GC_BGD

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,self.iters,cv2.GC_INIT_WITH_MASK)
               
        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return cv2.bitwise_not(out)
    
    def preprocess(self, image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    def autoseed(self, image: np.ndarray) -> list[Seed]:       
        return []

class GrabCutAutoBrush(GrabCut):
    def autoseed(self, image: np.ndarray) -> list[BrushSeed]:      
        sure_fgd_pixels, sure_bgd_pixels = auto_brushes(image)
        seeds = [BrushSeed(sure_fgd_pixels,1),BrushSeed(sure_bgd_pixels,0)]
        return seeds

    @property
    def name(self):
        return "GCautobrush"

class GrabCutAutoBox(GrabCut):
    def autoseed(self, image: np.ndarray) -> list[BoxSeed]:        
        seeds = [BoxSeed(x1, y1, x2, y2) for [[x1, y1], [x2, y2]] in auto_boxes(image)]
        return seeds
    
    @property
    def name(self):
        return "GCautobox"