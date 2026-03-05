import cv2
import numpy as np
from model.scribe import Scribe

class GrabCut(Scribe):
    def __init__(self, iters=5, border=15, fg_pct=10, bg_pct=10):
        self.iters = iters
        self.border = border
        self.fg_pct = fg_pct
        self.bg_pct = bg_pct

    def scribe(self, image):
        
        gray = image
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        h, w = gray.shape

        # 0/1/2/3 mask for grabCut
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)  # probable background by default

        # sure background: image border
        b = self.border
        mask[:b, :] = cv2.GC_BGD
        mask[-b:, :] = cv2.GC_BGD
        mask[:, :b] = cv2.GC_BGD
        mask[:, -b:] = cv2.GC_BGD

        # intensity-based seeds
        fg_thr = np.percentile(gray, self.fg_pct)           # darkest -> likely ink
        bg_thr = np.percentile(gray, 100 - self.bg_pct)     # brightest -> likely stone

        mask[gray <= fg_thr] = cv2.GC_PR_FGD
        mask[gray >= bg_thr] = cv2.GC_BGD  # sure background helps a lot

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(bgr, mask, None, bgdModel, fgdModel, self.iters, cv2.GC_INIT_WITH_MASK)

        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return cv2.bitwise_not(out)