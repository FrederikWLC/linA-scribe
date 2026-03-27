import cv2
import numpy as np
from utils.seeds import BoxSeed, BrushSeed, PointSeed

# returns a box seed given a thresholded image, used for autoseeding of MobileSAMv2AutoBox
def auto_box(thresh: np.ndarray) -> BoxSeed:
    # creates a box seed
    h,w = thresh.shape
    offsetx = w//10
    offsety = h//10
    
    ys, xs = np.where(thresh > 0)

    x1 = xs.min() + offsetx
    x2 = xs.max() - offsetx
    y1 = ys.min() + offsety
    y2 = ys.max() - offsety

    return BoxSeed(x1, y1, x2, y2)

# returns foreground pixels and background pixels as brush seeds given a thresholded image, used for autoseeding of GrabCutAutoBrush
def auto_brushes(prb_fgd_thresh,sure_fgd_thresh,sure_bgd_kernel_size=3,prb_fgd_kernel_size=3,sure_fgd_kernel_size=3) -> list[BrushSeed]:
    sure_bgd_kernel = np.ones((sure_bgd_kernel_size,sure_bgd_kernel_size), np.uint8)
    prb_fgd_kernel = np.ones((prb_fgd_kernel_size,prb_fgd_kernel_size), np.uint8)
    sure_fgd_kernel = np.ones((sure_fgd_kernel_size,sure_fgd_kernel_size), np.uint8)
    
    sure_bgd_brush = cv2.morphologyEx(sure_fgd_thresh.invert(), cv2.MORPH_ERODE, sure_bgd_kernel)
    prb_fgd_brush = cv2.morphologyEx(prb_fgd_thresh, cv2.MORPH_ERODE, prb_fgd_kernel)
    sure_fgd_brush = cv2.morphologyEx(prb_fgd_thresh, cv2.MORPH_ERODE,sure_fgd_kernel)

    ys, xs = sure_bgd_brush.nonzero()
    sure_bgd_pixels = np.column_stack((xs, ys))

    ys, xs = prb_fgd_brush.nonzero()
    prb_fgd_pixels = np.column_stack((xs, ys))

    ys, xs = sure_fgd_brush.nonzero()
    sure_fgd_pixels = np.column_stack((xs, ys))
    
    return [
            BrushSeed(sure_bgd_pixels,cv2.GC_BGD),
            BrushSeed(prb_fgd_pixels,cv2.GC_PR_FGD),
            BrushSeed(sure_fgd_pixels,cv2.GC_FGD)
            ]

# returns list of point seeds given a thresholded image, used for autoseeding of MobileSAMv2AutoPoint
def auto_points(thresh, num_fgd_points=20, num_bgd_points=20, fgd_kernel_size=3, bgd_kernel_size=3) -> list[PointSeed]:
    fgd_kernel = np.ones((fgd_kernel_size,fgd_kernel_size), np.uint8)
    bgd_kernel = np.ones((bgd_kernel_size,bgd_kernel_size), np.uint8)

    fgd_brush = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, fgd_kernel)
    bgd_brush = cv2.morphologyEx(thresh.invert(), cv2.MORPH_ERODE, bgd_kernel)
    fgd_points = place_points(fgd_brush,num_points=num_fgd_points)
    bgd_points = place_points(bgd_brush, num_points=num_bgd_points)
    return [
        PointSeed(x, y, 1) for x, y in fgd_points
    ] + [
        PointSeed(x, y, 0) for x, y in bgd_points
    ]

# places points randomly within the mask foreground, used for autoseeding of point-based models
# can be used for background too by inverting the supplied mask
def place_points(mask,num_points) -> list[tuple[int, int]]:
    ys, xs = mask.nonzero()
    fgd_pixels = np.column_stack((xs, ys))
    np.random.seed(42)  # for reproducibility
    indices = np.random.permutation(len(fgd_pixels))  # we randomize order of pixels
    return fgd_pixels[indices[:num_points]]  # we take first num_points