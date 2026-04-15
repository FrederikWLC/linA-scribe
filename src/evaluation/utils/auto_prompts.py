import cv2
import numpy as np
from scribe.prompts import PointPrompt


# returns foreground pixels and background pixels as mask given a thresholded image, used for autoseeding of GrabCutAutoBrush
def auto_brush(thresh,d_prb_erosion=3) -> np.ndarray:
    prb_erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(d_prb_erosion, d_prb_erosion))

    # Erosion to separate sure foreground and sure background (gap is filled with probable background)
    sure_bgd_brush = cv2.morphologyEx(thresh.invert(), cv2.MORPH_ERODE, prb_erosion_kernel)
    sure_fgd_brush = thresh

    # sure bgd => then sure fgd on top; rest is probable bgd
    mask = np.full(thresh.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    mask[sure_bgd_brush > 0] = cv2.GC_BGD
    mask[sure_fgd_brush > 0] = cv2.GC_FGD
    
    return mask

# places points randomly within the mask foreground, used for autoseeding of point-based models
# can be used for background too by inverting the supplied mask
def place_points_randomly(mask,num_points) -> list[tuple[int, int]]:
    ys, xs = mask.nonzero()
    fgd_pixels = np.column_stack((xs, ys))
    np.random.seed(42)  # for reproducibility
    indices = np.random.permutation(len(fgd_pixels))  # we randomize order of pixels
    return fgd_pixels[indices[:num_points]]  # we take first num_points

# returns list of point seeds given a thresholded image, used for autoseeding of MobileSAMv2AutoPoint
def auto_points(thresh, num_fgd_points=20, num_bgd_points=20, d_bgd_erosion=3, point_placement_func=place_points_randomly) -> list[PointPrompt]:
    bgd_erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(d_bgd_erosion, d_bgd_erosion))

    fgd_thresh, bgd_thresh = thresh, thresh.invert()
    fgd_brush,bgd_brush = fgd_thresh, cv2.morphologyEx(bgd_thresh, cv2.MORPH_ERODE, bgd_erosion_kernel)
    
    fgd_points = point_placement_func(fgd_brush,num_points=num_fgd_points)
    bgd_points = point_placement_func(bgd_brush,num_points=num_bgd_points)
    return [
        PointPrompt(x, y, 1) for x, y in fgd_points
    ] + [
        PointPrompt(x, y, 0) for x, y in bgd_points
    ]
