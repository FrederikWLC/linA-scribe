import cv2
import numpy as np
from utils.seeds import PointSeed

# returns foreground pixels and background pixels as mask given a thresholded image, used for autoseeding of GrabCutAutoBrush
def auto_brush(thresh,d_prb_erosion=3) -> np.ndarray:
    prb_erosion_kernel = np.ones((d_prb_erosion,d_prb_erosion), np.uint8)

    # Erosion to separate sure foreground and sure background (gap is filled with probable background)
    sure_bgd_brush = cv2.morphologyEx(thresh.invert(), cv2.MORPH_ERODE, prb_erosion_kernel)
    sure_fgd_brush = thresh

    # sure bgd => then sure fgd on top; rest is probable bgd
    mask = np.full(thresh.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    mask[sure_bgd_brush > 0] = cv2.GC_BGD
    mask[sure_fgd_brush > 0] = cv2.GC_FGD
    
    return mask

# returns list of point seeds given a thresholded image, used for autoseeding of MobileSAMv2AutoPoint
def auto_points(thresh, num_fgd_points=20, num_bgd_points=20, d_bgd_erosion=3) -> list[PointSeed]:
    bgd_erosion_kernel = np.ones((d_bgd_erosion,d_bgd_erosion), np.uint8)

    fgd_thresh, bgd_thresh = thresh, thresh.invert()
    fgd_brush,bgd_brush = fgd_thresh, cv2.morphologyEx(bgd_thresh, cv2.MORPH_ERODE, bgd_erosion_kernel)
    
    fgd_points = place_points(fgd_brush,num_points=num_fgd_points)
    bgd_points = place_points(bgd_brush,num_points=num_bgd_points)
    return [
        PointSeed(x, y, 1) for x, y in fgd_points
    ] + [
        PointSeed(x, y, 0) for x, y in bgd_points
    ]

def auto_points_boundary(thresh, num_fgd_points=20, num_boundary_points=20, num_sure_bgd_points=20, d_gap_erosion=3, d_boundary_erosion=3) -> list[PointSeed]:
    gap_erosion_kernel = np.ones((d_gap_erosion,d_gap_erosion), np.uint8)
    boundary_erosion_kernel = np.ones((d_boundary_erosion,d_boundary_erosion), np.uint8)

    # definition of regions:
    # 1. fgd [positive points],
    # 2. gap [no points],
    # 3. boundary [negative points],
    # 4.sure bgd [negative points]
    fgd = thresh
    gapped_bgd = cv2.morphologyEx(thresh.invert(), cv2.MORPH_ERODE, gap_erosion_kernel) # bgd - gap
    sure_bgd = cv2.morphologyEx(gapped_bgd, cv2.MORPH_ERODE, boundary_erosion_kernel) # bgd - gap - boundary
    boundary = cv2.subtract(gapped_bgd, sure_bgd) # bgd - gap - sure bgd
    
    # we place n_fgd_points, n_boundary_points, and n_sure_bgd_points randomly within the respective masks
    fgd_points = place_points(fgd,num_points=num_fgd_points)
    boundary_points = place_points(boundary, num_points=num_boundary_points)
    sure_bgd_points = place_points(sure_bgd, num_points=num_sure_bgd_points)

    return [
        PointSeed(x, y, 1) for x, y in fgd_points
    ] + [
        PointSeed(x, y, 0) for x, y in boundary_points
    ] + [
        PointSeed(x, y, 0) for x, y in sure_bgd_points
    ]


# places points randomly within the mask foreground, used for autoseeding of point-based models
# can be used for background too by inverting the supplied mask
def place_points(mask,num_points) -> list[tuple[int, int]]:
    ys, xs = mask.nonzero()
    fgd_pixels = np.column_stack((xs, ys))
    np.random.seed(42)  # for reproducibility
    indices = np.random.permutation(len(fgd_pixels))  # we randomize order of pixels
    return fgd_pixels[indices[:num_points]]  # we take first num_points