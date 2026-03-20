import cv2
import numpy as np
from utils.seeds import BoxSeed, BrushSeed, PointSeed

def auto_boxes(thresh: np.ndarray, min_area=25*25, max_area=200*200, max_boxes=20) -> list[BoxSeed]:

    kernel_erosion = np.ones((1,1), np.uint8)
    erosion = cv2.erode(thresh, kernel_erosion, iterations = 2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,2))
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_open)
    kernel_dilate = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(opening,kernel_dilate,iterations = 4)
    image = dilation

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #print("contours found:", len(contours))

    boxes = []

    for c in contours:        

        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if (area >= min_area) and (area <= max_area):
            boxes.append([[x, y], [x + w, y + h]])

        if len(boxes) >= max_boxes:
            break

    #show_boxes(image,boxes)
    return [BoxSeed(x1, y1, x2, y2) for [[x1, y1], [x2, y2]] in boxes]

def auto_box(thresh: np.ndarray, min_area=25*25, max_area=200*200, max_boxes=20) -> BoxSeed:
    boxes = auto_boxes(thresh, min_area, max_area, max_boxes)
    return BoxSeed.bigger_from_boxes(boxes)

def auto_brushes(thresh,erosion_iter=2) -> list[BrushSeed]:
    #return extract_positive_points(gthresh),extract_negative_points(gthresh)
    fgd_brush = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=erosion_iter)
    bgd_brush = cv2.morphologyEx(thresh.invert(), cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=erosion_iter)
    #prb_fgd_brush = cv2.morphologyEx(gthresh, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations=erosion_iter)
    
    ys, xs = fgd_brush.nonzero()
    fgd_pixels = np.column_stack((xs, ys))

    ys, xs = bgd_brush.nonzero()
    bgd_pixels = np.column_stack((xs, ys))

    #ys, xs = prb_fgd_brush.nonzero()
    #prb_fgd_pixels = np.column_stack((xs, ys))
    
    return [
            BrushSeed(fgd_pixels,cv2.GC_FGD),
            BrushSeed(bgd_pixels,cv2.GC_BGD)
            ]

def auto_points(thresh, num_points=10,erosion_iter=2) -> list[PointSeed]:
    fgd_brush = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=erosion_iter)
    bgd_brush = cv2.morphologyEx(thresh.invert(), cv2.MORPH_ERODE, np.ones((3,3), np.uint8), iterations=erosion_iter)
    fgd_points = extract_innermost_points(fgd_brush,num_points=num_points)
    bgd_points = extract_innermost_points(bgd_brush,num_points=num_points)
    return [
        PointSeed(x, y, 1) for x, y in fgd_points
    ] + [
        PointSeed(x, y, 0) for x, y in bgd_points
    ]

def extract_innermost_points(mask,num_points=9) -> list[tuple[int, int]]:
    w, h = mask.shape
    min_distance = round(np.sqrt(w * h)/num_points)
    border_margin = 1
    # add zero border so image edges are also "bad"
    padded = cv2.copyMakeBorder(
        mask,
        top=border_margin,
        bottom=border_margin,
        left=border_margin,
        right=border_margin,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 0).astype(np.uint8)

    # remove padding again so coordinates match original image
    dist = dist[
        border_margin:border_margin + mask.shape[0],
        border_margin:border_margin + mask.shape[1]
    ]
    points = []
    dist_work = dist.copy()
    for _ in range(num_points):
        _, max_val, _, most_exterior = cv2.minMaxLoc(dist_work)
        if max_val <= 0:
            break

        points.append(most_exterior)
        x,y = most_exterior
        # remove neighborhood around found point
        # for the next point not to be too nearby
        cv2.circle(dist_work, center=(x, y), radius=min_distance, color=0, thickness=-1)
        # for visualization of the distance transform and the found points:
        #dist_vis = cv2.normalize(dist_work, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #cv2.imshow("dist", dist_vis)
        #cv2.waitKey(0)

    return points