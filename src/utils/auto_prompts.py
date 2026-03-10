import cv2
import numpy as np
from model.baselines.gaussian import Gaussian

def show_boxes(image, boxes):
    vis = image.copy()

    # ensure color image so the rectangles are visible
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for (x1, y1), (x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("boxes", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def auto_boxes(image: np.ndarray, min_area=10000, max_area=250*250, min_boxes=10, max_boxes=20):

    gthresh = Gaussian().scribe(image)
    gthresh = cv2.bitwise_not(gthresh)
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(gthresh,kernel,iterations = 2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=2)
    image = closing

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #print("contours found:", len(contours))

    boxes = []

    for c in contours:        

        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if (area < min_area) or (area > max_area):
            continue
        boxes.append([[x, y], [x + w, y + h]])

        if len(boxes) >= max_boxes:
            break

    if len(boxes) < min_boxes:
        for c in contours:
            if len(boxes) >= min_boxes:
                break
            # Skip ones already included
            x, y, w, h = cv2.boundingRect(c)
            candidate = [[x, y], [x + w, y + h]]
            if candidate not in boxes:
                boxes.append(candidate)
    #show_boxes(image,boxes)
    return np.array(boxes)

def auto_brushes(image):
    gthresh = Gaussian().scribe(image)
    gthresh_not = cv2.bitwise_not(gthresh)
    #return extract_positive_points(gthresh),extract_negative_points(gthresh)
    fgd_brush = cv2.morphologyEx(gthresh_not, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=4)
    bgd_brush = cv2.morphologyEx(gthresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=3)
    
    ys, xs = fgd_brush.nonzero()
    fgd_pixels = np.column_stack((xs, ys))

    ys, xs = bgd_brush.nonzero()
    bgd_pixels = np.column_stack((xs, ys))

    return fgd_pixels, bgd_pixels


def extract_positive_points(mask,min_blob_area=50):
    num_labels, blob_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points = []
    for label in range(1, num_labels):  # skip background blobs
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_blob_area:
            continue
        blob = np.uint8(blob_labels == label) * 255
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        _, _, _, most_interior = cv2.minMaxLoc(dist)
        points.append(most_interior)
    return points

def extract_negative_points(mask,num_points=3,min_distance=20):
    background = cv2.bitwise_not(mask)
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    points = []
    dist_work = dist.copy()
    for _ in range(num_points):
        _, max_val, _, most_exterior = cv2.minMaxLoc(dist_work)
        if max_val <= 0:
            break

        points.append(most_exterior)

        # remove neighborhood around found point
        # for the next point not to be too nearby
        x,y = most_exterior
        cv2.circle(dist_work,(x,y),min_distance,0,-1)
    return points

