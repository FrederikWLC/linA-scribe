import cv2
import numpy as np

def auto_boxes(image: np.ndarray, min_area=250, min_boxes=10, max_boxes=14):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,  # size of block
        -10  # tweak this between -5 and -20
    )

    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    boxes = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
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

    return np.array(boxes)
