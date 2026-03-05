import cv2
from model.baselines import *

baselines = [
    CannyFill(),
    Gaussian(),
    Otsu(),
    Watershed(),
    GrabCut()]

raw_images = ["HT7a.jpg","HT7b.jpg"]

for img_name in raw_images:
    # Read the image.
    img = cv2.imread(f'data/raw/{img_name}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for baseline in baselines:
        binary = baseline.scribe(img)
        cv2.imwrite(f'data/{img_name[:-4]}-{baseline.__class__.__name__}.jpg', binary)