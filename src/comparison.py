import cv2
from model.baselines import *

baselines = [
    BilateralGaussian(),
    CannyFill(),
    GaussianThreshold(),
    Otsu(),
    UnsupervisedClustering()]

# Read the image.
img = cv2.imread('data/raw/HT7a.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for baseline in baselines:
    binary = baseline.scribe(img)
    cv2.imwrite(f'data/HT7a-{baseline.__class__.__name__}.jpg', binary)