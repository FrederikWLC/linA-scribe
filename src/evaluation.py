import cv2
from model.baselines import *

baselines = [
    CannyFill(),
    Gaussian(),
    Otsu(),
    Watershed(),
    GrabCut()]

documents = ["HT7a.jpg"]

raw_images = [cv2.imread(f'data/raw/{img_name}',cv2.IMREAD_GRAYSCALE) for img_name in documents]
ground_truths = [cv2.imread(f'data/ground_truth/registered/{img_name}', cv2.IMREAD_GRAYSCALE) for img_name in documents]

for baseline in baselines:
    metrics = baseline.evaluate(raw_images, ground_truths, tolerance=5) # Assuming ground_truth is defined
    print(f"{baseline.__class__.__name__} metrics: {metrics}")