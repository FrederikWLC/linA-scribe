import cv2
from pathlib import Path
import numpy as np

def load_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_rgb(image: np.ndarray, path: str | Path):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return path