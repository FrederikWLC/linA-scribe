import cv2
import numpy as np
from pathlib import Path
from utils.image_io import save_rgb

import numpy as np
from pathlib import Path
from utils.image_io import save_rgb

class Image(np.ndarray):

    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def save(self, out_path: str | Path):
        out_path = Path(out_path)
        save_rgb(self, out_path)
        return out_path

    def __repr__(self):
        return f"Image(shape={self.shape}, dtype={self.dtype})"
