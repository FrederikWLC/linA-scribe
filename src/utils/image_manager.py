from pathlib import Path
from utils.image_io import load_rgb

class ImageManager:
    def __init__(self, root):
        self.root = Path(root)
    
    def list(self):
        return [p.stem for p in self.root.glob("*.*")]
    
    def load(self, name: str):
        path = next(self.root.glob(f"{name}.*"))
        return load_rgb(path)
    
    def path(self, name: str):
        return next(self.root.glob(f"{name}.*"))
