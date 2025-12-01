from pathlib import Path
from utils.image_manager import ImageManager
from utils.image_io import save_rgb
from model.scribe import Scribe

class AppController:

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)

        self.raw_dir = self.root / "raw"
        self.mask_dir = self.root / "masks"
        self.overlay_dir = self.root / "overlays"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

        self.images = ImageManager(self.raw_dir)
        self.scribe = Scribe()

    def list_images(self):
        return self.images.list()

    def process(self, index: int):

        names = self.list_images()
        name = names[index]

        img = self.images.load(name)

        masks = self.scribe.scribe(img)

        results = []

        for i, mask in enumerate(masks):
            i = "" if len(masks) == 1 else i+1
            mask_path = self.mask_dir / f"{name}_mask{i}.png"
            mask.to_image().save(mask_path)

            overlay_path = self.overlay_dir / f"{name}_overlay{i}.png"
            mask.to_overlay().save(overlay_path)

            results.append({
                "name": name,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "score": mask.score,
            })

        return results
