import numpy as np

from scribe.baselines.gaussian import build_gaussian
from scribe.baselines.sam import SAMConfiguration
from scribe.prompts import BoxPrompt, PointPrompt, PointPromptList
from sam_api.modal_sam import ModalSAM


MODEL_OPTIONS: dict[str, dict[str, object]] = {
    "sam": {
        "label": "SAM",
        "import_path": "sam_api.modal_sam",
        "class_name": "ModalSAM",
        "requires_set_image": True,
        "accepts_prompts": True,
    },
    "gaussian": {
        "label": "Gaussian",
        "import_path": "scribe.baselines.gaussian",
        "class_name": "build_gaussian",
        "requires_set_image": False,
        "accepts_prompts": False,
    },
}

# Scribe service class holding classical model and per-user SAM instances, and methods to interact with them.
class ScribeService:
    def __init__(self):
        self.classical_model = build_gaussian()
        self.sam_model_instances = {}
        self.sam_images_by_user: dict[str, np.ndarray] = {}
        self.sam_image_hw_by_user: dict[str, tuple[int, int]] = {}

    def predict_with_classical(self, image: np.ndarray, granularity: int = 0) -> np.ndarray:
        return self.classical_model.predict_with_granularity(image, granularity=granularity)

    def set_image_for_sam(self, username: str, image: np.ndarray) -> None:
        sam_instance = self.get_sam_instance_for_user(username)
        sam_instance.setImage(image)
        self.sam_images_by_user[username] = image.copy()
        self.sam_image_hw_by_user[username] = (int(image.shape[0]), int(image.shape[1]))

    def get_sam_image_hw(self, username: str) -> tuple[int, int] | None:
        return self.sam_image_hw_by_user.get(username)
    
    def predict_with_sam(self, username: str, prompts=None) -> np.ndarray:
        sam_instance = self.get_sam_instance_for_user(username)
        if prompts is None:
            prompts = ([], PointPromptList([]))
        try:
            return sam_instance.decode_mask(prompts=prompts)
        except RuntimeError as exc: # In case the ModalSAM forgets the image for some reason
            image = self.sam_images_by_user.get(username)
            if image is None:
                raise RuntimeError("No image is set. Upload an image before running SAM.") from exc
            sam_instance.setImage(image)
            return sam_instance.decode_mask(prompts=prompts)

    def predict_with_sam_with_raw_prompts(
        self,
        username: str,
        image: np.ndarray,
        xs: list[float],
        ys: list[float],
        labels: list[int],
        x1s: list[float],
        y1s: list[float],
        x2s: list[float],
        y2s: list[float],
        coordinate_space: str = "percent",
    ) -> np.ndarray:
        prompts = build_sam_prompts_from_raw(
            xs=xs,
            ys=ys,
            labels=labels,
            x1s=x1s,
            y1s=y1s,
            x2s=x2s,
            y2s=y2s,
            coordinate_space=coordinate_space,
            image_hw=(image.shape[0], image.shape[1])
        )
        self.set_image_for_sam(username, image)
        return self.predict_with_sam(username, prompts=prompts)
    
    def get_sam_instance_for_user(self, username: str) -> ModalSAM | None:
        model = self.sam_model_instances.get(username)
        return model if model else self.create_sam_instance_for_user(username)

    def create_sam_instance_for_user(self, username: str) -> ModalSAM:
        if username not in self.sam_model_instances:
            configuration = SAMConfiguration("SAM2", use_bilateral_filter=True, use_autopoints=False)
            self.sam_model_instances[username] = ModalSAM(configuration=configuration)
        return self.sam_model_instances[username]

def build_sam_prompts_from_raw(
    xs: list[float],
    ys: list[float],
    labels: list[int],
    x1s: list[float],
    y1s: list[float],
    x2s: list[float],
    y2s: list[float],
    coordinate_space: str = "percent",
    image_hw: tuple[int, int] | None = None,
) -> tuple[list[BoxPrompt], PointPromptList]:
    if coordinate_space not in {"percent", "pixel"}:
        raise ValueError("coordinate_space must be 'percent' or 'pixel'")

    box_prompts: list[BoxPrompt] = []
    if x1s and y1s and x2s and y2s:
        box_prompts = make_box_prompts_from_raw(x1s, y1s, x2s, y2s, coordinate_space, image_hw)

    point_prompts = []
    if xs and ys and labels:
        point_prompts = make_point_prompts_from_raw(xs, ys, labels, coordinate_space, image_hw)

    return box_prompts, PointPromptList(point_prompts)

def make_point_prompts_from_raw(
    xs: list[float],
    ys: list[float],
    labels: list[int],
    coordinate_space: str,
    image_hw: tuple[int, int] | None = None,
) -> list[PointPrompt]:
    if not (len(xs) == len(ys) == len(labels)):
        raise ValueError("x, y, and labels must have the same length")

    if coordinate_space not in {"percent", "pixel"}:
        raise ValueError("coordinate_space must be 'percent' or 'pixel'")

    height, width = image_hw or (0, 0)
    prompts: list[PointPrompt] = []
    for x, y, label in zip(xs, ys, labels):
        if label not in {0, 1}:
            raise ValueError("Point labels must be 1 for foreground or 0 for background")
        if coordinate_space == "percent":
            point_x = round((float(x) / 100) * max(width - 1, 0))
            point_y = round((float(y) / 100) * max(height - 1, 0))
        else:
            point_x = round(float(x))
            point_y = round(float(y))
        prompts.append(PointPrompt(x=point_x, y=point_y, label=int(label)))
    return prompts

def make_box_prompts_from_raw(
    x1s: list[float],
    y1s: list[float],
    x2s: list[float],
    y2s: list[float],
    coordinate_space: str,
    image_hw: tuple[int, int] | None = None,
) -> list[BoxPrompt]:
    if coordinate_space not in {"percent", "pixel"}:
        raise ValueError("coordinate_space must be 'percent' or 'pixel'")

    height, width = image_hw or (0, 0)
    prompts: list[BoxPrompt] = []
    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        if coordinate_space == "percent":
            x1 = round((x1 / 100) * max(width - 1, 0))
            y1 = round((y1 / 100) * max(height - 1, 0))
            x2 = round((x2 / 100) * max(width - 1, 0))
            y2 = round((y2 / 100) * max(height - 1, 0))
        prompts.append(BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2))
    return prompts
