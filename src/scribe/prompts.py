import numpy as np


class Prompt:
    pass


class PointPrompt(Prompt):
    def __init__(self, x: int, y: int, label: int):
        self.x = x
        self.y = y
        self.label = label

class PointPromptList(Prompt):
    def __init__(self, point_prompts: list[PointPrompt]):
        self.point_prompts = point_prompts

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.point_prompts:
            return None, None
        points = np.array([[prompt.x, prompt.y] for prompt in self.point_prompts])
        labels = np.array([prompt.label for prompt in self.point_prompts])
        return points, labels
    
class BoxPrompt(Prompt):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def to_array(self) -> np.ndarray:
        return np.array([[self.x1, self.y1], [self.x2, self.y2]])
    
    
class BoxPointPrompt(Prompt):
    def __init__(self, box_prompt: BoxPrompt, point_prompt_list: PointPromptList):
        self.box_prompt = box_prompt
        self.point_prompt_list = point_prompt_list

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        box_array = self.box_prompt.to_array()
        points_array, labels_array = self.point_prompt_list.to_arrays()
        return box_array, points_array, labels_array

def build_box_point_prompts(box_prompts: list[BoxPrompt], point_prompt_list: PointPromptList) -> list[BoxPointPrompt]:
    points_array, _ = point_prompt_list.to_arrays()
    
    def filter_by_box(boxprompt: BoxPrompt) -> "PointPromptList":
        if points_array is None:
            return PointPromptList([])
        x1, y1, x2, y2 = boxprompt.to_array().flatten()
        mask = (
            (points_array[:, 0] >= x1)
            & (points_array[:, 0] <= x2)
            & (points_array[:, 1] >= y1)
            & (points_array[:, 1] <= y2)
        )
        filtered_points = [
            point_prompt_list.point_prompts[i]
            for i in np.nonzero(mask)[0]
        ]
        return PointPromptList(filtered_points)

    return [BoxPointPrompt(box_prompt, filter_by_box(box_prompt)) for box_prompt in box_prompts]

class BrushPrompt(Prompt):
    def __init__(self, pixels: list[tuple[int, int]], label: int):
        self.pixels = pixels
        self.label = label


def get_points_and_labels(promptlist: list[Prompt] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not promptlist:
        return None, None
    points = []
    labels = []
    for prompt in promptlist:
        if isinstance(prompt, PointPrompt):
            points.append([prompt.x, prompt.y])
            labels.append(prompt.label)
    return (np.array(points), np.array(labels)) if points else (None, None)


def get_brush_prompts(promptlist: list[Prompt] | None) -> list[BrushPrompt]:
    return [prompt for prompt in promptlist if isinstance(prompt, BrushPrompt)] if (not promptlist is None) else []
