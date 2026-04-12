import cv2
import numpy as np


class Prompt:
    pass


class PointPrompt(Prompt):
    def __init__(self, x: int, y: int, label: int):
        self.x = x
        self.y = y
        self.label = label


class BoxPrompt(Prompt):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class BrushPrompt(Prompt):
    def __init__(self, pixels: list[tuple[int, int]], label: int):
        self.pixels = pixels
        self.label = label


def get_point_prompts_and_labels(promptlist):
    points = []
    labels = []
    for prompt in promptlist:
        if isinstance(prompt, PointPrompt):
            points.append([prompt.x, prompt.y])
            labels.append(prompt.label)
    return (np.array(points), np.array(labels)) if points else (None, None)


def get_brush_prompts(promptlist):
    return [prompt for prompt in promptlist if isinstance(prompt, BrushPrompt)] if (not promptlist is None) else []