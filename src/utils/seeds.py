import numpy as np

def get_points_and_labels(seedlist):
    points = []
    labels = []
    for seed in seedlist:
        if isinstance(seed,PointSeed):
            points.append([seed.x, seed.y])
            labels.append(seed.label)
    return (np.array(points), np.array(labels)) if points else (None, None)

def get_boxseeds(seedlist):
    return [seed for seed in seedlist if isinstance(seed, BoxSeed)] if seedlist else []

def get_brushseeds(seedlist):
    return [seed for seed in seedlist if isinstance(seed, BrushSeed)] if seedlist else []

def get_boxes(seedlist):
    return [np.array([[seed.x1, seed.y1], [seed.x2, seed.y2]]) for seed in seedlist if isinstance(seed, BoxSeed)]

class Seed:
    pass

class PointSeed(Seed):
    def __init__(self, x: int, y: int, label: int):
        self.x = x
        self.y = y
        self.label = label

class BoxSeed(Seed):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def bigger_from_boxes(boxes: list['BoxSeed']) -> 'BoxSeed':
        if not boxes:
            return None
        x1 = min(box.x1 for box in boxes)
        y1 = min(box.y1 for box in boxes)
        x2 = max(box.x2 for box in boxes)
        y2 = max(box.y2 for box in boxes)
        return BoxSeed(x1, y1, x2, y2)

class BrushSeed(Seed):
    def __init__(self, pixels: list[tuple[int,int]], label: int):
        self.pixels = pixels
        self.label = label