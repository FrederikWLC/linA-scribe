import numpy as np


# a simple wrapper around numpy arrays to represent binary masks, with some utility methods for inversion and image conversion
# it helps reenforce the chosen convention, where 1 represents foreground (ink) and 0 represents background
# while in image form, 0 (black) represents foreground and 255 (white) represents background
class BinaryMask(np.ndarray):
    def __new__(cls, input_array):
        obj = (np.asarray(input_array) > 0).astype(np.uint8).view(cls)
        return obj

    @staticmethod
    def from_bool(bool_array: np.ndarray) -> "BinaryMask":
        return BinaryMask(np.asarray(bool_array).astype(np.uint8))

    @staticmethod
    def from_image(image: np.ndarray) -> "BinaryMask":
        if image is None:
            raise ValueError("Cannot create BinaryMask from a missing image.")
        # convert from uint8 image where 0 (black ink) is foreground and 255 (white empty area) is background
        # to binary mask where 1 is foreground and 0 is background
        return BinaryMask((image == 0).astype(np.uint8))

    # returns the union of two binary masks (logical AND), including only the 1s (foreground pixels) that are present in both masks
    @staticmethod
    def from_intersection(*masks: "BinaryMask") -> "BinaryMask":
        assert all(m.shape == masks[0].shape for m in masks)
        return BinaryMask(np.logical_and.reduce(masks))

    # returns the union of two binary masks (logical OR), including all 1s (foreground pixels) from both masks
    @staticmethod
    def from_union(*masks: "BinaryMask") -> "BinaryMask":
        assert all(m.shape == masks[0].shape for m in masks)
        return BinaryMask(np.logical_or.reduce(masks))

    # invert the mask (foreground becomes background and vice versa)
    def invert(self) -> "BinaryMask":
        return BinaryMask(1 - self)

    # returns uint8 array where foreground (1) is converted to 0 black and background (0) is converted to 255 white
    def to_image(self) -> np.ndarray:
        return (self.invert() * 255).astype(np.uint8)
