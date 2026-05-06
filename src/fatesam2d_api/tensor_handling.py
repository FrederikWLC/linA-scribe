from scribe.binary_mask import BinaryMask
import numpy as np
import torch
import cv2

# Convert a grayscale image into a normalized RGB tensor at the target size.
def _image_to_tensor(image_array: np.ndarray, image_size: int=1024) -> torch.Tensor:
    resized = cv2.resize(image_array, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    img_np = rgb_image.astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1)

def images_to_tensor(images, image_size: int = 1024) -> torch.Tensor:
    tensors = [_image_to_tensor(image, image_size=image_size) for image in images]
    return torch.stack(tensors, dim=0)

def _label_to_tensor(label_array: np.ndarray, image_size: int=1024) -> torch.Tensor:
    resized = cv2.resize(label_array, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    binary_img = BinaryMask.from_image(resized).astype(np.uint8)
    return torch.from_numpy(binary_img).unsqueeze(0).float()

def labels_to_tensor(labels, image_size: int = 1024) -> torch.Tensor:
    tensors = [_label_to_tensor(label, image_size=image_size) for label in labels]
    return torch.stack(tensors, dim=0)

# Dataset loader adapted to 2D image+label instead of 3D volumes.
def prepare_query(query_image, image_size=1024):
    arr = np.asarray(query_image)
    frame = _image_to_tensor(arr, image_size=image_size)
    return frame.unsqueeze(0), (int(frame.shape[1]), int(frame.shape[2]))

def add_support_image(existing_tensor, similarity_results, compute_device=torch.device("cpu")):
    new_images_tensor = torch.stack([data['image'] for data in similarity_results.values()], dim=0)
    new_images_tensor = new_images_tensor.to(compute_device)
    existing_tensor = existing_tensor.to(compute_device)
    return torch.cat((existing_tensor, new_images_tensor), dim=0)