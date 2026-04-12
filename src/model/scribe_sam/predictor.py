# Source: adapted from upstream FATE-SAM predictor utilities.
# Upstream reference: https://github.com/I3Tlab/FATE-SAM/blob/main/notebooks/predictor_utils.py

import os
import shutil
import tempfile
import warnings
from pathlib import Path


from model.scribe_sam.sam2.build_sam import device_setup, sam2_predictor, sam2_predictor_fate
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model.scribe_sam.sam2.utils.misc import load_video_frames

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
np.random.seed(100)
pd.set_option("display.max_rows", None)


def manhattan_distance_per_pixel(target_feats, image_feats):
    distances = []
    for target_feat, image_feat in zip(target_feats, image_feats):
        dist = torch.abs(target_feat - image_feat).sum(dim=-1).mean().item()
        distances.append(dist)
    return sum(distances) / len(distances)


def add_support_image(existing_tensor, similarity_results, compute_device=torch.device("cpu")):
    new_images_tensor = torch.stack([data["image"] for data in similarity_results.values()], dim=0)
    new_images_tensor = new_images_tensor.to(compute_device)
    existing_tensor = existing_tensor.to(compute_device)
    return torch.cat((existing_tensor, new_images_tensor), dim=0)


def dice_coefficient_label(pred_mask, query_label, actual_label):
    pred_label = (pred_mask == 1).astype(np.int32)
    gt_label = (query_label == actual_label).astype(np.int32)
    intersection = np.logical_and(pred_label, gt_label).sum()
    return (2 * intersection) / (pred_label.sum() + gt_label.sum())


def volume_overlap_error_label(pred_mask, query_label, actual_label):
    pred_label = (pred_mask == 1).astype(np.int32)
    gt_label = (query_label == actual_label).astype(np.int32)
    intersection = np.logical_and(pred_label, gt_label).sum()
    union = np.logical_or(pred_label, gt_label).sum()
    return 1 - (intersection / union)


def volume_difference_label(pred_mask, query_label, actual_label):
    pred_label = (pred_mask == 1).astype(np.int32)
    gt_label = (query_label == actual_label).astype(np.int32)
    return (pred_label.sum() - gt_label.sum()) / gt_label.sum()


def average_hausdorff(pred_mask, query_label, actual_label):
    from scipy.spatial.distance import directed_hausdorff

    pred_points = np.argwhere(pred_mask == 1)
    gt_points = np.argwhere(query_label == actual_label)
    if pred_points.size == 0 or gt_points.size == 0:
        return float("inf")
    hausdorff_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
    hausdorff_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
    return min(hausdorff_pred_to_gt, hausdorff_gt_to_pred)


def evaluation(pred_mask, query_label, actual_label):
    eval_result = {}
    pred_mask = pred_mask.squeeze()
    eval_result[f"DICE {actual_label}"] = dice_coefficient_label(pred_mask, query_label, actual_label)
    eval_result[f"VOE {actual_label}"] = volume_overlap_error_label(pred_mask, query_label, actual_label)
    eval_result[f"VD {actual_label}"] = volume_difference_label(pred_mask, query_label, actual_label)
    eval_result[f"AHD {actual_label}"] = average_hausdorff(pred_mask, query_label, actual_label)
    return eval_result


@torch.inference_mode()
def compute_features(folder_path, images, predictor, pickle_path=None, batch_size=1):
    """Compute deep features for each frame in a video using a given predictor."""
    inference_state = predictor.init_state(folder_path)
    inference_state["images"] = images
    inference_state["num_frames"] = inference_state["images"].size(0)

    features = []
    for idx in range(inference_state["num_frames"]):
        _, _, feature, _, _ = predictor._get_image_feature(inference_state, idx, batch_size=batch_size)
        features.append(feature)

    return features


def find_top_similar_images_embed(support_images, support_features, support_label_array, query_features, top_n=3):
    """Identify the top-N most similar support images for each query feature using Manhattan distance."""
    all_results = []
    for _i, feat_query in tqdm(enumerate(query_features), desc="Finding Similar Images"):
        similarities = []
        for feat_support in support_features:
            if len(feat_query) == len(feat_support) and all(tq.shape == ts.shape for tq, ts in zip(feat_query, feat_support)):
                dist = manhattan_distance_per_pixel(feat_query, feat_support)
                similarities.append(dist)
            else:
                similarities.append(float("inf"))

        similarities_tensor = torch.tensor(similarities)
        sorted_indices = torch.argsort(similarities_tensor, descending=False)

        result = {}
        count = 0
        for idx in sorted_indices:
            if (support_label_array[idx] > 0).any():
                result[idx.item()] = {
                    "image": support_images[idx],
                    "label": support_label_array[idx],
                    "score": similarities_tensor[idx].item(),
                }
                count += 1
            if count >= top_n:
                break

        all_results.append(result)
    return all_results


def _get_prompt_points_and_labels(prompts):
    if not prompts:
        return None, None
    points = []
    labels = []
    for seed in prompts:
        if hasattr(seed, "x") and hasattr(seed, "y") and hasattr(seed, "label"):
            points.append([seed.x, seed.y])
            labels.append(seed.label)
    if not points:
        return None, None
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _resize_mask_to_query(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_hw:
        return mask
    return cv2.resize(mask.astype(np.int32), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def run_inference_single_volume(image_folder, label, similarity_results, predictor, num_classes=0, prompts=None, query_seeds=None):
    """Run segmentation inference on a single image or frame-folder using top-N support images."""
    inference_state = predictor.init_state(image_folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
    image_len = inference_state["num_frames"]
    start_frame_idx = image_len // 2

    inference_state["images"] = add_support_image(inference_state["images"], similarity_results[start_frame_idx])
    inference_state["num_frames"] += len(similarity_results[start_frame_idx])
    predictor.reset_state(inference_state)
    mask_added_flag = False

    if prompts is None:
        prompts = query_seeds

    points, labels = _get_prompt_points_and_labels(prompts)
    if points is not None and labels is not None:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            clear_old_points=True,
        )

    query_hw = (inference_state["video_height"], inference_state["video_width"])

    for idx, (_k, s) in enumerate(similarity_results[start_frame_idx].items()):
        actual_labels = range(1, num_classes) if num_classes is not None and int(num_classes) > 1 else sorted(np.unique(s["label"]))[1:]
        if actual_labels:
            mask_added_flag = True
            resized_label = _resize_mask_to_query(s["label"], query_hw)
            for actual_label in actual_labels:
                mask = (resized_label == actual_label).astype(float)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx + image_len,
                    obj_id=actual_label,
                    mask=mask,
                )

    def _propagate_and_predict(reverse=False, offset=0):
        result = []
        seg_predictions = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_fate(
            inference_state, similarity_results, start_frame_idx=start_frame_idx + offset, reverse=reverse
        ):
            seg_predictions[out_frame_idx] = {
                out_obj_id: (out_mask_logits[x] > 0.0).cpu().numpy() for x, out_obj_id in enumerate(out_obj_ids)
            }

            if label is not None:
                row = {"query_img_idx": out_frame_idx}
                slice_label = label[out_frame_idx]
                for obj_id, pred_mask in seg_predictions[out_frame_idx].items():
                    row.update(evaluation(pred_mask, slice_label, obj_id))
                result.append(row)

        return {"result": result, "seg_predictions": seg_predictions}

    if mask_added_flag:
        out_reverse = _propagate_and_predict(reverse=True)
        out_forward = _propagate_and_predict(reverse=False, offset=1)

        dice_scores = out_reverse["result"] + out_forward["result"]
        seg_predictions = {**out_reverse["seg_predictions"], **out_forward["seg_predictions"]}
        dice_df = pd.DataFrame(dice_scores).sort_values(by="query_img_idx") if label is not None else None
        return dice_df, seg_predictions

    return None, None


def _materialize_query_image_dir(query_image_path: str) -> tuple[str, str | None]:
    if os.path.isdir(query_image_path):
        return query_image_path, None

    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(f"Query path does not exist: {query_image_path}")

    tmp_dir = tempfile.mkdtemp(prefix="fate_sam_query_")
    ext = os.path.splitext(query_image_path)[1].lower()
    target_name = f"00000{ext if ext in ['.jpg', '.jpeg', '.png', '.bmp'] else '.jpg'}"
    shutil.copy2(query_image_path, os.path.join(tmp_dir, target_name))
    return tmp_dir, tmp_dir


def load_image(image_folder):
    """Load frames from a folder and return them as a tensor."""
    compute_device = device_setup()
    image_tensor, _, _ = load_video_frames(
        video_path=image_folder,
        image_size=1024,
        offload_video_to_cpu=True,
        async_loading_frames=False,
        compute_device=compute_device,
    )
    return image_tensor


def load_label(label_path, frame_count=None):
    """Load 2D image label or folder labels as (num_frames, H, W)."""
    if label_path is None:
        return None

    if os.path.isfile(label_path):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Unable to read label image: {label_path}")
        n = frame_count if frame_count is not None else 1
        return np.stack([label] * n, axis=0)

    if os.path.isdir(label_path):
        frame_files = sorted(
            [
                os.path.join(label_path, p)
                for p in os.listdir(label_path)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )
        labels = []
        for fp in frame_files:
            lab = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if lab is None:
                raise FileNotFoundError(f"Unable to read label frame: {fp}")
            labels.append(lab)
        return np.stack(labels, axis=0) if labels else None

    raise FileNotFoundError(f"Label path does not exist: {label_path}")


def load_support_data_from_loader(loader):
    """Load support images and labels from a data loader."""
    sup_data = loader.load_data()
    support_images = [d["image"] for d in sup_data]
    support_images = torch.cat(support_images, dim=0)
    support_labels = [label for d in sup_data for label in d["label"]]
    return support_images, support_labels


def _as_gray_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        return cv2.cvtColor(np.transpose(image[:3], (1, 2, 0)), cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a 2D grayscale image or a 3D color image")


def as_gray_image(image: np.ndarray) -> np.ndarray:
    return _as_gray_image(image)


def _image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    gray = _as_gray_image(np.asarray(image))
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _label_to_mask(label: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    label_array = np.asarray(label)
    if label_array.ndim == 3 and label_array.shape[-1] in (3, 4):
        label_array = cv2.cvtColor(label_array[..., :3], cv2.COLOR_BGR2GRAY)
    if label_array.ndim == 3 and label_array.shape[0] in (3, 4):
        label_array = cv2.cvtColor(np.transpose(label_array[:3], (1, 2, 0)), cv2.COLOR_BGR2GRAY)
    label_array = label_array.astype(np.int32)
    if tuple(label_array.shape[:2]) != tuple(target_hw):
        label_array = cv2.resize(label_array, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return label_array.astype(np.int32)


def load_support_data_from_arrays(support_images, support_labels, image_size=1024):
    support_image_list = list(support_images)
    support_label_list = list(support_labels)
    if len(support_image_list) != len(support_label_list):
        raise ValueError("support_images and support_labels must have the same length")

    processed_images = []
    processed_labels = []
    for image, label in zip(support_image_list, support_label_list):
        image_array = np.asarray(image)
        gray = _as_gray_image(image_array)
        processed_images.append(_image_to_tensor(image_array, image_size=image_size))
        label_mask = _label_to_mask(label, gray.shape)
        unique_values = np.unique(label_mask)
        if unique_values.size <= 2:
            label_mask = (label_mask > 0).astype(np.int32)
        processed_labels.append(label_mask)

    return torch.stack(processed_images, dim=0), processed_labels


def run_single_image_inference(
    query_image_path,
    query_label_path,
    support_images,
    support_labels,
    num_classes=0,
    support_features=None,
    top_n=3,
    prompts=None,
    query_seeds=None,
):
    """Full pipeline for running inference on a query image using support images and labels."""
    if prompts is None:
        prompts = query_seeds

    query_folder, cleanup_dir = _materialize_query_image_dir(query_image_path)
    try:
        query_image = load_image(query_folder)
        query_label = load_label(query_label_path, frame_count=query_image.shape[0])

        predictor = sam2_predictor()
        fate_predictor = sam2_predictor_fate()

        if support_features is None:
            support_features = compute_features(
                folder_path=query_folder,
                images=support_images,
                predictor=predictor,
                pickle_path=None,
                batch_size=1,
            )

        query_feature = compute_features(
            folder_path=query_folder,
            images=query_image,
            predictor=predictor,
        )

        similarity_results = find_top_similar_images_embed(
            support_images, support_features, support_labels, query_feature, top_n=top_n
        )

        dice_df, seg_predictions = run_inference_single_volume(
            image_folder=query_folder,
            label=query_label,
            similarity_results=similarity_results,
            predictor=fate_predictor,
            num_classes=num_classes,
            prompts=prompts,
            query_seeds=query_seeds,
        )

        return dice_df, seg_predictions
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
