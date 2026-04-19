# Source: adapted from upstream FATE-SAM predictor utilities.
# Upstream reference: https://github.com/I3Tlab/FATE-SAM/blob/main/notebooks/predictor_utils.py

import os
import warnings
from pathlib import Path


from fatesam_api.model.dataset_loader import (
    add_support_image,
    prepare_query
)
from fatesam_api.model.sam2.build_sam import sam2_predictor, sam2_predictor_fate


import numpy as np
import torch
from scribe.prompts import get_point_prompts_and_labels
from tqdm import tqdm
import logging
import time

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
np.random.seed(100)

# Configure minimal logging so INFO messages are visible by default when no
# other logging configuration is present in the running process.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def manhattan_distance_per_pixel(target_feats, image_feats):
    distances = []
    for target_feat, image_feat in zip(target_feats, image_feats):
        dist = torch.abs(target_feat - image_feat).sum(dim=-1).mean().item()
        distances.append(dist)
    return sum(distances) / len(distances)


@torch.inference_mode()
def compute_features(images, predictor, video_height, video_width, batch_size=1):
    """Compute deep features for each frame in a video using a given predictor."""
    t0 = time.perf_counter()
    
    inference_state = predictor.init_state_from_images(
        images=images,
        video_height=video_height,
        video_width=video_width,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    features = []
    for idx in range(inference_state["num_frames"]):
        logger.info("compute_features: input tensor shape %s", tuple(images.shape))
        _, _, feature, _, _ = predictor._get_image_feature(inference_state, idx, batch_size=batch_size)
        features.append(feature)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == inference_state["num_frames"]:
            logger.info("compute_features: extracted %d/%d frames", idx + 1, inference_state["num_frames"])

    elapsed = time.perf_counter() - t0
    logger.info("compute_features: done, extracted %d features in %.3fs", len(features), elapsed)
    return features


def find_top_similar_images_embed(support_images, support_features, support_labels, query_features, top_n=5):
    """
    Identify the top-N most similar support images for each query feature using Manhattan distance.
    """

    all_results = []
    for i, feat_query in tqdm(enumerate(query_features), desc="Finding Similar Images"):
        similarities = []
        for feat_support in support_features:
            if (
                    len(feat_query) == len(feat_support) and
                    all(tq.shape == ts.shape for tq, ts in zip(feat_query, feat_support))
            ):
                dist = manhattan_distance_per_pixel(feat_query, feat_support)
                similarities.append(dist)
            else:
                similarities.append(float('inf'))

        similarities_tensor = torch.tensor(similarities)

        sorted_indices = torch.argsort(similarities_tensor, descending=False)

        result = {}
        count = 0
        for idx in sorted_indices:
            if (support_labels[idx] > 0).any():
                result[idx.item()] = {
                    'image': support_images[idx],
                    'label': support_labels[idx],
                    'score': similarities_tensor[idx].item(),
                }
                count += 1
            if count >= top_n:
                break

        all_results.append(result)
    return all_results
    
@torch.inference_mode()
def prepare_inference_state(
    query_image,
    support_images,
    support_labels,
    top_n=5
):
    """Prepare inference_state by adding support images and label masks.
    """
    
    query_image, (video_height, video_width) = prepare_query(query_image, image_size=1024)

    # allow caller to provide a predictor and/or cached support_features
    predictor = sam2_predictor()

    inference_state = predictor.init_state_from_images(
        images=query_image,
        video_height=video_height,
        video_width=video_width,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    logger.info("prepare_inference_state: created inference_state with keys: %s", list(inference_state.keys()))

    logger.info("prepare_inference_state: before to(), images tensor shape %s", tuple(inference_state["images"].shape))
    inference_state["images"] = inference_state["images"].to(inference_state["device"])
    logger.info("prepare_inference_state: video_height=%s video_width=%s", video_height, video_width)

    # FIND AND ADD SUPPORT IMAGES
    # ==============
    logger.info("prepare_inference_state: computing support features for %d support images", len(support_images))
    support_features = compute_features(
        images=support_images,
        predictor=predictor,
        batch_size=1,
        video_height=video_height,
        video_width=video_width
    )

    query_feature = compute_features(
        images=query_image,
        predictor=predictor,
        video_height=video_height,
        video_width=video_width
    )

    similarity_results = find_top_similar_images_embed(
        support_images, support_features, support_labels, query_feature,
        top_n=top_n
    )
    logger.info("prepare_inference_state: similarity_results for query: %s", similarity_results[0].keys())
    # Log the full similarity results for debugging
    logger.debug("prepare_inference_state: full similarity_results: %s", similarity_results)

    # Add support images
    inference_state["images"] = add_support_image(
        inference_state["images"],
        similarity_results[0],
        compute_device=inference_state["device"],
    )
    logger.info("prepare_inference_state: after add_support_image, images tensor shape %s", tuple(inference_state["images"].shape))
    logger.info("prepare_inference_state: added %d support images to inference_state", len(similarity_results[0]))
    inference_state["num_frames"] += top_n  # assuming top_n support images are added;
    predictor.reset_state(inference_state)
    logger.info("prepare_inference_state: reset predictor state after adding supports")
    # =============

    return inference_state, similarity_results

@torch.inference_mode()
def run_from_inference_state(
    inference_state,
    similarity_results,
    prompts=None,
    num_classes=0
):
    """Run propagation on a prepared inference_state, optionally adding point prompts.
    
    For single image: query at frame 0, supports at frames 1..N.

    Returns:
        dict[int, np.ndarray] | None:  mask predictions for query frame 0.
    """
    predictor = sam2_predictor_fate()
    points, labels = get_point_prompts_and_labels(prompts)
    if points is not None and labels is not None:
        logger.info("run_from_inference_state: adding point prompts to inference_state")
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
            clear_old_points=True,
        )

    # For single image: query at frame 0, supports are appended after the query frame
    start_frame_idx = 0

    for idx, (k, s) in enumerate(similarity_results[start_frame_idx].items()):
        actual_labels = range(1, num_classes) if num_classes > 0 else sorted(np.unique(s["label"]))[1:]
        if actual_labels:
            mask_added_flag = True
            for actual_label in actual_labels:
                # `s['label']` is a tensor; use tensor ops instead of numpy `.astype`
                mask = (s["label"] == actual_label).float().squeeze(0)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx + 1,  # supports start after query frame
                    obj_id=actual_label,
                    mask=mask,
                )

    def _propagate_and_predict(reverse=False, offset=0):
        """
        Helper function to propagate masks through the volume and collect predictions.
        """
        seg_predictions = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_fate(
                inference_state, similarity_results, start_frame_idx=start_frame_idx + offset, reverse=reverse):
            seg_predictions[out_frame_idx] = {
                out_obj_id: (out_mask_logits[x] > 0.0).cpu().numpy() for x, out_obj_id in enumerate(out_obj_ids)
            }

        return seg_predictions

    out_reverse = _propagate_and_predict(reverse=True)
    out_forward = _propagate_and_predict(reverse=False, offset=1)

    seg_predictions = {**out_reverse, **out_forward}
    return seg_predictions[0]


#Full pipeline for running inference on one query image using support images and labels.
def run_single_image_inference(
    query_image,
    support_images,
    support_labels,
    prompts=None
):

    inference_state, similarity_results = prepare_inference_state(
        query_image=query_image,
        support_images=support_images,
        support_labels=support_labels,
    )
    
    query_frame_prediction = run_from_inference_state(
        inference_state=inference_state,
        similarity_results=similarity_results,
        prompts=prompts,
    )

    return query_frame_prediction
