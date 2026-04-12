# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import config


def _ensure_hydra_initialized():
    if GlobalHydra.instance().is_initialized():
        return
    config_dir = Path(config.SAM2_CONFIG_PATH).resolve().parent
    initialize_config_dir(config_dir=str(config_dir), version_base="1.2")


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    _ensure_hydra_initialized()

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    _ensure_hydra_initialized()
    hydra_overrides = [
        "++model._target_=model.scribe_sam.sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

def build_sam2_video_predictor_fate(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    _ensure_hydra_initialized()
    hydra_overrides = [
        "++model._target_=model.scribe_sam.sam2.sam2_video_predictor_FATE.SAM2VideoPredictorFATE",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def device_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device


def _sam2_cfg_name() -> str:
    configured = Path(config.SAM2_CONFIG_PATH)
    if configured.exists():
        return configured.name

    config_dir = configured.parent
    fallback_names = ["sam2_hiera_t.yaml", "sam2.1_hiera_t.yaml"]
    for name in fallback_names:
        candidate = config_dir / name
        if candidate.exists():
            logging.warning(
                "Configured SAM2 config not found at %s; falling back to %s",
                configured,
                candidate,
            )
            return name

    raise FileNotFoundError(
        f"Could not find SAM2 config. Tried configured path {configured} and fallbacks {fallback_names} in {config_dir}."
    )


def sam2_checkpoint_path() -> Path:
    configured = Path(config.SAM2_CHECKPOINT_PATH)
    if configured.exists():
        return configured

    ckpt_dir = configured.parent
    fallback_names = ["sam2_hiera_tiny.pt", "sam2.1_hiera_tiny.pt"]
    for name in fallback_names:
        candidate = ckpt_dir / name
        if candidate.exists():
            logging.warning(
                "Configured SAM2 checkpoint not found at %s; falling back to %s",
                configured,
                candidate,
            )
            return candidate

    raise FileNotFoundError(
        f"Could not find SAM2 checkpoint. Tried configured path {configured} and fallbacks {fallback_names} in {ckpt_dir}."
    )


def sam2_predictor():
    device = device_setup()
    return build_sam2_video_predictor(
        config_file=_sam2_cfg_name(),
        ckpt_path=str(sam2_checkpoint_path()),
        device=device,
    )


def sam2_predictor_fate():
    device = device_setup()
    return build_sam2_video_predictor_fate(
        config_file=_sam2_cfg_name(),
        ckpt_path=str(sam2_checkpoint_path()),
        device=device,
    )

