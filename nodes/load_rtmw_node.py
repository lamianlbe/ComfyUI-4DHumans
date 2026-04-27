"""
Load RTMW-x — OpenMMLab's RTM-WholeBody pose estimator.

Outputs 133-point COCO-WholeBody:
  0..16    body  (COCO-17)
  17..22   feet  (big_toe / small_toe / heel × L/R)
  23..90   face  (68 iBUG points)
  91..111  left  hand (21 MANO joints)
  112..132 right hand (21 MANO joints)

This is the "base" 2D pose estimator we use for everything that BMP's
17-point body output doesn't cover (feet, face, hands). Optionally
the hand portion (91..132) gets overridden by WiLoR for higher
quality — see BMPRTMWPoseNode for the composite path.

RTMW-x is top-down: needs a person bbox per detection. We feed BMP's
bboxes directly via mmpose's ``inference_topdown(model, img,
bboxes=[xyxy])``. No external person detector needed.

Hardcoded weight path (ComfyUI convention):
    models/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth   (~250 MB)
    models/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth   (~250 MB, faster variant)

Download:
    https://download.openmmlab.com/mmpose/v1/projects/rtmw/<filename>

Configs are vendored under bmp_configs/mmpose_rtmw/ — same approach
as BMP since mmpose's pip wheel doesn't ship config files. License:
mmpose Apache-2.0; RTMW weights Apache-2.0.

Score range note: RTMW's SimCC codec actually emits TWO score
channels when ``decode_visibility=True`` (which our vendored configs
have):

  * ``pred_instances.keypoint_scores``: raw ``min(max simcc_x,
    max simcc_y)``. NOT normalised — typical range 0.2 - 3+. Despite
    the name this is the LESS useful one for downstream filtering.
  * ``pred_instances.keypoints_visible``: same peaks but with
    ``simcc * decode_beta * sigma`` then softmax. **Clean [0, 1]
    probability** — what you actually want for "is this joint
    visible/reliable". BMPRTMWPose downstream prefers this field
    automatically.

So the user-observed values like ``[1.9, 1.23, ...]`` are the raw
``keypoint_scores`` (the verification path used those because we
didn't request ``keypoints_visible`` at debug time). The composite
node uses the cleaner ``keypoints_visible`` automatically.
"""

import logging
import os

import torch

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


# Hardcoded layout: weights under models/rtmw/, configs vendored in repo.
RTMW_MODELS_DIR = os.path.join(models_dir, "rtmw")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTMW_CONFIGS_ROOT = os.path.join(
    REPO_ROOT, "bmp_configs", "mmpose_rtmw",
)

# Variant → (config relative path, expected weight filename, HF URL)
_VARIANTS = {
    "rtmw-x_384x288": (
        "configs/wholebody_2d_keypoint/rtmpose/cocktail14/"
        "rtmw-x_8xb320-270e_cocktail14-384x288.py",
        "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        # ↑ Whole-body AP 70.2 (cocktail14 v1.0). Recommended default.
    ),
    "rtmw-x_256x192": (
        "configs/wholebody_2d_keypoint/rtmpose/cocktail14/"
        "rtmw-x_8xb704-270e_cocktail14-256x192.py",
        "rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth",
        # ↑ Whole-body AP 67.2, ~2× faster. For real-time use.
    ),
}


class LoadRTMWNode:
    """Load mmpose's RTMW-x whole-body 2D pose estimator (133 keypoints)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. RTMW-x is small (~250 MB "
                            "weights, ~17 GFLOPs at 384×288 input) — runs "
                            "comfortably on any GPU. CPU is usable but "
                            "~5-10× slower."
                        ),
                    },
                ),
                "variant": (
                    list(_VARIANTS.keys()),
                    {
                        "default": "rtmw-x_384x288",
                        "tooltip": (
                            "Input resolution. 384×288 = WB AP 70.2 "
                            "(slower but more accurate, recommended for "
                            "video where you can afford a few extra ms "
                            "per person). 256×192 = WB AP 67.2 "
                            "(2× faster). Pick variant matching the "
                            "downloaded checkpoint."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("RTMW",)
    RETURN_NAMES = ("rtmw",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, variant):
        cfg_rel, weight_filename, *_ = _VARIANTS[variant]
        cfg_path = os.path.join(RTMW_CONFIGS_ROOT, cfg_rel)
        weight_path = os.path.join(RTMW_MODELS_DIR, weight_filename)

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Vendored RTMW config missing: {cfg_path}\n"
                f"Should ship with the repo under bmp_configs/mmpose_rtmw/. "
                f"Reinstall/pull the repo."
            )
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"RTMW weights not found at:\n  {weight_path}\n\n"
                f"Download from:\n"
                f"  https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
                f"{weight_filename}\n\n"
                f"Place at the path above and retry. Typical command:\n"
                f"  mkdir -p {RTMW_MODELS_DIR}\n"
                f"  wget -O {weight_path} \\\n"
                f"    https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
                f"{weight_filename}"
            )

        try:
            from mmpose.apis import init_model
        except ImportError as e:
            raise ImportError(
                "mmpose is not installed in this Python env. RTMW-x "
                "requires mmpose>=1.3 + mmcv>=2.2 + mmengine. See your "
                "BMP install — same dependency stack."
            ) from e

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadRTMW: requested CUDA but torch.cuda.is_available() "
                "is False — falling back to CPU."
            )

        _logger.info(
            "Loading RTMW-x (%s) on %s\n  config:  %s\n  weights: %s",
            variant, device_str, cfg_path, weight_path,
        )

        # mmpose resolves `metainfo=dict(from_file='configs/_base_/...')` in
        # the test pipeline relative to CWD. Our vendored config tree
        # mirrors mmpose's relative paths under bmp_configs/mmpose_rtmw/,
        # so we briefly cd there during init_model so those lookups
        # land in our vendored copies. After init the model carries its
        # config in-memory and CWD doesn't matter.
        original_cwd = os.getcwd()
        try:
            os.chdir(RTMW_CONFIGS_ROOT)
            model = init_model(cfg_path, weight_path, device=device_str)
        finally:
            os.chdir(original_cwd)

        _logger.info(
            "RTMW-x ready. Output: 133 keypoints (body 0..16, feet "
            "17..22, face 23..90, hands 91..132). Note: scores are "
            "SimCC unnormalized (range 0.2-3+, not 0-1)."
        )

        return ({
            "model":       model,
            "device":      device_str,
            "variant":     variant,
            "config_path": cfg_path,
            "weight_path": weight_path,
        },)
