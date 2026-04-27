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


# Weights live under ComfyUI/models/rtmw/. Config files are resolved
# from the *installed* mmpose package (its data files include all the
# wholebody_2d_keypoint configs we need). Falls back to our vendored
# bmp_configs/mmpose_rtmw/ if the installed mmpose doesn't ship the
# config — only happens with old (pre-1.3) mmpose, since BMP's own
# fork of mmpose is based on 1.3.1 and ships RTMW configs in-tree.
RTMW_MODELS_DIR = os.path.join(models_dir, "rtmw")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTMW_VENDORED_CONFIGS_ROOT = os.path.join(
    REPO_ROOT, "bmp_configs", "mmpose_rtmw",
)


def _resolve_rtmw_config(rel_path: str) -> str:
    """Find a RTMW config file. Prefer the installed mmpose package's
    bundled configs (so a wheel-only install works on any host without
    needing this repo's vendored copy), fall back to vendored paths if
    the installed mmpose lacks them.

    ``rel_path`` is the path relative to ``mmpose/configs/``, e.g.
    ``"wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_*.py"``.
    """
    # Try installed package first — works when BMP's wheel was built
    # with our MANIFEST.in (recursive-include of mmpose/configs/**).
    try:
        import mmpose
        installed_root = os.path.dirname(os.path.abspath(mmpose.__file__))
        candidate = os.path.join(
            installed_root, "configs", rel_path,
        )
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass

    # Fallback: vendored copy (kept around for the case where the
    # mmpose install lacks data files, e.g. broken pip wheels).
    fallback = os.path.join(
        RTMW_VENDORED_CONFIGS_ROOT, "configs", rel_path,
    )
    return fallback

# Variant → (config relative path under mmpose/configs/, expected
# weight filename).
_VARIANTS = {
    "rtmw-x_384x288": (
        "wholebody_2d_keypoint/rtmpose/cocktail14/"
        "rtmw-x_8xb320-270e_cocktail14-384x288.py",
        "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        # ↑ Whole-body AP 70.2 (cocktail14 v1.0). Recommended default.
    ),
    "rtmw-x_256x192": (
        "wholebody_2d_keypoint/rtmpose/cocktail14/"
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
        cfg_path = _resolve_rtmw_config(cfg_rel)
        weight_path = os.path.join(RTMW_MODELS_DIR, weight_filename)

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"RTMW config not found:\n  {cfg_path}\n\n"
                f"Tried installed mmpose package + vendored fallback. "
                f"Make sure mmpose is installed (pip install bboxmaskpose "
                f"if you went the BMP-fork route, or upstream mmpose). "
                f"If the installed mmpose's wheel doesn't ship configs, "
                f"the vendored copy under bmp_configs/mmpose_rtmw/ should "
                f"have been picked up — re-pull the repo."
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

        # mmpose resolves ``metainfo=dict(from_file='configs/_base_/...')``
        # in the test pipeline relative to CWD. Walk up from the chosen
        # config until we hit the parent of the ``configs/`` directory —
        # that's the cwd mmpose's relative lookups need. Works whether
        # cfg_path resolved from the installed mmpose package or our
        # vendored fallback.
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        chdir_target = cfg_dir
        while chdir_target and os.path.basename(chdir_target) != "configs":
            parent = os.path.dirname(chdir_target)
            if parent == chdir_target:
                chdir_target = None  # walked past root, give up
                break
            chdir_target = parent
        if chdir_target:
            chdir_target = os.path.dirname(chdir_target)  # parent of configs/

        original_cwd = os.getcwd()
        try:
            if chdir_target:
                os.chdir(chdir_target)
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
