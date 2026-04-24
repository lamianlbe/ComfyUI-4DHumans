"""
Load CrowdSAM — a few-shot person segmenter tuned on CrowdHuman.

CrowdSAM (Cai et al., ECCV 2024, arxiv:2407.11464) targets heavily-
occluded crowd scenes. Pipeline:

  1. DINOv2 ViT-L extracts dense patch features from the full image.
  2. A small adapter head (trained 10-shot on CrowdHuman) produces a
     per-patch foreground similarity map.
  3. Positive patches are sampled as point prompts, batched, and fed
     to a modified SAM ViT-L (segment_anything_cs) whose mask decoder
     is conditioned on DINOv2 features.
  4. NMS + stability filter + region cleanup.

The authors report that on CrowdHuman it matches fully-supervised
detectors; in our use case the interesting property is **per-instance
masks for tightly-overlapping people** — something YOLO-seg and SAM3
both blur on dense crowds.

Hardcoded weight paths (follow the ComfyUI ``models/`` convention):
    models/crowdsam/sam_vit_l_0b3195.pth              (SAM ViT-L)
    models/crowdsam/dinov2_vitl14_pretrain.pth        (DINOv2 ViT-L)
    models/crowdsam/crowdhuman_10shot.pth             (adapter head)

Download from:
    SAM ViT-L (~1.3 GB):
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    DINOv2 ViT-L (~1.1 GB):
      https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
    CrowdHuman adapter (~200 MB, Google Drive):
      https://drive.google.com/file/d/18034Wbd_Q01W0eBxlOuh4VzuOIob_eqy/view?usp=sharing
      (save as crowdhuman_10shot.pth)

DINOv2 source is vendored under ``crowdsam_lib/dinov2/`` — no torch
hub download happens at runtime.
"""

import logging
import os

import torch

from folder_paths import models_dir

from ..crowdsam_lib import DINOV2_REPO_PATH, ensure_lib_importable

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded weight paths
# ---------------------------------------------------------------------------
CROWDSAM_DIR = os.path.join(models_dir, "crowdsam")

SAM_VIT_L_FILENAME       = "sam_vit_l_0b3195.pth"
SAM_VIT_L_PATH           = os.path.join(CROWDSAM_DIR, SAM_VIT_L_FILENAME)
SAM_VIT_L_URL            = (
    "https://dl.fbaipublicfiles.com/segment_anything/" + SAM_VIT_L_FILENAME
)

DINOV2_VITL14_FILENAME   = "dinov2_vitl14_pretrain.pth"
DINOV2_VITL14_PATH       = os.path.join(CROWDSAM_DIR, DINOV2_VITL14_FILENAME)
DINOV2_VITL14_URL        = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
    + DINOV2_VITL14_FILENAME
)

# Upstream release filename was ``10_shot.pth``; we rename to make it
# less ambiguous what it is (there's only one CrowdHuman adapter shipped).
ADAPTER_CROWDHUMAN_FILENAME = "crowdhuman_10shot.pth"
ADAPTER_CROWDHUMAN_PATH     = os.path.join(CROWDSAM_DIR, ADAPTER_CROWDHUMAN_FILENAME)
ADAPTER_CROWDHUMAN_URL      = (
    "https://drive.google.com/file/d/"
    "18034Wbd_Q01W0eBxlOuh4VzuOIob_eqy/view?usp=sharing"
)


def _require(path, what, url):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{what} not found at:\n  {path}\n\n"
            f"Download from:\n  {url}\n"
            f"and place it at the path above."
        )


def _build_config(device_str, max_prompts, points_per_batch,
                  pos_sim_thresh, pred_iou_thresh, box_nms_thresh,
                  stability_score_thresh, min_mask_region_area,
                  max_size, grid_size):
    """Build the dict CrowdSAM expects (normally loaded from yaml).

    Values are the CrowdHuman defaults from upstream's crowdhuman.yaml,
    with a few knobs promoted to node inputs for quality/speed tuning.
    """
    return {
        "environ": {
            "device": device_str,
            "seed": 42,
        },
        "data": {
            "dataset": "crowdhuman",
        },
        "model": {
            # ``source='local'`` torch.hub load — our vendored copy
            "dino_repo":              DINOV2_REPO_PATH,
            "dino_checkpoint":        DINOV2_VITL14_PATH,
            "dino_model":             "dinov2_vitl14",
            "sam_checkpoint":         SAM_VIT_L_PATH,
            "sam_model":              "vit_l",
            "sam_arch":               "crowdsam",
            "sam_adapter_checkpoint": ADAPTER_CROWDHUMAN_PATH,
            "n_class":                1,
            "max_size":               max_size,
            # trained adapter mode (not the training-free ref-feature path)
            "trainfree":              False,
        },
        "test": {
            "output_rles":            True,
            # multi-crop inference is worth it for very dense tiny
            # people; most single-shot consumer scenes don't need it
            # and it ~2x's runtime.
            "crop_n_layers":          0,
            "crop_nms_thresh":        0.7,
            "crop_overlap_ratio":     0.341,

            "pos_sim_thresh":         pos_sim_thresh,
            "apply_box_offsets":      False,
            "grid_size":              grid_size,

            "max_prompts":            max_prompts,
            "filter_thresh":          0.7,
            "points_per_batch":       points_per_batch,
            "mask_selection":         "max_iou",
            "max_size":               max_size,

            "fuse_simmap":            False,
            "min_mask_region_area":   min_mask_region_area,
            "box_nms_thresh":         box_nms_thresh,
            "stability_score_thresh": stability_score_thresh,
            "stability_score_offset": 1,
            "pred_iou_thresh":        pred_iou_thresh,
        },
    }


class LoadCrowdSAMNode:
    """Load CrowdSAM (DINOv2 ViT-L + SAM ViT-L + 10-shot CrowdHuman adapter)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. SAM ViT-L + DINOv2 ViT-L "
                            "together use ~4 GiB VRAM + activations; "
                            "CPU works but is 50-100x slower — only "
                            "practical for single-image debugging."
                        ),
                    },
                ),
                "max_prompts": (
                    "INT",
                    {
                        "default": 500,
                        "min": 50,
                        "max": 5000,
                        "step": 50,
                        "tooltip": (
                            "Cap on foreground point prompts per frame. "
                            "Upper bound on number of people the model "
                            "will try to segment — 500 handles very "
                            "dense crowds; lower it (200) for speed if "
                            "your scenes have <10 people."
                        ),
                    },
                ),
                "points_per_batch": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Points per SAM mask-decoder batch. "
                            "Larger = faster but more VRAM."
                        ),
                    },
                ),
                "pos_sim_thresh": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": (
                            "DINOv2-adapter foreground similarity "
                            "threshold. Lower = more candidate points "
                            "(better recall on small / occluded people, "
                            "more compute); higher = cleaner but may "
                            "miss hard cases."
                        ),
                    },
                ),
                "pred_iou_thresh": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": (
                            "Minimum SAM-predicted mask IoU to keep a "
                            "candidate. 0.1 is CrowdHuman's default "
                            "(lenient); raise for cleaner output."
                        ),
                    },
                ),
                "box_nms_thresh": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.1,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": (
                            "IoU threshold for box NMS across "
                            "candidates. Lower = more aggressive "
                            "deduplication."
                        ),
                    },
                ),
                "stability_score_thresh": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Filter masks whose stability score "
                            "(how invariant the mask is to a small "
                            "threshold perturbation) is below this. "
                            "0.8 is CrowdHuman's tuned default."
                        ),
                    },
                ),
                "min_mask_region_area": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 10000,
                        "step": 10,
                        "tooltip": (
                            "Drop masks smaller than this many pixels "
                            "and also fill holes up to this size. 0 = "
                            "disable cleanup."
                        ),
                    },
                ),
                "max_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 512,
                        "max": 2048,
                        "step": 64,
                        "tooltip": (
                            "Longest-side resize target fed into SAM. "
                            "SAM ViT-L was trained at 1024; stick with "
                            "it unless you know what you're doing."
                        ),
                    },
                ),
                "grid_size": (
                    "INT",
                    {
                        "default": 192,
                        "min": 96,
                        "max": 384,
                        "step": 32,
                        "tooltip": (
                            "Resolution of the DINOv2 similarity map "
                            "grid. 192 matches the CrowdHuman config."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CROWDSAM",)
    RETURN_NAMES = ("crowdsam",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, max_prompts, points_per_batch,
             pos_sim_thresh, pred_iou_thresh, box_nms_thresh,
             stability_score_thresh, min_mask_region_area,
             max_size, grid_size):
        _require(SAM_VIT_L_PATH,       "SAM ViT-L checkpoint",       SAM_VIT_L_URL)
        _require(DINOV2_VITL14_PATH,   "DINOv2 ViT-L checkpoint",    DINOV2_VITL14_URL)
        _require(ADAPTER_CROWDHUMAN_PATH, "CrowdHuman adapter",      ADAPTER_CROWDHUMAN_URL)

        ensure_lib_importable()

        # Device fallback — model expects the string form ("cuda" or "cpu").
        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadCrowdSAM: requested CUDA but torch.cuda.is_available() "
                "is False — falling back to CPU."
            )

        config = _build_config(
            device_str=device_str,
            max_prompts=int(max_prompts),
            points_per_batch=int(points_per_batch),
            pos_sim_thresh=float(pos_sim_thresh),
            pred_iou_thresh=float(pred_iou_thresh),
            box_nms_thresh=float(box_nms_thresh),
            stability_score_thresh=float(stability_score_thresh),
            min_mask_region_area=int(min_mask_region_area),
            max_size=int(max_size),
            grid_size=int(grid_size),
        )

        _logger.info(
            "Loading CrowdSAM on device=%s (SAM ViT-L + DINOv2 ViT-L + "
            "CrowdHuman 10-shot adapter)", device_str,
        )

        # Lazy import: the vendored package touches torch.hub (for
        # DINOv2) at construction time, so we defer until paths are
        # validated. CrowdSAM's ``__init__`` also takes a ``logger``
        # positional argument that it ignores — we pass our module
        # logger to shut that up.
        from crowdsam.model import CrowdSAM

        model = CrowdSAM(config, _logger)

        _logger.info(
            "CrowdSAM ready (max_prompts=%d, points_per_batch=%d, "
            "pos_sim_thresh=%.2f, pred_iou_thresh=%.2f)",
            max_prompts, points_per_batch,
            pos_sim_thresh, pred_iou_thresh,
        )

        return ({
            "model":     model,
            "device":    device_str,
            "config":    config,
            # retained for restore-to-cuda / offload-to-cpu hooks that
            # a future perf pass might want; CrowdSAM itself doesn't
            # expose a direct nn.Module handle, but predictor.model is
            # the SAM backbone and predictor.dino_model is the DINOv2
            # backbone.
            "predictor": model.predictor,
        },)
