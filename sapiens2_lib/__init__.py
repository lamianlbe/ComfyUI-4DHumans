"""
Standalone Sapiens2 pose runtime — torch + numpy + cv2 + safetensors only.

Why standalone: Meta ships a ``sapiens.backbones.standalone.sapiens2``
file specifically so users can drop the model into projects without
importing the full mmcv / mmdet / mmpose / sapiens.engine /
sapiens.registry tower. We follow the same approach for the pose
head + UDP heatmap codec, so this whole package is pure torch +
numpy + cv2 + safetensors.

Vendored from facebookresearch/sapiens2 (CC-BY-NC 4.0). See LICENSE
in the upstream repo. Files in this directory:

  sapiens2_backbone.py      — backbone (verbatim from
                              sapiens/backbones/standalone/sapiens2.py)
  pose_heatmap_head.py      — pose head, with `@MODELS.register_module`
                              decorator + `from sapiens.registry...`
                              + `pose_pck_accuracy` import stripped.
                              `loss(...)` removed (training-only).
  udp_heatmap.py            — UDP heatmap codec, import path
                              changed from `.utils` to `.codec_utils`.
  codec_utils/              — codec helper utilities (verbatim from
                              upstream `pose/src/datasets/codecs/utils/`),
                              wrapped in __init__.py for re-export.
  keypoints308_metainfo.py  — 308-keypoint definition with skeleton
                              edges + colors (verbatim).

Public API (use from ComfyUI nodes):

    from sapiens2_lib import (
        Sapiens2PosePipeline,         # high-level wrapper
        SAPIENS2_1B_HPARAMS,          # arch hyperparams
        SAPIENS2_HEAD_HPARAMS,        # decode head hyperparams
        SAPIENS2_INPUT_SIZE,          # (W=768, H=1024)
        SAPIENS2_HEATMAP_SIZE,        # (W=192, H=256)  (input / 4)
        load_keypoints308_metainfo,
    )

The pipeline construct + load + infer flow looks like:

    pipe = Sapiens2PosePipeline.from_safetensors(
        path="models/sapiens2/sapiens2_1b_pose.safetensors",
        device="cuda",
    )
    keypoints, scores = pipe.predict(
        image_bgr_uint8,   # (H, W, 3) BGR uint8
        bboxes_xyxy,        # (N, 4) float
    )
    # keypoints: (N, 308, 2) float in original image pixel coords
    # scores:    (N, 308)    float in [0, 1]
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

LIB_PATH = os.path.dirname(os.path.abspath(__file__))


def ensure_lib_importable():
    """Prepend our vendored package to ``sys.path`` if not already there.

    Idempotent. Pose nodes call this once at module load time.
    """
    if LIB_PATH not in sys.path:
        sys.path.insert(0, LIB_PATH)


# Make sub-imports work without the user having to mess with sys.path:
# We import inside this __init__ relatively, so users can just do
# `from sapiens2_lib import Sapiens2PosePipeline`.
from .sapiens2_backbone import Sapiens2  # noqa: E402
from .pose_heatmap_head import PoseHeatmapHead  # noqa: E402
from .udp_heatmap import UDPHeatmap  # noqa: E402


# --------------------------------------------------------------------------
# Hyperparameters extracted from the upstream config
#   sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/
#       sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py
#
# These mirror what `init_model(config, ckpt)` would build via the mmcv
# registry — we hard-code them to skip the registry / Config.fromfile
# machinery entirely.
# --------------------------------------------------------------------------

# 1B variant: this is the only size we ship for now. Add 0.4b / 0.8b /
# 5b dicts here if/when we expose those as additional load options.
SAPIENS2_1B_HPARAMS = dict(
    arch="sapiens2_1b",
    img_size=(1024, 768),         # (H, W)  — note: backbone arg
    patch_size=16,
    final_norm=True,
    use_tokenizer=False,
    with_cls_token=True,
    out_type="featmap",
)

# Decode head channel config. Output 308 → can be parametrized later
# if we expose other keypoint sets (hand-only, face-only, etc.).
SAPIENS2_HEAD_HPARAMS = dict(
    in_channels=1536,             # = backbone embed_dims for 1B
    out_channels=308,
    deconv_out_channels=(1536, 1024),
    deconv_kernel_sizes=(4, 4),
    conv_out_channels=(768, 512, 256),
    conv_kernel_sizes=(1, 1, 1),
    loss_decode=None,             # inference-only
)

# Image preprocessor stats, ImageNet-style. BGR→RGB toggle on; we feed
# BGR uint8 (cv2 convention) and let the preprocessor swap channels.
SAPIENS2_INPUT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAPIENS2_INPUT_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)
SAPIENS2_BGR_TO_RGB = True

# Top-down crop / heatmap dimensions.
# input_size in upstream is (W, H) i.e. (768, 1024). heatmap is /4.
SAPIENS2_INPUT_SIZE   = (768, 1024)   # (W, H) the network wants
SAPIENS2_HEATMAP_SIZE = (192,  256)   # (W, H)
SAPIENS2_HEATMAP_SIGMA = 6


# --------------------------------------------------------------------------
# Topdown crop transform
# --------------------------------------------------------------------------
# Upstream goes:  bbox  →  PoseGetBBoxCenterScale  →  PoseTopdownAffine
#                            (use_udp=True)        →  cropped image
# We reimplement here with cv2 directly. Math is from
# sapiens/pose/src/datasets/transforms/{topdown_affine.py,
# common_transforms.py}.

# Padding to extend bbox (1.25x in upstream's PoseGetBBoxCenterScale —
# the test pipeline default). Larger means more context around the
# subject; affects keypoint accuracy on bbox edges.
_BBOX_PADDING = 1.25


def _bbox_xyxy_to_center_scale(
    bboxes_xyxy: np.ndarray,
    aspect_ratio: float,
    padding: float = _BBOX_PADDING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert xyxy bboxes to (center, scale) the topdown affine wants.

    aspect_ratio = input_w / input_h (so 768/1024 = 0.75 for sapiens2).
    Returns center (N, 2), scale (N, 2) — both float32.
    """
    bb = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
    x1, y1, x2, y2 = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5

    # Match input aspect ratio by inflating the shorter axis (so the
    # final affine doesn't squish the body).
    pixel_std = 200.0  # legacy mmpose convention; only relative matters
    new_w = np.where(w > h * aspect_ratio, w, h * aspect_ratio)
    new_h = new_w / aspect_ratio
    scale_x = new_w * padding / pixel_std
    scale_y = new_h * padding / pixel_std

    center = np.stack([cx, cy], axis=-1)        # (N, 2)
    scale  = np.stack([scale_x, scale_y], axis=-1) * pixel_std  # (N, 2)
    return center, scale


def _get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: Tuple[int, int],   # (W, H)
) -> np.ndarray:
    """Return the 2x3 affine warp matrix for the UDP topdown crop.

    Matches mmpose's `get_udp_warp_matrix(center, scale, rot=0, output_size)`
    with rot=0. UDP differs from the legacy affine by using
    output_size-1 (instead of output_size) as the destination grid
    extent — this avoids an off-by-half-pixel that previously hurt
    sub-pixel keypoint precision.
    """
    out_w, out_h = output_size
    sx, sy = scale[0], scale[1]

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, sy * 0.5], dtype=np.float32)
    src[2] = center + np.array([sx * 0.5, 0.0], dtype=np.float32)

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [(out_w - 1) * 0.5, (out_h - 1) * 0.5]
    dst[1] = [(out_w - 1) * 0.5, (out_h - 1) * 0.5 + (out_h - 1) * 0.5]
    dst[2] = [(out_w - 1) * 0.5 + (out_w - 1) * 0.5, (out_h - 1) * 0.5]

    import cv2 as _cv2
    return _cv2.getAffineTransform(src, dst)


def _topdown_crop(
    image_bgr: np.ndarray,
    bboxes_xyxy: np.ndarray,
    input_size: Tuple[int, int] = SAPIENS2_INPUT_SIZE,
):
    """For each bbox, return (crop, center, scale).

    crop: (input_h, input_w, 3) uint8 BGR
    center: (2,) float32
    scale:  (2,) float32  — the bbox's "size in original image" in the
            UDP convention (used later to map keypoints back).
    """
    import cv2 as _cv2
    in_w, in_h = input_size
    aspect = in_w / in_h

    centers, scales = _bbox_xyxy_to_center_scale(bboxes_xyxy, aspect)
    crops = []
    for i in range(centers.shape[0]):
        M = _get_udp_warp_matrix(centers[i], scales[i], input_size)
        crop = _cv2.warpAffine(image_bgr, M, (in_w, in_h),
                                flags=_cv2.INTER_LINEAR)
        crops.append(crop)

    return np.stack(crops, axis=0), centers, scales


def _normalize_for_network(crops_bgr_u8: np.ndarray) -> torch.Tensor:
    """Stack of BGR uint8 crops → BCHW float32 tensor matching the
    pretrained model's input distribution (RGB, ImageNet-normalized).
    """
    if SAPIENS2_BGR_TO_RGB:
        crops = crops_bgr_u8[..., ::-1]   # (N, H, W, 3) RGB
    else:
        crops = crops_bgr_u8
    crops = crops.astype(np.float32)
    crops -= SAPIENS2_INPUT_MEAN
    crops /= SAPIENS2_INPUT_STD
    crops = crops.transpose(0, 3, 1, 2)   # (N, 3, H, W)
    return torch.from_numpy(np.ascontiguousarray(crops))


# --------------------------------------------------------------------------
# Topdown estimator wrapper (replaces upstream PoseTopdownEstimator)
# --------------------------------------------------------------------------

class _PoseTopdownEstimator(torch.nn.Module):
    """Minimal backbone+head module with the same parameter naming as
    upstream's mmpose-based ``PoseTopdownEstimator``, so the safetensors
    state_dict (saved with that naming) loads cleanly via
    ``model.load_state_dict(strict=False)``.
    """

    def __init__(self, backbone: Sapiens2, decode_head: PoseHeatmapHead):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns (B, C, h, w) when out_type='featmap'
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return self.decode_head(feats)


# --------------------------------------------------------------------------
# High-level pipeline
# --------------------------------------------------------------------------

class Sapiens2PosePipeline:
    """Convenient wrapper: load weights, run inference on (image, bboxes).

    Example:

        pipe = Sapiens2PosePipeline.from_safetensors(
            "models/sapiens2/sapiens2_1b_pose.safetensors",
            device="cuda",
        )
        kp, sc = pipe.predict(image_bgr_uint8, bboxes_xyxy)
        # kp.shape == (N, 308, 2),  sc.shape == (N, 308)
    """

    def __init__(self, model: _PoseTopdownEstimator, codec: UDPHeatmap,
                 device: str):
        self.model = model
        self.codec = codec
        self.device = device

    @classmethod
    def from_safetensors(
        cls,
        path: str,
        device: str = "cuda",
        arch_hparams: Optional[dict] = None,
        head_hparams: Optional[dict] = None,
    ) -> "Sapiens2PosePipeline":
        """Build a 1B pose model and load weights from a .safetensors file.

        Pass `arch_hparams` / `head_hparams` to swap in 0.4b/0.8b/5b
        sizes; defaults are the 1B variant.
        """
        from safetensors.torch import load_file

        backbone = Sapiens2(**(arch_hparams or SAPIENS2_1B_HPARAMS))
        head = PoseHeatmapHead(**(head_hparams or SAPIENS2_HEAD_HPARAMS))
        model = _PoseTopdownEstimator(backbone, head)

        state = load_file(path, device="cpu")
        # The upstream-saved state_dict already uses `backbone.*` and
        # `decode_head.*` prefixes (matching our module nesting), so a
        # direct load works. We allow strict=False because some
        # checkpoint variants embed extra keys (e.g. data_preprocessor
        # buffers) that we don't carry on this trimmed wrapper.
        incompat = model.load_state_dict(state, strict=False)
        if incompat.missing_keys:
            # Pose-head conv weights MUST be present; if they're missing
            # something is wrong with the checkpoint.
            head_missing = [k for k in incompat.missing_keys
                            if k.startswith("decode_head.")]
            if head_missing:
                raise RuntimeError(
                    f"Sapiens2 pose head weights missing from "
                    f"{path}: {head_missing[:5]}{'...' if len(head_missing) > 5 else ''}. "
                    f"Check you downloaded the *_pose.safetensors variant "
                    f"(not *_pretrain.safetensors which only has backbone)."
                )
            # Backbone-only missing keys are acceptable (e.g. unused
            # storage tokens) — log but continue.

        codec = UDPHeatmap(
            input_size=SAPIENS2_INPUT_SIZE,
            heatmap_size=SAPIENS2_HEATMAP_SIZE,
            sigma=SAPIENS2_HEATMAP_SIGMA,
        )

        model.to(device)
        model.eval()
        return cls(model, codec, device)

    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self

    @torch.inference_mode()
    def predict(
        self,
        image_bgr: np.ndarray,
        bboxes_xyxy: np.ndarray,
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-down inference on a list of bboxes within one image.

        Args:
            image_bgr:   (H, W, 3) uint8 BGR (cv2 convention).
            bboxes_xyxy: (N, 4) float xyxy in original image pixels.
            batch_size:  per-forward batch on the ViT. 4 is a safe
                         default for the 1B model on a 24 GB GPU at
                         1024×768.

        Returns:
            keypoints: (N, 308, 2) float32 in original image pixel coords.
            scores:    (N, 308) float32 confidence in [0, 1].
        """
        bboxes_xyxy = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if bboxes_xyxy.shape[0] == 0:
            return (np.zeros((0, SAPIENS2_HEAD_HPARAMS["out_channels"], 2),
                              dtype=np.float32),
                    np.zeros((0, SAPIENS2_HEAD_HPARAMS["out_channels"]),
                              dtype=np.float32))

        crops, centers, scales = _topdown_crop(
            image_bgr, bboxes_xyxy, SAPIENS2_INPUT_SIZE,
        )
        inputs = _normalize_for_network(crops).to(self.device)

        # Forward in batches to bound VRAM
        heatmaps_list = []
        N = inputs.shape[0]
        for i in range(0, N, batch_size):
            chunk = inputs[i:i + batch_size]
            hm = self.model(chunk)            # (b, K, hm_h, hm_w)
            heatmaps_list.append(hm.detach().to("cpu", non_blocking=True))
        heatmaps = torch.cat(heatmaps_list, dim=0).float().numpy()

        # UDP decode: heatmaps (N, K, hm_h, hm_w) → (N, K, 2) in input
        # crop coords + (N, K) scores.
        kp_local_list = []
        sc_list = []
        for i in range(N):
            kp_i, sc_i = self.codec.decode(heatmaps[i])  # (1, K, 2), (1, K)
            kp_local_list.append(kp_i[0])
            sc_list.append(sc_i[0])
        kp_local = np.stack(kp_local_list, axis=0)       # (N, K, 2)
        scores   = np.stack(sc_list,       axis=0)       # (N, K)

        # Map keypoints from input-crop coords back to original image
        # coords. Formula matches vis_pose.py:103-105 exactly.
        in_w, in_h = SAPIENS2_INPUT_SIZE
        input_size = np.array([in_w, in_h], dtype=np.float32)
        # centers (N, 2), scales (N, 2)  — per-bbox
        kp_global = (
            kp_local / input_size                # → [0, 1] of input crop
            * scales[:, None, :]                 # → bbox span in img
            + centers[:, None, :]                # → bbox center
            - 0.5 * scales[:, None, :]           # → top-left of bbox
        )
        return kp_global.astype(np.float32), scores.astype(np.float32)


# --------------------------------------------------------------------------
# Metainfo accessor
# --------------------------------------------------------------------------

def load_keypoints308_metainfo() -> dict:
    """Return the 308-keypoint metainfo dict (skeleton edges, joint
    names, colors). Sourced from
    ``sapiens/pose/configs/_base_/keypoints308.py`` verbatim — that
    file is plain Python that defines a dict at module level.
    """
    from . import keypoints308_metainfo as _meta
    # Upstream module exposes its definitions as module-level names.
    # We re-pack into a dict so callers don't import-coupled.
    out = {}
    for name in (
        "dataset_info", "skeleton_info", "keypoint_info",
        "joint_weights", "sigmas", "skeleton",
    ):
        if hasattr(_meta, name):
            out[name] = getattr(_meta, name)
    return out


__all__ = [
    "ensure_lib_importable",
    "Sapiens2",
    "PoseHeatmapHead",
    "UDPHeatmap",
    "Sapiens2PosePipeline",
    "SAPIENS2_1B_HPARAMS",
    "SAPIENS2_HEAD_HPARAMS",
    "SAPIENS2_INPUT_SIZE",
    "SAPIENS2_HEATMAP_SIZE",
    "load_keypoints308_metainfo",
]
