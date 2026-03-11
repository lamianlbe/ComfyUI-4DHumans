"""
Sapiens Single Person Pose Tracking node.

Runs Sapiens COCO-WholeBody 133-keypoint pose estimation per frame on
the full image.  Optionally fuses with SMPLest-X 3D predictions when
SMPLest-X is connected (handles edge-clipped / occluded limbs).
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ..humans4d.hmr2.utils.sapiens_inference import (
    run_sapiens_on_bbox,
    fuse_sapiens_smplestx,
)
from ._pose_utils import run_smplestx_on_bbox


class SapiensSinglePoseNode:
    """Single-person Sapiens pose estimation (no PHALP tracker needed)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sapiens": ("SAPIENS",),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Overlay the skeleton on the original frame "
                            "instead of a black canvas."
                        ),
                    },
                ),
            },
            "optional": {
                "smplestx": ("SMPLESTX",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, sapiens, debug, smplestx=None):
        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape
        whole_bbox = np.array([0, 0, img_w, img_h], dtype=np.float32)
        use_fusion = smplestx is not None

        pbar = comfy.utils.ProgressBar(B)
        pose_images = []

        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()

            if debug:
                canvas = img_np.copy()
            else:
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            result = run_sapiens_on_bbox(img_np, whole_bbox, sapiens)
            if result is not None:
                raw_kp = result["pixel_kp"].copy()
                sub = None

                if use_fusion:
                    sx_result = run_smplestx_on_bbox(
                        img_np, whole_bbox,
                        smplestx["model"], smplestx["cfg"])
                    if sx_result is not None:
                        raw_kp, sub = fuse_sapiens_smplestx(
                            raw_kp, sx_result["kp2d"],
                            img_h, img_w, frame_idx=t)

                canvas = render_sapiens_dwpose(
                    canvas, raw_kp, img_h, img_w, substituted=sub)

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0))
            pbar.update(1)

        return (torch.stack(pose_images),)
