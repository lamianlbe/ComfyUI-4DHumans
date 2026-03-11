"""
Sapiens Single Person Pose Tracking node.

Runs Sapiens COCO-WholeBody 133-keypoint pose estimation per frame on
the full image.  Optionally fuses with SMPLest-X 3D predictions when
SMPLest-X is connected (handles edge-clipped / occluded limbs).

Supports FPS interpolation to 30fps with keypoint-level linear
interpolation for smooth results.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ..humans4d.hmr2.utils.sapiens_inference import (
    run_sapiens_on_bbox,
    fuse_sapiens_smplestx,
)
from ._pose_utils import run_smplestx_on_bbox, resample_keypoints


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
                "interpolate_30fps": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Resample output to 30fps using keypoint-level "
                            "linear interpolation. Requires fps input."
                        ),
                    },
                ),
            },
            "optional": {
                "smplestx": ("SMPLESTX",),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": (
                            "Input video FPS. Used when interpolate_30fps "
                            "is enabled."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, sapiens, debug, interpolate_30fps=False,
                    smplestx=None, fps=24.0):
        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape
        whole_bbox = np.array([0, 0, img_w, img_h], dtype=np.float32)
        use_fusion = smplestx is not None

        pbar = comfy.utils.ProgressBar(2 * B)

        # Pass 1: inference
        timeline = [None] * B
        timeline_sub = [None] * B
        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
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

                timeline[t] = raw_kp
                timeline_sub[t] = sub
            pbar.update(1)

        # FPS resampling
        do_resample = interpolate_30fps and fps > 0 and abs(fps - 30.0) > 0.1
        if do_resample:
            timeline, src_indices = resample_keypoints(timeline, fps, 30.0)
            timeline_sub = [timeline_sub[i] for i in src_indices]
            n_out = len(timeline)
        else:
            n_out = B
            src_indices = list(range(B))

        # Pass 2: render
        pose_images = []
        for t in range(n_out):
            src_t = src_indices[t]
            if debug:
                canvas = (images_nchw[src_t].permute(1, 2, 0)
                          * 255).byte().numpy().copy()
            else:
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            if timeline[t] is not None:
                canvas = render_sapiens_dwpose(
                    canvas, timeline[t], img_h, img_w,
                    substituted=timeline_sub[t])

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0))
            pbar.update(1)

        return (torch.stack(pose_images),)
