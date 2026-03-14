"""
Sapiens Single Person Pose Tracking (Goliath) node.

Uses the Sapiens 1B Goliath model (308 keypoints after teeth removal).
No SMPLest-X fusion — pure Sapiens only.
Body/hand/feet joints are connected with lines; face landmarks are dots only.

Supports FPS interpolation to 30fps with keypoint-level linear interpolation.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_goliath import render_goliath
from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox
from ._pose_utils import resample_keypoints


class SapiensGoliathPoseNode:
    """Single-person Goliath 308-keypoint pose estimation."""

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
                "show_face": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Draw dense face landmark dots. "
                            "Disable to show only body/hand/feet."
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

    def render_pose(self, images, sapiens, debug, show_face,
                    interpolate_30fps=False, fps=24.0):
        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape
        whole_bbox = np.array([0, 0, img_w, img_h], dtype=np.float32)

        pbar = comfy.utils.ProgressBar(2 * B)

        # Pass 1: inference
        timeline = [None] * B
        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
            result = run_sapiens_on_bbox(img_np, whole_bbox, sapiens)
            if result is not None:
                timeline[t] = result["pixel_kp"].copy()
            pbar.update(1)

        # FPS resampling
        do_resample = interpolate_30fps and fps > 0 and abs(fps - 30.0) > 0.1
        if do_resample:
            timeline, src_indices = resample_keypoints(timeline, fps, 30.0)
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
                canvas = render_goliath(
                    canvas, timeline[t], img_h, img_w,
                    show_face=show_face)

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0))
            pbar.update(1)

        return (torch.stack(pose_images),)
