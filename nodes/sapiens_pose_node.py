"""
Sapiens 2D Human Pose node.

Unified node that replaces the former single-person and multi-person
Sapiens nodes.  Accepts RGB images (required), an optional Sapiens model,
and optional PromptHMR 3D pose data (POSE_3D).

Logic:
  - Neither connected  → error
  - Sapiens only       → full COCO-WholeBody 133 keypoints from Sapiens
  - 3D only            → body+feet from PromptHMR (23 keypoints), no face/hands
  - Both               → body+feet from 3D, face+hands from Sapiens

Output is rendered as DWPose-style skeleton images.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox
from ._pose_utils import (
    openpose25_to_coco_wholebody,
    fuse_3d_body_with_sapiens,
    resample_keypoints,
)


# Target FPS lookup
_FPS_MAP = {
    "source": None,
    "16fps": 16.0,
    "24fps": 24.0,
    "30fps": 30.0,
}


class SapiensPoseNode:
    """Sapiens 2D whole-body pose estimation (single or multi person)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
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
                "frame_rate": (
                    ["source", "16fps", "24fps", "30fps"],
                    {
                        "default": "source",
                        "tooltip": (
                            "Output frame rate. 'source' keeps original. "
                            "Other options resample to the target FPS."
                        ),
                    },
                ),
            },
            "optional": {
                "sapiens": ("SAPIENS",),
                "pose_3d": ("POSE_3D",),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": (
                            "Source video FPS. Required when frame_rate "
                            "is not 'source'."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("pose_images", "fps")
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, debug, frame_rate,
                    sapiens=None, pose_3d=None, fps=24.0):
        if sapiens is None and pose_3d is None:
            raise ValueError(
                "Sapiens 2D Human Pose: at least one of 'sapiens' or "
                "'pose_3d' must be connected."
            )

        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape

        use_sapiens = sapiens is not None
        use_3d = pose_3d is not None
        n_persons = pose_3d["n_persons"] if use_3d else 1

        pbar = comfy.utils.ProgressBar(2 * B)

        # -----------------------------------------------------------
        # Pass 1: build per-frame, per-person COCO-WB keypoints
        # -----------------------------------------------------------
        # frame_kps[t] = list of (133, 3) arrays, one per person
        frame_kps = [[] for _ in range(B)]

        whole_bbox = np.array([0, 0, img_w, img_h], dtype=np.float32)

        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()

            # Get Sapiens prediction if available
            sapiens_kp = None
            if use_sapiens:
                result = run_sapiens_on_bbox(img_np, whole_bbox, sapiens)
                if result is not None:
                    sapiens_kp = result["pixel_kp"]  # (133, 3)

            if use_3d and use_sapiens:
                # Both: body+feet from 3D, face+hands from Sapiens
                for p_idx in range(n_persons):
                    j2d = pose_3d["persons"][p_idx]["body_joints2d"][t]
                    if j2d is not None:
                        kp = fuse_3d_body_with_sapiens(j2d, sapiens_kp)
                        frame_kps[t].append(kp)
                    elif sapiens_kp is not None:
                        frame_kps[t].append(sapiens_kp.copy())

            elif use_3d:
                # 3D only: body+feet from PromptHMR, no face/hands
                for p_idx in range(n_persons):
                    j2d = pose_3d["persons"][p_idx]["body_joints2d"][t]
                    if j2d is not None:
                        kp = openpose25_to_coco_wholebody(j2d)
                        frame_kps[t].append(kp)

            elif use_sapiens:
                # Sapiens only: full 133-keypoint prediction
                if sapiens_kp is not None:
                    frame_kps[t].append(sapiens_kp.copy())

            pbar.update(1)

        # -----------------------------------------------------------
        # Frame rate resampling (per-person linear interpolation)
        # -----------------------------------------------------------
        target_fps = _FPS_MAP.get(frame_rate)
        do_resample = (target_fps is not None
                       and fps > 0
                       and abs(fps - target_fps) > 0.1)

        output_fps = float(target_fps) if do_resample else float(fps)

        if do_resample:
            # Convert frame_kps[t][p] → per-person timelines for interpolation
            per_person = []
            for p in range(n_persons):
                timeline = []
                for t in range(B):
                    if p < len(frame_kps[t]):
                        timeline.append(frame_kps[t][p])
                    else:
                        timeline.append(None)
                per_person.append(timeline)

            # Resample each person's timeline independently
            resampled_persons = []
            src_indices = None
            for p in range(n_persons):
                resampled, s_idx = resample_keypoints(
                    per_person[p], fps, target_fps)
                resampled_persons.append(resampled)
                if src_indices is None:
                    src_indices = s_idx

            # Recombine into per-frame structure
            n_out = len(src_indices)
            frame_kps_out = [[] for _ in range(n_out)]
            for t in range(n_out):
                for p in range(n_persons):
                    if resampled_persons[p][t] is not None:
                        frame_kps_out[t].append(resampled_persons[p][t])
        else:
            frame_kps_out = frame_kps
            src_indices = list(range(B))
            n_out = B

        # -----------------------------------------------------------
        # Pass 2: render
        # -----------------------------------------------------------
        pose_images = []
        for out_t in range(n_out):
            src_t = src_indices[out_t]
            if debug:
                canvas = (images_nchw[src_t].permute(1, 2, 0)
                          * 255).byte().numpy().copy()
            else:
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            for kp in frame_kps_out[out_t]:
                canvas = render_sapiens_dwpose(canvas, kp, img_h, img_w)

            pose_images.append(
                torch.from_numpy(canvas.astype(np.float32) / 255.0))
            pbar.update(1)

        return (torch.stack(pose_images), output_fps)
