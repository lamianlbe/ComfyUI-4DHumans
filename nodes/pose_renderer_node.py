"""
Sapiens PromptHMR Pose Renderer node.

Fuses body keypoints from POSE_3D with face/hand keypoints from POSE_2D
and renders DWPose-style skeleton images.  Supports debug overlay,
frame rate resampling, and toggling face / hand+foot visibility.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ._pose_utils import (
    openpose25_to_coco_wholebody,
    fuse_3d_body_with_sapiens,
    resample_keypoints,
)


class PoseRendererNode:
    """Render fused 2D+3D pose skeletons as DWPose-style images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_2d": ("POSE_2D",),
                "pose_3d": ("POSE_3D",),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": "Source video FPS.",
                    },
                ),
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
                "target_fps": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": (
                            "Output frame rate. 0 keeps source fps. "
                            "Any value 1-30 resamples with interpolation."
                        ),
                    },
                ),
                "show_face": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Show face keypoints from Sapiens. "
                            "When False only PromptHMR body keypoints are shown."
                        ),
                    },
                ),
                "show_hand_foot": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Show hand & foot keypoints from Sapiens. "
                            "When False only PromptHMR body keypoints are shown."
                        ),
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("pose_images", "fps")
    FUNCTION = "render"
    CATEGORY = "4dhumans"

    def render(self, pose_2d, pose_3d, fps, debug, target_fps,
               show_face, show_hand_foot, images=None):

        n_persons = pose_3d["n_persons"]
        B = pose_3d["n_frames"]
        img_h = pose_3d["img_h"]
        img_w = pose_3d["img_w"]

        if debug and images is None:
            raise ValueError(
                "Pose Renderer: 'images' must be connected when debug=True."
            )

        if images is not None:
            images_nchw = images.permute(0, 3, 1, 2)

        pbar = comfy.utils.ProgressBar(2 * B)

        # -----------------------------------------------------------
        # Pass 1: build per-frame, per-person COCO-WB keypoints
        # -----------------------------------------------------------
        frame_kps = [[] for _ in range(B)]

        for t in range(B):
            for p_idx in range(n_persons):
                # Body from POSE_3D (OpenPose 25 2D joints)
                j2d = pose_3d["persons"][p_idx]["body_joints2d"][t]

                # Face/hands from POSE_2D (COCO-WB 133)
                sapiens_kp = pose_2d["persons"][p_idx]["keypoints"][t]

                if j2d is not None:
                    kp = fuse_3d_body_with_sapiens(
                        j2d, sapiens_kp,
                        show_face=show_face,
                        show_hand_foot=show_hand_foot,
                    )
                    frame_kps[t].append(kp)
                elif sapiens_kp is not None and show_face and show_hand_foot:
                    # Fallback: use full Sapiens prediction
                    frame_kps[t].append(sapiens_kp.copy())

            pbar.update(1)

        # -----------------------------------------------------------
        # Frame rate resampling (per-person linear interpolation)
        # -----------------------------------------------------------
        do_resample = (target_fps >= 1.0
                       and fps > 0
                       and abs(fps - target_fps) > 0.1)

        output_fps = float(target_fps) if do_resample else float(fps)

        if do_resample:
            per_person = []
            for p in range(n_persons):
                timeline = []
                for t in range(B):
                    if p < len(frame_kps[t]):
                        timeline.append(frame_kps[t][p])
                    else:
                        timeline.append(None)
                per_person.append(timeline)

            resampled_persons = []
            src_indices = None
            for p in range(n_persons):
                resampled, s_idx = resample_keypoints(
                    per_person[p], fps, target_fps)
                resampled_persons.append(resampled)
                if src_indices is None:
                    src_indices = s_idx

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
            if debug and images is not None:
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
