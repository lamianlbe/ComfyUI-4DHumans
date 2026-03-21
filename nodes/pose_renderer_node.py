"""
Sapiens PromptHMR Pose Renderer node.

Fuses body keypoints from 3D data with face/hand keypoints from 2D data
within a unified POSES dict and renders DWPose-style skeleton images.
Supports debug overlay, frame rate resampling, and toggling face /
hand+foot visibility.  Only renders persons with visible=True.
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
                "poses": ("POSES",),
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

    def render(self, poses, debug, target_fps,
               show_face, show_hand_foot, images=None):

        n_persons = poses["n_persons"]
        B = poses["n_frames"]
        img_h = poses["img_h"]
        img_w = poses["img_w"]
        fps = poses["fps"]

        if debug and images is None:
            raise ValueError(
                "Pose Renderer: 'images' must be connected when debug=True."
            )

        if images is not None:
            images_nchw = images.permute(0, 3, 1, 2)

        # Collect visible person indices
        visible_indices = [
            p for p in range(n_persons) if poses["persons"][p].get("visible", True)
        ]
        n_visible = len(visible_indices)

        pbar = comfy.utils.ProgressBar(2 * B)

        # -----------------------------------------------------------
        # Pass 1: build per-frame, per-person COCO-WB keypoints
        # -----------------------------------------------------------
        frame_kps = [[] for _ in range(B)]

        for t in range(B):
            for p_idx in visible_indices:
                person = poses["persons"][p_idx]
                j2d = person["body_joints2d"][t]
                sapiens_kp = person["keypoints"][t]

                if j2d is not None:
                    kp = fuse_3d_body_with_sapiens(
                        j2d, sapiens_kp,
                        show_face=show_face,
                        show_hand_foot=show_hand_foot,
                    )
                    frame_kps[t].append(kp)
                elif sapiens_kp is not None and show_face and show_hand_foot:
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
            for vp in range(n_visible):
                timeline = []
                for t in range(B):
                    if vp < len(frame_kps[t]):
                        timeline.append(frame_kps[t][vp])
                    else:
                        timeline.append(None)
                per_person.append(timeline)

            resampled_persons = []
            src_indices = None
            for vp in range(n_visible):
                resampled, s_idx = resample_keypoints(
                    per_person[vp], fps, target_fps)
                resampled_persons.append(resampled)
                if src_indices is None:
                    src_indices = s_idx

            n_out = len(src_indices)
            frame_kps_out = [[] for _ in range(n_out)]
            for t in range(n_out):
                for vp in range(n_visible):
                    if resampled_persons[vp][t] is not None:
                        frame_kps_out[t].append(resampled_persons[vp][t])
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
