"""
Pose Renderer — render the COCO-WB 133-keypoint 2D timeline as a
DWPose-style skeleton image.

History note: this node used to fuse 3D body (projected to 2D) with 2D
face/hands. That mattered when the 3D backbone (PromptHMR / NLF) was
the only quality body source. With BMPRTMWPose now producing
high-quality 133-keypoint 2D per frame, projecting 3D back to 2D only
adds error (3D model error + estimated camera intrinsics + projection
rounding), so the renderer now uses ``person["keypoints"][t]``
directly. Frames where ``keypoints[t]`` is None are intentionally not
drawn — that signal is set upstream when no body source survived
(BMPRTMWPose Phase 2.8) and faking it from a less-trusted source would
hide real gaps.

Supports debug overlay, frame-rate resampling, and toggling face /
hand+foot visibility. Only renders persons with visible=True.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_sapiens import render_sapiens_dwpose
from ._pose_utils import is_face_visible, resample_keypoints


class PoseRendererNode:
    """Render COCO-WB 133-keypoint 2D timelines as DWPose-style images."""

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
                            "Show face keypoints (COCO-WB slots 23..90). "
                            "When False those slots are zeroed before "
                            "rendering."
                        ),
                    },
                ),
                "face_smart_filter": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When show_face=True, additionally skip "
                            "drawing face landmarks for (slot, frame) "
                            "pairs where the face appears NOT to be "
                            "visible to camera (e.g. person turned "
                            "around). Three-tier check: (1) 3D body "
                            "normal from smpl_j3d shoulders × torso "
                            "— used when Pose3DUpgrade was run; works "
                            "even for upside-down poses, (2) RTMW/"
                            "ViTPose pre-FaRL face mean confidence "
                            "stashed by BMPRTMWPose — strong data-"
                            "driven signal, (3) 2D head-kpt geometry "
                            "fallback (no upright assumption). Disable "
                            "for content where you'd rather always "
                            "render whatever FaRL produced."
                        ),
                    },
                ),
                "show_hand_foot": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Show hand (91..132) and foot (17..22) "
                            "keypoints. When False both ranges are "
                            "zeroed and only body+head (0..16) is "
                            "rendered."
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
               show_face, face_smart_filter, show_hand_foot, images=None):

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
        #
        # Read directly from person["keypoints"][t] — that's the
        # 133-keypoint 2D output composed by BMPRTMWPose (BMP body +
        # RTMW feet + FaRL face + WiLoR hands + ViTPose fallback) or
        # an equivalent 2D pose node. No projection from 3D, no
        # cross-source fusion: a single 2D source is more accurate
        # than projecting 3D back through estimated intrinsics.
        # Frames where keypoints[t] is None are silently skipped —
        # that's the upstream node telling us "no body source
        # survived this (slot, frame); don't fake one."
        # -----------------------------------------------------------
        frame_kps = [[] for _ in range(B)]
        face_filtered = 0  # counts (slot, frame) where smart filter zeroed face

        for t in range(B):
            for p_idx in visible_indices:
                kp133 = poses["persons"][p_idx]["keypoints"][t]
                if kp133 is None:
                    continue
                kp = kp133.copy()
                if not show_face:
                    kp[23:91] = 0.0          # face 68pt
                elif face_smart_filter:
                    # 3-tier visibility check: 3D body normal → RTMW
                    # face conf → 2D head-kpt geometry fallback. Zero
                    # face slots when face appears not visible (back
                    # view / occluded).
                    if not is_face_visible(poses, p_idx, t):
                        kp[23:91] = 0.0
                        face_filtered += 1
                if not show_hand_foot:
                    kp[17:23]  = 0.0         # feet 6pt
                    kp[91:133] = 0.0         # hands 42pt
                frame_kps[t].append(kp)

            pbar.update(1)

        if show_face and face_smart_filter and face_filtered > 0:
            # Diagnostic visibility — at-a-glance "how many faces did
            # the smart filter hide". Useful for verifying it caught
            # back-view frames without over-rejecting frontal ones.
            import logging
            logging.getLogger(__name__).info(
                "PoseRenderer: face_smart_filter zeroed face on "
                "%d (slot, frame) pairs (back-view / occluded).",
                face_filtered,
            )

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
