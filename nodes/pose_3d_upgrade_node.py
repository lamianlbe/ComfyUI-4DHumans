"""
Pose3DUpgrade — augment a 2D POSES dict with 3D body data from
Fast SAM 3D Body, producing a POSES dict that's fully compatible
with the legacy NPZ schema (body_joints2d / body_joints / smpl_j3d
/ cam_int / scale / offset).

Use case: BMPRTMWPose produces high-quality 2D 133-keypoint POSES
but its 3D fields are all None. Downstream tools (PoseRenderer,
4D-Humans mesh renderer, NLF converter) expect those fields to be
populated. This node runs the 3D backbone from Fast SAM 3D Body
on the same masks and merges the 3D output back into the input
POSES — preserving the 2D keypoints intact, only filling fields
that were None.

Pipeline:

    images ─┬─► [BMPInstanceSeg] ──┬─► masks ──┐
            │                       │           │
            │                       └─► bmp_pose │  ┌──► [BMP+RTMW Pose] ──► poses (2D, 133pt) ──┐
            └───────────────────────────────────┴──┤                                              │
                                                    │                                              │
                                                    │              ┌─────────────────────────────┘
                                                    ▼              ▼
                                              [Pose 3D Upgrade] ◄── poses
                                              ◄── bmp_masks
                                              ◄── images
                                              ◄── fast_sam_3d_body
                                                    │
                                                    ▼
                                          POSES (2D + 3D, NPZ-compatible)

Slot ownership in the merged POSES:

    keypoints       (133, 3)  ← INPUT POSES (BMP / RTMW / FaRL / WiLoR / ViTPose
                                              composite from BMPRTMWPose)
    keypoints_raw   (133, 3)  ← INPUT POSES (preserved snapshot)
    body_joints2d   (~25, 2)  ← Fast SAM 3D Body (MHR-projected 2D)
    body_joints     (~25, 3)  ← Fast SAM 3D Body (3D body joints in cam frame)
    smpl_j3d        (24,  3)  ← Fast SAM 3D Body (SMPL-24 joints in cam frame)
    cam_int         (3, 3)    ← Fast SAM 3D Body (estimated/default intrinsics)
    scale, offset             ← Fast SAM 3D Body crop affine

Note: ``body_joints2d`` is intentionally MHR-projected (not COCO-WB)
because that's what the legacy NPZ schema stores there. Code that
reads NPZ ``body_joints2d`` already expects MHR projection. The
high-quality 133-pt 2D output stays in the ``keypoints`` field
where downstream consumers find it.
"""

import logging
import time
from typing import List, Optional

import numpy as np
import torch

import comfy.utils

from .fastsam3db_farl_pose_node import _linear_interp_timeline
from ._fastsam3db_inference import run_fastsam3db_video

_logger = logging.getLogger(__name__)


class Pose3DUpgradeNode:
    """Take 2D POSES + masks + Fast SAM 3D Body → output augmented
    POSES with full 3D fields filled. 2D keypoints are preserved
    unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "poses":             ("POSES",),
                "bmp_masks":         ("MASK",),
                "images":            ("IMAGE",),
                "fast_sam_3d_body":  ("FASTSAM3DBODY",),
                "pose_fps": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": (
                            "Target FPS for Fast SAM 3D Body inference. "
                            "Intermediate frames are linearly interpolated. "
                            "Lower = faster, but jittery 3D motion. "
                            "15 is a reasonable balance for 24-30 fps "
                            "source video."
                        ),
                    },
                ),
            },
            "optional": {
                "yolo11_pose": ("YOLO11POSE",),
            },
        }

    RETURN_TYPES = ("POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "upgrade"
    CATEGORY = "4dhumans"

    def upgrade(self, poses, bmp_masks, images, fast_sam_3d_body,
                pose_fps=15.0, yolo11_pose=None):
        # ---------------------------------------------------------------
        # Phase 0: validate alignment between POSES and masks
        # ---------------------------------------------------------------
        n_persons_poses = int(poses.get("n_persons", 0))
        n_frames_poses  = int(poses.get("n_frames", 0))
        img_h_poses     = int(poses.get("img_h", 0))
        img_w_poses     = int(poses.get("img_w", 0))
        fps             = float(poses.get("fps", 30.0))

        B, img_h, img_w, _C = images.shape
        if (img_h != img_h_poses) or (img_w != img_w_poses):
            _logger.warning(
                "Pose3DUpgrade: images shape (%d, %d) doesn't match "
                "POSES img_h/img_w (%d, %d). Using images shape as "
                "ground truth.",
                img_h, img_w, img_h_poses, img_w_poses,
            )
        if B != n_frames_poses and n_frames_poses > 0:
            _logger.warning(
                "Pose3DUpgrade: image frame count (%d) doesn't match "
                "POSES n_frames (%d).",
                B, n_frames_poses,
            )

        # Reshape masks to (B, n_persons, H, W)
        if bmp_masks.dim() == 4:
            bmp_masks = bmp_masks[..., 0]
        M = bmp_masks.shape[0]
        if M == 0 or B == 0:
            _logger.warning(
                "Pose3DUpgrade: empty masks or images — returning "
                "input POSES unchanged."
            )
            return (poses,)
        if M % B != 0:
            raise ValueError(
                f"Pose3DUpgrade: mask count ({M}) is not a multiple "
                f"of frame count ({B})."
            )
        n_persons_masks = M // B
        if n_persons_masks != n_persons_poses:
            _logger.warning(
                "Pose3DUpgrade: mask slots (%d) ≠ POSES persons (%d). "
                "Using mask slot count for 3D inference; POSES persons "
                "list will be padded/truncated as needed.",
                n_persons_masks, n_persons_poses,
            )
        n_persons = n_persons_masks
        masks = bmp_masks.reshape(B, n_persons, bmp_masks.shape[-2],
                                   bmp_masks.shape[-1])
        masks_np = masks.cpu().numpy() > 0.5  # (B, N, H, W) bool

        rgb = images[..., :3]
        images_np_u8 = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

        _logger.info(
            "Pose3DUpgrade: %d frames, %d persons (mask slots), %dx%d, "
            "pose_fps=%.1f (src %.1f)",
            B, n_persons, img_w, img_h, pose_fps, fps,
        )

        # ---------------------------------------------------------------
        # Phase 1: per-frame mask → bbox + pose_fps sampling
        # ---------------------------------------------------------------
        if pose_fps > 0 and pose_fps < fps:
            phmr_stride = max(1, int(round(float(fps) / float(pose_fps))))
        else:
            phmr_stride = 1
        sampled_frames = set(range(0, B, phmr_stride))
        if B > 0:
            sampled_frames.add(B - 1)

        mask_bboxes_per_frame: List[list] = [[] for _ in range(B)]
        person_indices_per_frame: List[list] = [[] for _ in range(B)]

        for t in range(B):
            if t not in sampled_frames:
                continue
            for p_idx in range(n_persons):
                mask_frame = masks_np[t, p_idx]
                ys, xs = np.where(mask_frame)
                if len(xs) == 0:
                    continue
                x1 = int(xs.min()); y1 = int(ys.min())
                x2 = int(xs.max() + 1); y2 = int(ys.max() + 1)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                mask_bboxes_per_frame[t].append((x1, y1, x2, y2))
                person_indices_per_frame[t].append(p_idx)

        n_sampled = len(sampled_frames)
        pbar = comfy.utils.ProgressBar(B + 1)
        t0 = time.perf_counter()

        # ---------------------------------------------------------------
        # Phase 2: Fast SAM 3D Body + MHR2SMPL
        # ---------------------------------------------------------------
        result = run_fastsam3db_video(
            images_np_u8=images_np_u8,
            mask_bboxes_per_frame=mask_bboxes_per_frame,
            masks_np=masks_np,
            person_indices_per_frame=person_indices_per_frame,
            fastsam3db_dict=fast_sam_3d_body,
            yolo11pose_dict=yolo11_pose,
            n_persons=n_persons,
            img_h=img_h,
            img_w=img_w,
            pbar=pbar,
        )
        persons_fs = result["persons"]
        cam_int_per_frame = result["cam_int"]
        fastsam3db_time = result["fastsam3db_time_s"]
        mhr2smpl_time = result["mhr2smpl_time_s"]

        scale_per_frame = [
            1.0 if cam_int_per_frame[t] is not None else None
            for t in range(B)
        ]
        offset_per_frame = [
            np.zeros(2, dtype=np.float64)
            if cam_int_per_frame[t] is not None else None
            for t in range(B)
        ]

        # ---------------------------------------------------------------
        # Phase 3: linear interpolate 3D timelines for skipped frames
        # ---------------------------------------------------------------
        if phmr_stride > 1:
            for p_idx in range(n_persons):
                _linear_interp_timeline(persons_fs[p_idx]["body_joints2d"])
                _linear_interp_timeline(persons_fs[p_idx]["body_joints"])
                _linear_interp_timeline(persons_fs[p_idx]["smpl_j3d"])
                # mhr_kp2d not used downstream of this node, skip
                # (FastSAM3DBodyFaRLPose interpolated it because FaRL
                # consumed it; we don't run FaRL here.)
            _linear_interp_timeline(cam_int_per_frame)
            scale_arr = [
                np.array([v], dtype=np.float64) if v is not None else None
                for v in scale_per_frame
            ]
            _linear_interp_timeline(scale_arr)
            scale_per_frame = [
                float(v[0]) if v is not None else None for v in scale_arr
            ]
            _linear_interp_timeline(offset_per_frame)

        pbar.update(1)

        # ---------------------------------------------------------------
        # Phase 4: merge 3D into input POSES
        #
        # Take the input POSES dict as the base — preserves 2D
        # keypoints exactly. Then overlay 3D body + cam fields. If the
        # input POSES had fewer persons than n_persons_masks, pad with
        # blank entries; if it had more (which shouldn't happen given
        # they share the bmp_masks source), truncate.
        # ---------------------------------------------------------------
        in_persons = list(poses.get("persons", []))
        merged_persons = []
        body_filled  = 0
        smpl_filled  = 0
        for p_idx in range(n_persons):
            if p_idx < len(in_persons):
                base_person = dict(in_persons[p_idx])  # shallow copy
            else:
                # No matching input slot — synthesise a blank one so
                # the output schema is consistent.
                base_person = {
                    "visible": True,
                    "body_joints2d": [None] * B,
                    "body_joints":   [None] * B,
                    "smpl_j3d":      [None] * B,
                    "keypoints":     [None] * B,
                    "keypoints_raw": [None] * B,
                }

            # Overlay 3D fields from Fast SAM 3D Body output.
            base_person["body_joints2d"] = persons_fs[p_idx]["body_joints2d"]
            base_person["body_joints"]   = persons_fs[p_idx]["body_joints"]
            base_person["smpl_j3d"]      = persons_fs[p_idx]["smpl_j3d"]
            body_filled += sum(
                1 for v in persons_fs[p_idx]["body_joints"] if v is not None
            )
            smpl_filled += sum(
                1 for v in persons_fs[p_idx]["smpl_j3d"] if v is not None
            )

            merged_persons.append(base_person)

        merged_poses = {
            "n_persons": n_persons,
            "n_frames":  B,
            "img_h":     int(img_h),
            "img_w":     int(img_w),
            "fps":       float(fps),
            "persons":   merged_persons,
            "cam_int":   cam_int_per_frame,
            "scale":     scale_per_frame,
            "offset":    offset_per_frame,
        }

        # Forward any optional metadata from the input POSES (e.g. the
        # filter parameters Pose Editor stashes there for non-
        # destructive restore).
        for key in ("_filter_velocity_threshold", "_filter_smooth_sigma"):
            if key in poses:
                merged_poses[key] = poses[key]

        elapsed = time.perf_counter() - t0
        cam_filled = sum(1 for v in cam_int_per_frame if v is not None)
        _logger.info(
            "Pose3DUpgrade: %d frames, %d persons | FastSAM3DBody "
            "%.2fs (%d sampled) | MHR2SMPL %.2fs | body_joints "
            "filled: %d (slot,frame) | smpl_j3d filled: %d | "
            "cam_int filled: %d | total %.2fs",
            B, n_persons, fastsam3db_time, n_sampled,
            mhr2smpl_time, body_filled, smpl_filled,
            cam_filled, elapsed,
        )

        return (merged_poses,)
