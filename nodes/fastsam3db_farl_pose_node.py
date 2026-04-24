"""
FastSAM3DBodyFaRLPose node — Fast SAM 3D Body + RetinaFace + FaRL face
+ (optional) YOLO11m-Pose.

Output is a POSES dict fully compatible with the existing NPZ format
so downstream nodes (Pose Editor, Save/Load, Renderer, NLF converter)
work unchanged.

Face stack (pyfacer, MIT) replaces the previous RTMPose-Face:

  * RetinaFace MobileNet-0.25 runs per-frame to produce real 5-point
    face landmarks (any pose, any roll — no synthesis from MHR head
    kps, which only worked for frontal/upright).
  * FaRL ViT-B consumes those 5 points, computes a similarity
    alignment to a canonical 448 px template, and regresses 68
    landmarks in the standard 300W/iBUG convention — a direct match
    for COCO-WholeBody indices 23..90 with zero remapping.
  * Detected faces are assigned to tracked persons by testing the
    face bbox center against each person's segmentation mask (greedy,
    score-sorted) so per-person identities stay stable across frames.

Phases:
  Phase 0: input validation + reshaping (masks, images)
  Phase 1: per-frame mask → bbox + sampling strategy for pose_fps
  Phase 2: Fast SAM 3D Body + MHR2SMPL (per-person sequential smoother)
  Phase 3: linear interpolation of skipped frames (pose_fps < fps)
  Phase 4: assemble COCO-WB body+feet+hands from MHR
  Phase 5: RetinaFace + FaRL fill COCO-WB face slots 23..90 per person
  Phase 6: store keypoints_raw snapshot for Pose Editor's non-destructive edits
  Phase 7: pack the POSES dict
"""

import logging

import numpy as np
import torch

import comfy.utils

from ._fastsam3db_inference import (
    run_fastsam3db_video,
    mhr70_to_coco_wb_body_feet_hands,
)
from ._farl_face_inference import run_farl_face_video

_logger = logging.getLogger(__name__)


def _linear_interp_timeline(timeline):
    """In-place fill None entries by linearly interpolating between
    the nearest non-None neighbours.  Mirrors the helper already used
    in sapiens_prompthmr_pose_node.py."""
    valid = [i for i, v in enumerate(timeline) if v is not None]
    if len(valid) < 2:
        return
    for a, b in zip(valid[:-1], valid[1:]):
        if b - a <= 1:
            continue
        va = timeline[a]
        vb = timeline[b]
        for i in range(a + 1, b):
            alpha = (i - a) / (b - a)
            timeline[i] = va * (1.0 - alpha) + vb * alpha


class FastSAM3DBodyFaRLPoseNode:
    """Fast SAM 3D Body + FaRL face pose estimator.

    Emits the same POSES dict as SapiensPromptHMRPose but using a
    different backbone stack.  Downstream nodes (Renderer, Editor,
    NLF converter, Save/Load) remain compatible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "fast_sam_3d_body": ("FASTSAM3DBODY",),
                "farl_face": ("FARLFACE",),
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
                "face_frame_batch_size": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "How many frames to feed FaRL per batch. "
                            "Larger = faster but more GPU memory. "
                            "32 is a reasonable default for 512×960 "
                            "frames on a 24 GB GPU."
                        ),
                    },
                ),
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
                            "FaRL always runs per-frame."
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
    FUNCTION = "estimate_pose"
    CATEGORY = "4dhumans"

    def estimate_pose(self, images, masks, fast_sam_3d_body, farl_face,
                      fps, face_frame_batch_size=32, pose_fps=15.0,
                      yolo11_pose=None):
        # ---------------------------------------------------------------
        # Phase 0: input validation + reshape
        # ---------------------------------------------------------------
        B, img_h, img_w, C = images.shape
        rgb = images[..., :3]

        if masks.dim() == 4:
            masks = masks[..., 0]
        M = masks.shape[0]
        if M % B != 0:
            raise ValueError(
                f"FastSAM3DBody+FaRL Pose: mask count ({M}) must be a "
                f"multiple of frame count ({B})."
            )
        n_persons = M // B
        masks = masks.reshape(B, n_persons, masks.shape[-2], masks.shape[-1])
        masks_np = masks.cpu().numpy() > 0.5  # (B, N, H, W) bool
        images_np_u8 = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

        _logger.info(
            "FastSAM3DBody+FaRL: %d frames, %d persons, %dx%d, "
            "pose_fps=%.1f (src %.1f)",
            B, n_persons, img_w, img_h, pose_fps, fps,
        )

        # ---------------------------------------------------------------
        # Phase 1: per-frame mask bbox + pose_fps sampling decision
        # ---------------------------------------------------------------
        # Stride for 3D backbone. Always include first + last frame so
        # the interpolation has endpoints.
        if pose_fps > 0 and pose_fps < fps:
            phmr_stride = max(1, int(round(float(fps) / float(pose_fps))))
        else:
            phmr_stride = 1
        sampled_frames = set(range(0, B, phmr_stride))
        if B > 0:
            sampled_frames.add(B - 1)

        mask_bboxes_per_frame = [[] for _ in range(B)]
        person_indices_per_frame = [[] for _ in range(B)]

        # Per-frame bbox summary so multi-person regressions are easy to
        # eyeball. For each frame we log:  t | p_idx [mask_px] (x1,y1,x2,y2) w×h
        # Skipped persons (empty mask / degenerate bbox) also get a line
        # with the reason so you can tell "person 1 missing on frame 42"
        # from "person 1 present but too small".
        _logger.info(
            "Phase 1 per-frame bbox dump (B=%d, n_persons=%d, stride=%d):",
            B, n_persons, phmr_stride,
        )
        for t in range(B):
            if t not in sampled_frames:
                continue
            frame_rows = []
            for p_idx in range(n_persons):
                mask_frame = masks_np[t, p_idx]
                mask_px = int(mask_frame.sum())
                ys, xs = np.where(mask_frame)
                if len(xs) == 0:
                    frame_rows.append(f"p{p_idx}:EMPTY")
                    continue
                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max() + 1)
                y2 = int(ys.max() + 1)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                w = x2 - x1
                h = y2 - y1
                if w < 2 or h < 2:
                    frame_rows.append(
                        f"p{p_idx}:TINY({x1},{y1},{x2},{y2}) {w}x{h}"
                    )
                    continue
                mask_bboxes_per_frame[t].append((x1, y1, x2, y2))
                person_indices_per_frame[t].append(p_idx)
                frame_rows.append(
                    f"p{p_idx}:[{mask_px}px]({x1},{y1},{x2},{y2}) {w}x{h}"
                )
            _logger.info("  frame %4d | %s", t, " | ".join(frame_rows))

        n_sampled = len(sampled_frames)
        pbar = comfy.utils.ProgressBar(B * 2)

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

        scale_per_frame = [1.0 if cam_int_per_frame[t] is not None else None
                           for t in range(B)]
        offset_per_frame = [np.zeros(2, dtype=np.float64)
                            if cam_int_per_frame[t] is not None else None
                            for t in range(B)]

        # ---------------------------------------------------------------
        # Phase 3: Linear interpolation on skipped frames
        # ---------------------------------------------------------------
        if phmr_stride > 1:
            for p_idx in range(n_persons):
                _linear_interp_timeline(persons_fs[p_idx]["body_joints2d"])
                _linear_interp_timeline(persons_fs[p_idx]["body_joints"])
                _linear_interp_timeline(persons_fs[p_idx]["smpl_j3d"])
                _linear_interp_timeline(persons_fs[p_idx]["mhr_kp2d"])
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

        # ---------------------------------------------------------------
        # Phase 4: assemble COCO-WB body + feet + hands from MHR
        # ---------------------------------------------------------------
        persons_coco_wb_133 = [
            [None] * B for _ in range(n_persons)
        ]
        for p_idx in range(n_persons):
            for t in range(B):
                mhr_kp2d = persons_fs[p_idx]["mhr_kp2d"][t]
                if mhr_kp2d is None:
                    continue
                if mhr_kp2d.shape[-1] == 2:
                    kp2d = np.concatenate(
                        [mhr_kp2d, np.ones((mhr_kp2d.shape[0], 1), dtype=np.float32)],
                        axis=-1,
                    )
                else:
                    kp2d = mhr_kp2d.astype(np.float32)

                body_feet, rhand, lhand = mhr70_to_coco_wb_body_feet_hands(kp2d)
                coco_wb = np.zeros((133, 3), dtype=np.float32)
                coco_wb[:23] = body_feet           # 0..22  body + feet
                coco_wb[91:112] = lhand             # 91..111 left hand
                coco_wb[112:133] = rhand            # 112..132 right hand
                persons_coco_wb_133[p_idx][t] = coco_wb

        # ---------------------------------------------------------------
        # Phase 5: RetinaFace + FaRL for 68 face landmarks (COCO-WB 23..90)
        #
        # RetinaFace detects faces per frame (any pose/roll), and each
        # face is matched to a tracked person by testing the face bbox
        # center against that person's mask. FaRL then regresses the 68
        # iBUG landmarks from the matched 5-point detections.
        # ---------------------------------------------------------------
        face_kp_68_timeline, farl_time = run_farl_face_video(
            images_np_u8=images_np_u8,
            masks_np=masks_np,
            farl_face_dict=farl_face,
            n_persons=n_persons,
            img_h=img_h,
            img_w=img_w,
            frame_batch_size=face_frame_batch_size,
            pbar=pbar,
        )

        # Fill face slice 23..90 into the 133-layout
        for p_idx in range(n_persons):
            for t in range(B):
                face68 = face_kp_68_timeline[p_idx][t]
                kp133 = persons_coco_wb_133[p_idx][t]
                if face68 is None or kp133 is None:
                    continue
                kp133[23:91] = face68  # 68 face keypoints in 300W order

        # ---------------------------------------------------------------
        # Phase 6: assemble POSES "persons" list (with keypoints_raw snapshot)
        # ---------------------------------------------------------------
        poses_persons = []
        for p_idx in range(n_persons):
            kp_main = persons_coco_wb_133[p_idx]
            kp_raw = [kp.copy() if kp is not None else None for kp in kp_main]
            poses_persons.append({
                "visible": True,
                "body_joints2d": persons_fs[p_idx]["body_joints2d"],
                "body_joints":   persons_fs[p_idx]["body_joints"],
                "smpl_j3d":      persons_fs[p_idx]["smpl_j3d"],
                "keypoints":     kp_main,
                "keypoints_raw": kp_raw,
            })

        # ---------------------------------------------------------------
        # Phase 7: pack
        # ---------------------------------------------------------------
        poses = {
            "n_persons": n_persons,
            "n_frames": B,
            "img_h": int(img_h),
            "img_w": int(img_w),
            "fps": float(fps),
            "persons": poses_persons,
            "cam_int": cam_int_per_frame,
            "scale": scale_per_frame,
            "offset": offset_per_frame,
        }

        total = fastsam3db_time + mhr2smpl_time + farl_time
        _logger.info(
            "FastSAM3DBody+FaRL Pose: %d frames, %d persons, "
            "pose_fps=%.1f stride=%d | "
            "FastSAM3DBody %.2fs (%d sampled frames) | "
            "MHR2SMPL %.2fs | "
            "RetinaFace+FaRL %.2fs | total %.2fs",
            B, n_persons, pose_fps, phmr_stride,
            fastsam3db_time, n_sampled,
            mhr2smpl_time,
            farl_time, total,
        )

        return (poses,)
