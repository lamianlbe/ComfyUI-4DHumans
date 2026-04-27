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
            │                       └─► bmp_pose │  ┌──► [BMP+RTMW Pose] ──► pose_bundle ──┐
            └───────────────────────────────────┴──┤                                       │
                                                    │                                       │
                                                    ▼                                       ▼
                                              [Pose 3D Upgrade] ◄────────────── pose_bundle
                                              ◄── fast_sam_3d_body
                                                    │
                                                    ▼
                                          POSES (2D + 3D, NPZ-compatible)

The ``pose_bundle`` is the *single* contract — it carries the 2D
``poses`` dict, the ``bmp_masks`` they were produced from, the
``images``, and the pre-computed ``sampled_frames`` schedule the
upstream node chose based on its ``fps`` / ``pose_3d_fps`` settings.
This eliminates the previous "two nodes, two fps inputs that can
disagree" foot-gun.

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
from .bmp_seg_node import _BMPResult, _BMPReplayModel

_logger = logging.getLogger(__name__)


def _yolo_pose_from_input_poses(
    input_poses: dict,
    bbox_by_slot: List[List[Optional[np.ndarray]]],
    n_persons: int,
    B: int,
) -> dict:
    """Build a YOLO11POSE-compatible dict from the input POSES'
    composite 2D keypoints, so Fast SAM 3D Body can use the user's
    BMP/ViTPose/RTMW-fused wrist data as ``yolo_pose_keypoints``
    without requiring a separate LoadYOLO11Pose connection.

    The output dict matches the shape FastSAM3DBody's
    ``yolo11_pose["model"]`` consumer expects: a callable model with
    ``.to(device)`` + ``.predict(source=chunk)`` returning
    Ultralytics-Result-like objects (each with ``.boxes`` and
    ``.keypoints``). We reuse the same replay-model machinery
    BMPInstanceSegmentation already uses for ``bmp_pose``.

    bbox source preference per (slot, frame):
      1. mask-derived bbox from ``bbox_by_slot[t][p]`` (preferred —
         tight body extent matches BMP's IoU tracking)
      2. derived from visible body keypoints (fallback for slots/
         frames where BMP missed the mask but ViTPose's full-image
         fallback fired in BMPRTMWPose; we still have keypoints,
         just no mask)
      3. skip the slot for that frame (not enough data)
    """
    persons_in = input_poses.get("persons", [])
    cache: List[Optional[_BMPResult]] = [None] * B

    for t in range(B):
        bboxes  = []
        scores  = []
        kpts17s = []
        for p_idx in range(n_persons):
            if p_idx >= len(persons_in):
                continue
            kp_list = persons_in[p_idx].get("keypoints", [])
            if t >= len(kp_list):
                continue
            kp133 = kp_list[t]
            if kp133 is None:
                continue
            kp17 = np.asarray(kp133[:17, :3], dtype=np.float32)

            # Pick bbox: prefer mask-derived, else derive from visible body kpts
            bb = bbox_by_slot[t][p_idx]
            if bb is None:
                vis = kp17[:, 2] > 0.1
                if int(vis.sum()) < 3:
                    continue   # too few visible joints to make a meaningful bbox
                xs = kp17[vis, 0]
                ys = kp17[vis, 1]
                pad_x = max(1.0, (float(xs.max()) - float(xs.min())) * 0.20)
                pad_y = max(1.0, (float(ys.max()) - float(ys.min())) * 0.20)
                bb = np.array([
                    float(xs.min()) - pad_x, float(ys.min()) - pad_y,
                    float(xs.max()) + pad_x, float(ys.max()) + pad_y,
                ], dtype=np.float32)

            score = float(kp17[:, 2].mean())
            bboxes.append(np.asarray(bb, dtype=np.float32))
            scores.append(score)
            kpts17s.append(kp17)

        if bboxes:
            cache[t] = _BMPResult(
                np.stack(bboxes,  axis=0),
                np.array(scores, dtype=np.float32),
                np.stack(kpts17s, axis=0),
            )

    n_frames_with_data = sum(1 for r in cache if r is not None)
    return {
        "model":             _BMPReplayModel(cache),
        "checkpoint_path":   "<auto from input POSES (BMP/ViTPose/RTMW composite)>",
        "_bmp_cache":        cache,
        "_n_frames_filled":  n_frames_with_data,
    }


class Pose3DUpgradeNode:
    """Take 2D POSES + masks + Fast SAM 3D Body → output augmented
    POSES with full 3D fields filled. 2D keypoints are preserved
    unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_bundle":       ("POSE_BUNDLE",),
                "fast_sam_3d_body":  ("FASTSAM3DBODY",),
            },
            "optional": {
                "yolo11_pose": (
                    "YOLO11POSE",
                    {
                        "tooltip": (
                            "Optional. Wired into Fast SAM 3D Body's "
                            "hand-bbox decoder for slightly more "
                            "accurate SMPL hand pose. If NOT connected, "
                            "Pose3DUpgrade auto-builds an equivalent "
                            "dict from the input POSES' composite 2D "
                            "keypoints (BMP/ViTPose/RTMW fusion), "
                            "which is typically HIGHER quality than a "
                            "standalone YOLO11m-Pose anyway. Most "
                            "users should leave this disconnected."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "upgrade"
    CATEGORY = "4dhumans"

    def upgrade(self, pose_bundle, fast_sam_3d_body, yolo11_pose=None):
        # ---------------------------------------------------------------
        # Phase 0: unpack the bundle (single contract from BMPRTMWPose)
        # ---------------------------------------------------------------
        poses          = pose_bundle["poses"]
        bmp_masks      = pose_bundle["bmp_masks"]
        images         = pose_bundle["images"]
        fps            = float(pose_bundle["fps"])
        pose_3d_fps    = float(pose_bundle.get("pose_3d_fps", fps))
        phmr_stride    = int(pose_bundle.get("phmr_stride", 1))
        sampled_frames_list = pose_bundle.get("sampled_frames", None)
        n_persons_bundle = int(pose_bundle.get("n_persons", 0))
        B_bundle       = int(pose_bundle.get("n_frames", 0))
        img_h_bundle   = int(pose_bundle.get("img_h", 0))
        img_w_bundle   = int(pose_bundle.get("img_w", 0))

        B, img_h, img_w, _C = images.shape
        if (img_h != img_h_bundle) or (img_w != img_w_bundle):
            _logger.warning(
                "Pose3DUpgrade: images shape (%d, %d) doesn't match "
                "bundle img_h/img_w (%d, %d). Using images shape as "
                "ground truth.",
                img_h, img_w, img_h_bundle, img_w_bundle,
            )
        if B != B_bundle and B_bundle > 0:
            _logger.warning(
                "Pose3DUpgrade: image frame count (%d) doesn't match "
                "bundle n_frames (%d).",
                B, B_bundle,
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
        if n_persons_masks != n_persons_bundle:
            _logger.warning(
                "Pose3DUpgrade: mask slots (%d) ≠ bundle n_persons (%d). "
                "Using mask slot count for 3D inference.",
                n_persons_masks, n_persons_bundle,
            )
        n_persons = n_persons_masks
        masks = bmp_masks.reshape(B, n_persons, bmp_masks.shape[-2],
                                   bmp_masks.shape[-1])
        masks_np = masks.cpu().numpy() > 0.5  # (B, N, H, W) bool

        rgb = images[..., :3]
        images_np_u8 = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

        _logger.info(
            "Pose3DUpgrade: %d frames, %d persons (mask slots), %dx%d, "
            "pose_3d_fps=%.1f (src %.1f, stride=%d, sampled=%d)",
            B, n_persons, img_w, img_h, pose_3d_fps, fps,
            phmr_stride,
            len(sampled_frames_list) if sampled_frames_list is not None else B,
        )

        # ---------------------------------------------------------------
        # Phase 1: per-frame mask → bbox; sampling schedule from bundle
        # ---------------------------------------------------------------
        # The upstream BMPRTMWPose pre-computed phmr_stride + sampled_frames
        # from its (fps, pose_3d_fps) settings. We just consume them — no
        # local fps/stride logic anywhere in this node.
        if sampled_frames_list is not None:
            sampled_frames = set(int(t) for t in sampled_frames_list)
        else:
            # Defensive fallback: bundle came from a future/older
            # producer that didn't ship sampled_frames. Treat as
            # "every frame" so we degrade safely rather than crash.
            sampled_frames = set(range(B))

        # Two structures:
        #   mask_bboxes_per_frame / person_indices_per_frame:
        #       flat lists for SAMPLED frames only (fed to Fast SAM 3D Body)
        #   bbox_by_slot[t][p]:
        #       per-slot indexed dict for ALL frames (used to build the
        #       auto-extracted yolo_pose dict when yolo11_pose is None)
        mask_bboxes_per_frame: List[list] = [[] for _ in range(B)]
        person_indices_per_frame: List[list] = [[] for _ in range(B)]
        bbox_by_slot: List[List[Optional[np.ndarray]]] = [
            [None] * n_persons for _ in range(B)
        ]

        for t in range(B):
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
                bbox_by_slot[t][p_idx] = np.array(
                    [x1, y1, x2, y2], dtype=np.float32,
                )
                if t in sampled_frames:
                    mask_bboxes_per_frame[t].append((x1, y1, x2, y2))
                    person_indices_per_frame[t].append(p_idx)

        n_sampled = len(sampled_frames)
        pbar = comfy.utils.ProgressBar(B + 1)
        t0 = time.perf_counter()

        # ---------------------------------------------------------------
        # Auto-build yolo11_pose from input POSES when not connected.
        # Uses the composite 2D keypoints (BMP / ViTPose / RTMW fusion)
        # which are higher quality than what a standalone YOLO11m-Pose
        # would produce — and saves the user a separate node load.
        # ---------------------------------------------------------------
        yolo_pose_was_auto = False
        if yolo11_pose is None:
            yolo11_pose = _yolo_pose_from_input_poses(
                input_poses=poses,
                bbox_by_slot=bbox_by_slot,
                n_persons=n_persons,
                B=B,
            )
            yolo_pose_was_auto = True
            n_filled = yolo11_pose.get("_n_frames_filled", 0)
            _logger.info(
                "Pose3DUpgrade: auto-built yolo11_pose from input POSES "
                "composite (%d frames with detections). Fast SAM 3D "
                "Body will use BMP/ViTPose/RTMW-fused wrist data for "
                "hand-bbox derivation.",
                n_filled,
            )

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
            "Pose3DUpgrade: %d frames, %d persons | yolo_pose=%s | "
            "FastSAM3DBody %.2fs (%d sampled) | MHR2SMPL %.2fs | "
            "body_joints filled: %d (slot,frame) | smpl_j3d filled: "
            "%d | cam_int filled: %d | total %.2fs",
            B, n_persons,
            "auto-from-POSES" if yolo_pose_was_auto
            else ("user-connected" if yolo11_pose is not None else "none"),
            fastsam3db_time, n_sampled,
            mhr2smpl_time, body_filled, smpl_filled,
            cam_filled, elapsed,
        )

        return (merged_poses,)
