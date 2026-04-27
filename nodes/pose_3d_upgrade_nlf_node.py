"""
Pose3DUpgradeNLF — NLF backend variant of Pose3DUpgrade.

Same input/output contract as Pose3DUpgradeNode (consumes a POSE_BUNDLE
from BMPRTMWPose; outputs a 2D-preserving POSES dict with the 3D fields
filled in). The only difference is which backbone runs:

    Pose3DUpgrade      ← Fast SAM 3D Body  (top-down per-mask-bbox; MHR-70
                                            mesh; better hand detail when
                                            wired with YOLO11m-Pose)
    Pose3DUpgradeNLF   ← NLF (this node)   (full-frame self-detect; SMPL-24
                                            joints only; no mesh; no hand
                                            decoder)

When to pick which:
  - **NLF** (this node) — strong baseline 3D body for general scenes,
    robust to crowding because it runs full-frame and has its own
    detector. No fingertip-quality hand 3D. Output is non-commercial
    research only (see LoadNLFNode license).
  - **Fast SAM 3D Body** — better mesh detail and hand 3D when paired
    with YOLO11m-Pose / BMP wrist signals; weaker on extreme crowding
    because it relies on our mask bboxes being clean. Open license.

Slot ownership in the merged POSES (matches Pose3DUpgrade):

    keypoints       (133, 3)  ← INPUT POSES (BMP / RTMW / FaRL / WiLoR /
                                              ViTPose composite)
    keypoints_raw   (133, 3)  ← INPUT POSES (preserved snapshot)
    body_joints2d   (25, 2)   ← NLF (SMPL-24 projected via 55° FOV intrinsic
                                     → OpenPose-25)
    body_joints     (25, 3)   ← NLF (SMPL-24 → OpenPose-25, camera frame)
    smpl_j3d        (24, 3)   ← NLF joints3d_nonparam (camera frame, metres)
    cam_int         (3, 3)    ← NLF default 55° FOV pinhole
    scale, offset             ← 1.0 / (0, 0)  (NLF doesn't pad)

Pipeline:

    images ─┬─► [BMPInstanceSeg] ──┬─► masks ──┐
            │                       │           │
            │                       └─► bmp_pose │  ┌──► [BMP+RTMW Pose] ──► pose_bundle ──┐
            └───────────────────────────────────┴──┤                                       │
                                                    │                                       │
                                                    ▼                                       ▼
                                          [Pose 3D Upgrade NLF] ◄──────────────── pose_bundle
                                          ◄── pose_3d_model (POSE3D from LoadNLF)
                                                    │
                                                    ▼
                                          POSES (2D + 3D, NPZ-compatible)

NLF doesn't take per-bbox crops as input — it self-detects on the full
frame — so the bundle's bmp_masks are used solely for IoU-matching NLF
detections back to the correct BMP-tracked person slot. This keeps
person identity stable across frames in the merged POSES output, even
though NLF itself is identity-agnostic.
"""

import logging
import time

import numpy as np

import comfy.utils

from .fastsam3db_farl_pose_node import _linear_interp_timeline
from ._nlf_inference import run_nlf_video

_logger = logging.getLogger(__name__)


class Pose3DUpgradeNLFNode:
    """Take POSE_BUNDLE + NLF model → output POSES with 3D fields filled."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_bundle":     ("POSE_BUNDLE",),
                "pose_3d_model":   ("POSE3D",),
                "batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "NLF batch size for detect_smpl_batched. "
                            "NLF runs on the full frame so VRAM "
                            "scales with H × W × batch. 4 is a safe "
                            "default for 1080p on 24 GB; raise on "
                            "Blackwell / lower for 4K. 1 is fine for "
                            "CPU debugging."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "upgrade"
    CATEGORY = "4dhumans"

    def upgrade(self, pose_bundle, pose_3d_model, batch_size=4):
        # ---------------------------------------------------------------
        # Phase 0: validate the model is actually NLF, then unpack bundle
        # ---------------------------------------------------------------
        backend = pose_3d_model.get("backend", "?")
        if backend != "nlf":
            raise ValueError(
                f"Pose3DUpgradeNLF: pose_3d_model backend must be 'nlf', "
                f"got {backend!r}. Wire LoadNLFNode's output here, not "
                f"LoadPromptHMRNode's."
            )

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
                "Pose3DUpgradeNLF: images shape (%d, %d) doesn't match "
                "bundle img_h/img_w (%d, %d). Using images shape.",
                img_h, img_w, img_h_bundle, img_w_bundle,
            )
        if B != B_bundle and B_bundle > 0:
            _logger.warning(
                "Pose3DUpgradeNLF: image frame count (%d) ≠ bundle "
                "n_frames (%d).",
                B, B_bundle,
            )

        if bmp_masks.dim() == 4:
            bmp_masks = bmp_masks[..., 0]
        M = bmp_masks.shape[0]
        if M == 0 or B == 0:
            _logger.warning(
                "Pose3DUpgradeNLF: empty masks or images — returning "
                "input POSES unchanged."
            )
            return (poses,)
        if M % B != 0:
            raise ValueError(
                f"Pose3DUpgradeNLF: mask count ({M}) is not a multiple "
                f"of frame count ({B})."
            )
        n_persons_masks = M // B
        if n_persons_masks != n_persons_bundle:
            _logger.warning(
                "Pose3DUpgradeNLF: mask slots (%d) ≠ bundle n_persons (%d). "
                "Using mask slot count for 3D inference.",
                n_persons_masks, n_persons_bundle,
            )
        n_persons = n_persons_masks
        masks = bmp_masks.reshape(B, n_persons, bmp_masks.shape[-2],
                                   bmp_masks.shape[-1])
        masks_np = masks.cpu().numpy() > 0.5  # (B, N, H, W) bool

        rgb = images[..., :3]
        images_np_u8 = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

        if sampled_frames_list is not None:
            sampled_frames = sorted(set(int(t) for t in sampled_frames_list))
        else:
            sampled_frames = list(range(B))

        _logger.info(
            "Pose3DUpgradeNLF: %d frames, %d persons, %dx%d, "
            "pose_3d_fps=%.1f (src %.1f, stride=%d, sampled=%d), "
            "batch_size=%d",
            B, n_persons, img_w, img_h, pose_3d_fps, fps,
            phmr_stride, len(sampled_frames), batch_size,
        )

        # ---------------------------------------------------------------
        # Phase 1: per-frame mask → bbox (used only for IoU matching;
        # NLF detects on the full frame). We compute bboxes for ALL
        # frames so that an interpolated frame can still recover its
        # slot identity if NLF is later asked to run there too — but
        # we only feed sampled-frame bboxes into run_nlf_video.
        # ---------------------------------------------------------------
        sampled_set = set(sampled_frames)
        mask_bboxes_per_frame = [[] for _ in range(B)]
        person_indices_per_frame = [[] for _ in range(B)]
        for t in range(B):
            if t not in sampled_set:
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

        n_sampled_with_data = sum(
            1 for t in sampled_frames if len(mask_bboxes_per_frame[t]) > 0
        )

        # Progress bar covers: NLF inference frames + 1 (post-processing)
        pbar = comfy.utils.ProgressBar(B + 1)
        t0 = time.perf_counter()

        # ---------------------------------------------------------------
        # Phase 2: NLF batched inference
        # ---------------------------------------------------------------
        result = run_nlf_video(
            images_np_u8=images_np_u8,
            mask_bboxes_per_frame=mask_bboxes_per_frame,
            person_indices_per_frame=person_indices_per_frame,
            sampled_frames=sampled_frames,
            nlf_dict=pose_3d_model,
            n_persons=n_persons,
            img_h=img_h,
            img_w=img_w,
            batch_size=batch_size,
            pbar=pbar,
        )
        persons_nlf       = result["persons"]
        cam_int_per_frame = result["cam_int"]
        nlf_time          = result["nlf_time_s"]

        # NLF doesn't pad/scale — scale=1, offset=zeros for every frame
        # that produced output. Mirror the run_fastsam3db_video pattern
        # so downstream code (sapiens_prompthmr_to_nlf_node, NPZ saver)
        # treats both backends interchangeably.
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
                _linear_interp_timeline(persons_nlf[p_idx]["body_joints2d"])
                _linear_interp_timeline(persons_nlf[p_idx]["body_joints"])
                _linear_interp_timeline(persons_nlf[p_idx]["smpl_j3d"])
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
        # Phase 4: merge 3D into input POSES (preserve 2D keypoints)
        # ---------------------------------------------------------------
        in_persons = list(poses.get("persons", []))
        merged_persons = []
        body_filled = 0
        smpl_filled = 0
        for p_idx in range(n_persons):
            if p_idx < len(in_persons):
                base_person = dict(in_persons[p_idx])
            else:
                base_person = {
                    "visible": True,
                    "body_joints2d": [None] * B,
                    "body_joints":   [None] * B,
                    "smpl_j3d":      [None] * B,
                    "keypoints":     [None] * B,
                    "keypoints_raw": [None] * B,
                }

            base_person["body_joints2d"] = persons_nlf[p_idx]["body_joints2d"]
            base_person["body_joints"]   = persons_nlf[p_idx]["body_joints"]
            base_person["smpl_j3d"]      = persons_nlf[p_idx]["smpl_j3d"]
            body_filled += sum(
                1 for v in persons_nlf[p_idx]["body_joints"] if v is not None
            )
            smpl_filled += sum(
                1 for v in persons_nlf[p_idx]["smpl_j3d"] if v is not None
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

        # Forward Pose Editor's stashed filter params, if present.
        for key in ("_filter_velocity_threshold", "_filter_smooth_sigma"):
            if key in poses:
                merged_poses[key] = poses[key]

        elapsed = time.perf_counter() - t0
        cam_filled = sum(1 for v in cam_int_per_frame if v is not None)
        _logger.info(
            "Pose3DUpgradeNLF: %d frames, %d persons | NLF %.2fs "
            "(%d sampled with masks) | body_joints filled: %d "
            "(slot,frame) | smpl_j3d filled: %d | cam_int filled: %d "
            "| total %.2fs",
            B, n_persons, nlf_time, n_sampled_with_data,
            body_filled, smpl_filled, cam_filled, elapsed,
        )

        return (merged_poses,)
