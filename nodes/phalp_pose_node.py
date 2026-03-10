import numpy as np
import torch
import comfy.model_management
import comfy.utils

from ..humans4d.hmr2.utils.render_openpose import render_openpose
from .load_phalp_node import _ensure_phalp_importable


def _joints_to_openpose(joints_2d_flat, img_h, img_w, clip_boundary):
    """
    Convert a (90,) normalised 2d_joints vector from PHALP track history into
    an OpenPose-compatible (25, 3) array with pixel (x, y, confidence) columns.
    """
    new_size = max(img_h, img_w)
    left = (new_size - img_w) // 2
    top  = (new_size - img_h) // 2

    joints_norm = joints_2d_flat.reshape(45, 2)
    joints_px   = joints_norm * new_size - np.array([left, top])

    openpose_xy = joints_px[:25]
    confidence  = np.ones((25, 1), dtype=np.float32)

    if clip_boundary >= 0:
        lo_x, hi_x = -clip_boundary, img_w + clip_boundary
        lo_y, hi_y = -clip_boundary, img_h + clip_boundary
        out_of_frame = (
            (openpose_xy[:, 0] < lo_x) | (openpose_xy[:, 0] > hi_x) |
            (openpose_xy[:, 1] < lo_y) | (openpose_xy[:, 1] > hi_y)
        )
        confidence[out_of_frame] = 0.0

    return np.concatenate([openpose_xy, confidence], axis=-1)   # (25, 3)


def _smooth_track_joints(joint_series, sigma):
    """
    Apply Gaussian temporal smoothing to a track's 2D joint time series.

    Parameters
    ----------
    joint_series : np.ndarray  (T, 25, 3)  – (x, y, confidence) per frame
    sigma        : float       – Gaussian sigma in frames; 0 = no-op
    """
    if sigma <= 0 or len(joint_series) < 3:
        return joint_series

    T = len(joint_series)
    half   = max(1, int(3 * sigma))
    k      = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = joint_series.copy()
    for j in range(25):
        for c in range(2):  # x and y; leave confidence alone
            padded          = np.pad(joint_series[:, j, c], half, mode="edge")
            smoothed[:, j, c] = np.convolve(padded, kernel, mode="valid")[:T]
    return smoothed


class PHALPPoseControlNetNode:
    """
    Runs the full PHALP tracking pipeline on an IMAGE batch and renders
    OpenPose-style skeletons on a black canvas.

    Two-pass mode (enabled when retroactive_fill, smooth_sigma>0, or
    interpolate_missing is active):

      Pass 1 – run full tracking, recording every track's data including
               tentative frames.
      Pass 2 – post-process then render:
               • retroactive_fill  – draw early tentative frames for tracks
                                     that are later confirmed.
               • smooth_sigma      – Gaussian-smooth detected joint positions
                                     along the time axis to reduce jitter.
               • interpolate_missing – linearly interpolate joint positions
                                     for frames between two detections where
                                     the person was temporarily lost.

    Requires the PHALP node to be loaded upstream via 'Load PHALP'.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "phalp": ("PHALP",),
                "clip_boundary": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 512.0,
                        "step": 1.0,
                        "tooltip": (
                            "Keypoints projected outside the image by more than this "
                            "many pixels are hidden. "
                            "0 = strict edge clipping. "
                            "> 0 = allow slightly off-screen joints. "
                            "-1 = draw all joints unconditionally."
                        ),
                    },
                ),
                "draw_predicted": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When a confirmed person is not detected in a frame "
                            "(occluded, off-screen, etc.), draw their LAST KNOWN "
                            "pose instead of leaving a blank. "
                            "Note: this reuses the previous detection's frozen pose "
                            "and may produce a ghost/trail effect. "
                            "Has no effect on frames that are filled by interpolation."
                        ),
                    },
                ),
                "retroactive_fill": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Two-pass mode: run full tracking first, then render. "
                            "Tracks confirmed later in the video are retroactively "
                            "drawn for their early tentative frames, so no skeleton "
                            "is lost during the n_init warm-up period. "
                            "Has no effect when n_init=1."
                        ),
                    },
                ),
                "interpolate_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When a confirmed person is temporarily not detected "
                            "between two detected frames, linearly interpolate their "
                            "joint positions from the surrounding detections. "
                            "Produces smooth transitions instead of blank frames or "
                            "frozen ghost poses. "
                            "Frames before the first detection or after the last "
                            "detection of a track are not interpolated."
                        ),
                    },
                ),
                "smooth_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": (
                            "Gaussian temporal smoothing on detected joint positions "
                            "to reduce per-frame jitter. "
                            "Sigma is in frames. "
                            "0 = disabled. 1-2 = light. 3-5 = strong."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    # ------------------------------------------------------------------
    # pass-1 helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run_tracking(tracker, images_nchw, measurements):
        """
        Forward pass: run PHALP tracking and record every track's state
        (tentative and confirmed alike) for every frame.

        Returns
        -------
        snapshots : list[dict]
            snapshots[t] maps track_id -> {
                'joints_2d':         np.ndarray (90,) or None,
                'is_confirmed':      bool,
                'time_since_update': int,
            }
        """
        snapshots = []
        for t, img_tensor in enumerate(images_nchw):
            img_np     = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()
            frame_name = str(t)

            (pred_bbox, pred_bbox_pad, pred_masks,
             pred_scores, pred_classes,
             gt_tids, gt_annots) = tracker.get_detections(
                img_np, frame_name, t, {}, measurements
            )
            extra_data = tracker.run_additional_models(
                img_np, pred_bbox, pred_masks, pred_scores,
                pred_classes, frame_name, t, measurements, gt_tids, gt_annots
            )
            detections = tracker.get_human_features(
                img_np, pred_masks, pred_bbox, pred_bbox_pad,
                pred_scores, frame_name, pred_classes, t,
                measurements, gt_tids, gt_annots, extra_data
            )

            tracker.tracker.predict()
            tracker.tracker.update(detections, t, frame_name, shot=0)

            frame_snap = {}
            for track_ in tracker.tracker.tracks:
                hist   = track_.track_data.get("history", [])
                joints = hist[-1]["2d_joints"] if hist else None
                frame_snap[track_.track_id] = {
                    "joints_2d":         joints,
                    "is_confirmed":      track_.is_confirmed(),
                    "time_since_update": track_.time_since_update,
                }
            snapshots.append(frame_snap)

        return snapshots

    # ------------------------------------------------------------------
    # pass-2 post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_smoothing(snapshots, eventually_confirmed, img_h, img_w,
                         clip_boundary, sigma):
        """
        Gaussian-smooth the detected joint positions for each confirmed track
        and write the result back as 'smoothed_kp' in the snapshot dicts.
        Only detected frames (time_since_update==0) are smoothed; gap frames
        are left for _interpolate_gaps to handle.
        """
        B = len(snapshots)
        for tid in eventually_confirmed:
            detected_frames = [
                t for t in range(B)
                if tid in snapshots[t]
                and snapshots[t][tid]["joints_2d"] is not None
                and snapshots[t][tid]["time_since_update"] == 0
            ]
            if len(detected_frames) < 3:
                continue

            kp_series = np.stack([
                _joints_to_openpose(
                    snapshots[t][tid]["joints_2d"], img_h, img_w, clip_boundary
                )
                for t in detected_frames
            ])  # (N_detected, 25, 3)

            smoothed = _smooth_track_joints(kp_series, sigma)

            for i, t in enumerate(detected_frames):
                snapshots[t][tid]["smoothed_kp"] = smoothed[i]

    @staticmethod
    def _interpolate_gaps(snapshots, eventually_confirmed, img_h, img_w,
                          clip_boundary):
        """
        For each confirmed track, linearly interpolate joint positions for
        frames that lie between two detected anchor frames.

        Interpolation uses 'smoothed_kp' as the anchor when available
        (i.e. after smoothing has already run), so the interpolated curve
        is consistent with the smoothed detections.

        Gap frames get an 'interpolated_kp' key written into their snapshot.
        Frames before the first or after the last detection are not touched.
        """
        B = len(snapshots)
        for tid in eventually_confirmed:
            track_frames = sorted(t for t in range(B) if tid in snapshots[t])

            # Anchor frames: actual detections
            detected_frames = [
                t for t in track_frames
                if snapshots[t][tid]["joints_2d"] is not None
                and snapshots[t][tid]["time_since_update"] == 0
            ]
            if len(detected_frames) < 2:
                continue

            # Build anchor keypoints (prefer smoothed)
            anchor_kps = {}
            for t in detected_frames:
                tdata = snapshots[t][tid]
                anchor_kps[t] = (
                    tdata["smoothed_kp"] if "smoothed_kp" in tdata
                    else _joints_to_openpose(
                        tdata["joints_2d"], img_h, img_w, clip_boundary
                    )
                )

            # Fill each bounded gap with linear interpolation
            for i in range(len(detected_frames) - 1):
                t0, t1 = detected_frames[i], detected_frames[i + 1]
                if t1 - t0 <= 1:
                    continue  # adjacent frames, no gap

                kp0, kp1 = anchor_kps[t0], anchor_kps[t1]

                gap_frames = [
                    t for t in track_frames
                    if t0 < t < t1
                    and snapshots[t][tid]["time_since_update"] > 0
                ]
                for t in gap_frames:
                    alpha = (t - t0) / (t1 - t0)
                    snapshots[t][tid]["interpolated_kp"] = (
                        kp0 + alpha * (kp1 - kp0)
                    )

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def render_pose(self, images, phalp, clip_boundary, draw_predicted,
                    retroactive_fill, interpolate_missing, smooth_sigma):
        if not _ensure_phalp_importable():
            raise RuntimeError("phalp package not found. See 'Load PHALP' node.")

        tracker = phalp["tracker"]
        tracker.setup_deepsort()

        images_nchw = images.permute(0, 3, 1, 2)   # (B, C, H, W)
        B, C, img_h, img_w = images_nchw.shape

        new_size     = max(img_h, img_w)
        left         = (new_size - img_w) // 2
        top          = (new_size - img_h) // 2
        measurements = [img_h, img_w, new_size, left, top]

        pbar = comfy.utils.ProgressBar(B)

        use_two_pass = retroactive_fill or interpolate_missing or (smooth_sigma > 0)

        if use_two_pass:
            # ── Pass 1: full tracking ──────────────────────────────────
            snapshots = self._run_tracking(tracker, images_nchw, measurements)
            pbar.update(B // 2)

            eventually_confirmed = {
                tid
                for frame_snap in snapshots
                for tid, tdata in frame_snap.items()
                if tdata["is_confirmed"]
            }

            # ── Post-processing (order matters) ───────────────────────
            # 1. Smooth detected frames first so interpolation anchors
            #    are already smooth.
            if smooth_sigma > 0:
                self._apply_smoothing(
                    snapshots, eventually_confirmed,
                    img_h, img_w, clip_boundary, smooth_sigma
                )

            # 2. Interpolate gap frames using (smoothed) anchors.
            if interpolate_missing:
                self._interpolate_gaps(
                    snapshots, eventually_confirmed,
                    img_h, img_w, clip_boundary
                )

            # ── Pass 2: render ─────────────────────────────────────────
            pose_images = []
            for t, frame_snap in enumerate(snapshots):
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                for tid, tdata in frame_snap.items():
                    confirmed    = tdata["is_confirmed"]
                    since_update = tdata["time_since_update"]

                    # ── decide whether to draw this track in this frame ──
                    if confirmed:
                        pass  # always eligible; key selection below
                    elif tid in eventually_confirmed:
                        # tentative frame that will later be confirmed
                        if since_update > 0:
                            continue  # Kalman-only before any detection
                    else:
                        continue  # never confirmed — false positive

                    # ── pick the best available keypoints ──────────────
                    # Priority: smoothed > interpolated > raw > frozen (draw_predicted)
                    if "smoothed_kp" in tdata:
                        kp = tdata["smoothed_kp"]
                    elif "interpolated_kp" in tdata:
                        kp = tdata["interpolated_kp"]
                    elif since_update == 0 and tdata["joints_2d"] is not None:
                        kp = _joints_to_openpose(
                            tdata["joints_2d"], img_h, img_w, clip_boundary
                        )
                    elif draw_predicted and tdata["joints_2d"] is not None:
                        # frozen last-known pose (may look like ghost)
                        kp = _joints_to_openpose(
                            tdata["joints_2d"], img_h, img_w, clip_boundary
                        )
                    else:
                        continue  # no data or draw_predicted disabled

                    canvas = render_openpose(canvas, kp)

                pose_images.append(
                    torch.from_numpy(canvas.astype(np.float32) / 255.0)
                )
                pbar.update(1)

        else:
            # ── Single-pass mode (original behaviour) ─────────────────
            pose_images = []
            for t, img_tensor in enumerate(images_nchw):
                img_np     = (img_tensor.permute(1, 2, 0) * 255).byte().numpy()
                frame_name = str(t)

                (pred_bbox, pred_bbox_pad, pred_masks,
                 pred_scores, pred_classes,
                 gt_tids, gt_annots) = tracker.get_detections(
                    img_np, frame_name, t, {}, measurements
                )
                extra_data = tracker.run_additional_models(
                    img_np, pred_bbox, pred_masks, pred_scores,
                    pred_classes, frame_name, t, measurements, gt_tids, gt_annots
                )
                detections = tracker.get_human_features(
                    img_np, pred_masks, pred_bbox, pred_bbox_pad,
                    pred_scores, frame_name, pred_classes, t,
                    measurements, gt_tids, gt_annots, extra_data
                )

                tracker.tracker.predict()
                tracker.tracker.update(detections, t, frame_name, shot=0)

                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                for track_ in tracker.tracker.tracks:
                    if not track_.is_confirmed():
                        continue
                    if track_.time_since_update > 0 and not draw_predicted:
                        continue

                    hist = track_.track_data["history"][-1]
                    kp   = _joints_to_openpose(
                        hist["2d_joints"], img_h, img_w, clip_boundary
                    )
                    canvas = render_openpose(canvas, kp)

                pose_images.append(
                    torch.from_numpy(canvas.astype(np.float32) / 255.0)
                )
                pbar.update(1)

        return (torch.stack(pose_images),)
