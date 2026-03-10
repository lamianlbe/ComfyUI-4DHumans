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


def _double_frames_kp(timeline):
    """
    2x frame interpolation on a keypoint timeline.
    For N input frames produces 2*N-1 output frames.
    Odd frames are linearly interpolated from their neighbours.
    """
    N = len(timeline)
    if N < 2:
        return list(timeline)

    out = []
    for i in range(N - 1):
        out.append(timeline[i])
        if timeline[i] is not None and timeline[i + 1] is not None:
            out.append((timeline[i] + timeline[i + 1]) * 0.5)
        else:
            out.append(timeline[i] if timeline[i] is not None else timeline[i + 1])
    out.append(timeline[-1])
    return out


class PHALPPoseControlNetNode:
    """
    Runs the full PHALP tracking pipeline on an IMAGE batch and renders
    OpenPose-style skeletons on a black canvas.

    Requires the PHALP node to be loaded upstream via 'Load PHALP'.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "phalp": ("PHALP",),
                "single_person_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Assume all detections belong to one person. "
                            "Merges all track IDs into a single timeline so that "
                            "smoothing and interpolation work seamlessly across "
                            "track switches caused by detector hiccups. "
                            "When on: gap frames are interpolated and filled. "
                            "When off: only detected frames are drawn per track."
                        ),
                    },
                ),
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
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Overlay the pose skeleton on the original video frame "
                            "instead of a black canvas. Useful for verifying that "
                            "the detected skeleton aligns with the person."
                        ),
                    },
                ),
                "double_frame": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "2x frame interpolation: insert a linearly interpolated "
                            "pose frame between every pair of original frames. "
                            "Output has 2*N-1 frames. Useful for generating smoother "
                            "ControlNet-guided animations from low-fps source video."
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
    # pass-1: run tracker and collect snapshots
    # ------------------------------------------------------------------

    @staticmethod
    def _run_tracking(tracker, images_nchw, measurements):
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
    # single-person mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_single_person_timeline(snapshots, eventually_confirmed,
                                      img_h, img_w, clip_boundary):
        B = len(snapshots)
        timeline = [None] * B

        for t in range(B):
            best_kp = None
            best_su = float("inf")

            for tid, tdata in snapshots[t].items():
                if not tdata["is_confirmed"] and tid not in eventually_confirmed:
                    continue
                if tdata["joints_2d"] is None:
                    continue
                su = tdata["time_since_update"]
                if su < best_su:
                    best_su = su
                    best_kp = tdata["joints_2d"]

            if best_su == 0 and best_kp is not None:
                timeline[t] = _joints_to_openpose(
                    best_kp, img_h, img_w, clip_boundary
                )

        return timeline

    @staticmethod
    def _smooth_timeline(timeline, sigma):
        detected = [(t, kp) for t, kp in enumerate(timeline) if kp is not None]
        if sigma <= 0 or len(detected) < 3:
            return
        kp_series = np.stack([kp for _, kp in detected])
        smoothed  = _smooth_track_joints(kp_series, sigma)
        for i, (t, _) in enumerate(detected):
            timeline[t] = smoothed[i]

    @staticmethod
    def _interpolate_timeline(timeline):
        detected_times = [t for t in range(len(timeline)) if timeline[t] is not None]
        if len(detected_times) < 2:
            return
        t_first, t_last = detected_times[0], detected_times[-1]
        for t in range(t_first + 1, t_last):
            if timeline[t] is not None:
                continue
            t0 = max(dt for dt in detected_times if dt < t)
            t1 = min(dt for dt in detected_times if dt > t)
            alpha = (t - t0) / (t1 - t0)
            timeline[t] = timeline[t0] + alpha * (timeline[t1] - timeline[t0])

    @staticmethod
    def _fill_nearest(timeline):
        B = len(timeline)
        last = None
        for t in range(B):
            if timeline[t] is not None:
                last = timeline[t]
            elif last is not None:
                timeline[t] = last
        last = None
        for t in range(B - 1, -1, -1):
            if timeline[t] is not None:
                last = timeline[t]
            elif last is not None:
                timeline[t] = last

    # ------------------------------------------------------------------
    # multi-person mode helpers (per-track)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_smoothing(snapshots, eventually_confirmed, img_h, img_w,
                         clip_boundary, sigma):
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
            ])
            smoothed = _smooth_track_joints(kp_series, sigma)
            for i, t in enumerate(detected_frames):
                snapshots[t][tid]["smoothed_kp"] = smoothed[i]

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def render_pose(self, images, phalp, single_person_mode, clip_boundary,
                    smooth_sigma, debug, double_frame):
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

        # single_person_mode → draw_predicted=True, interpolate_missing=True
        # multi-person       → draw_predicted=False, interpolate_missing=False
        draw_predicted     = single_person_mode
        interpolate_missing = single_person_mode

        # Always use two-pass when smoothing or single-person mode is active
        use_two_pass = single_person_mode or smooth_sigma > 0

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

            if single_person_mode:
                # ── Single-person pipeline ─────────────────────────────
                timeline = self._build_single_person_timeline(
                    snapshots, eventually_confirmed,
                    img_h, img_w, clip_boundary
                )

                if smooth_sigma > 0:
                    self._smooth_timeline(timeline, smooth_sigma)

                self._interpolate_timeline(timeline)
                self._fill_nearest(timeline)

                if double_frame:
                    timeline = _double_frames_kp(timeline)

                pose_images = []
                n_out = len(timeline)
                for t in range(n_out):
                    if debug:
                        # Use original frame as background (clamp to source range)
                        src_t = min(t * B // n_out, B - 1) if double_frame else t
                        bg = (images_nchw[src_t].permute(1, 2, 0) * 255).byte().numpy()
                        canvas = bg.copy()
                    else:
                        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                    if timeline[t] is not None:
                        canvas = render_openpose(canvas, timeline[t])

                    pose_images.append(
                        torch.from_numpy(canvas.astype(np.float32) / 255.0)
                    )
                    pbar.update(1)

            else:
                # ── Multi-person pipeline ──────────────────────────────
                if smooth_sigma > 0:
                    self._apply_smoothing(
                        snapshots, eventually_confirmed,
                        img_h, img_w, clip_boundary, smooth_sigma
                    )

                pose_images = []
                for t, frame_snap in enumerate(snapshots):
                    if debug:
                        bg = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
                        canvas = bg.copy()
                    else:
                        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                    for tid, tdata in frame_snap.items():
                        confirmed    = tdata["is_confirmed"]
                        since_update = tdata["time_since_update"]

                        if confirmed:
                            pass
                        elif tid in eventually_confirmed:
                            if since_update > 0:
                                continue
                        else:
                            continue

                        if since_update == 0:
                            if "smoothed_kp" in tdata:
                                kp = tdata["smoothed_kp"]
                            elif tdata["joints_2d"] is not None:
                                kp = _joints_to_openpose(
                                    tdata["joints_2d"], img_h, img_w,
                                    clip_boundary
                                )
                            else:
                                continue
                        else:
                            continue  # multi-person: only draw detected

                        canvas = render_openpose(canvas, kp)

                    pose_images.append(
                        torch.from_numpy(canvas.astype(np.float32) / 255.0)
                    )
                    pbar.update(1)

        else:
            # ── Single-pass mode (no smoothing, no single-person) ─────
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

                if debug:
                    canvas = img_np.copy()
                else:
                    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                for track_ in tracker.tracker.tracks:
                    if not track_.is_confirmed():
                        continue
                    if track_.time_since_update > 0:
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
