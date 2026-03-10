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

    Coordinate convention used by PHALP
    ------------------------------------
    get_3d_parameters() projects joints into [0, render.res] space where the
    full square-padded image (side = new_image_size = max(H, W)) spans [0, 1]
    after division by render.res.  We therefore multiply by new_image_size and
    subtract the padding offsets to recover original pixel coordinates.

    Parameters
    ----------
    joints_2d_flat : np.ndarray  shape (90,)
    img_h, img_w   : int   – original (unpadded) image dimensions
    clip_boundary  : float – extra pixels beyond the image boundary that are
                             still considered visible. -1 disables clipping.
    """
    new_size = max(img_h, img_w)
    left = (new_size - img_w) // 2
    top  = (new_size - img_h) // 2

    # (45, 2) in [0, 1] of padded-square space  →  pixel coords
    joints_norm = joints_2d_flat.reshape(45, 2)
    joints_px   = joints_norm * new_size - np.array([left, top])

    # First 25 joints follow the OpenPose body-25 layout (set by joint_map in
    # phalp's SMPL wrapper: [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, ...])
    openpose_xy = joints_px[:25]                          # (25, 2)
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

    Returns
    -------
    np.ndarray  (T, 25, 3)  with x/y smoothed, confidence unchanged
    """
    if sigma <= 0 or len(joint_series) < 3:
        return joint_series

    T = len(joint_series)
    half  = max(1, int(3 * sigma))
    k     = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = joint_series.copy()
    for j in range(25):
        for c in range(2):  # smooth x and y; leave confidence column alone
            padded = np.pad(joint_series[:, j, c], half, mode="edge")
            conv   = np.convolve(padded, kernel, mode="valid")
            smoothed[:, j, c] = conv[:T]
    return smoothed


class PHALPPoseControlNetNode:
    """
    Runs the full PHALP tracking pipeline on an IMAGE batch:

      1. Detects people with Detectron2 (per frame).
      2. Extracts SMPL pose + appearance embeddings with HMAR.
      3. Associates people across frames with DeepSort + pose/appearance
         distance metrics.
      4. Uses a learned Pose Transformer to *predict* pose for briefly
         occluded / off-screen people.
      5. Projects the tracked/predicted 3D skeleton to 2D and renders an
         OpenPose-style skeleton on a black canvas.

    When retroactive_fill is enabled the node uses a two-pass strategy:
      Pass 1 – run full tracking across ALL frames, recording every track's
               pose data even while it is still tentative (not yet confirmed).
      Pass 2 – render; for tracks that eventually get confirmed, the early
               tentative frames are retroactively filled in so no skeleton
               is missing from the output even when n_init > 1.

    smooth_sigma applies Gaussian temporal smoothing to each confirmed
    track's 2D joint positions, reducing per-frame jitter.  It is applied
    after the full tracking pass so it automatically uses bidirectional
    context (past and future frames).

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
                        "default": True,
                        "tooltip": (
                            "If enabled, also draw skeletons for tracks whose "
                            "person was not detected in this frame but whose pose "
                            "was predicted by PHALP's temporal model."
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
                            "Has no effect when n_init=1. "
                            "Automatically enabled when smooth_sigma > 0."
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
                            "Gaussian temporal smoothing applied to each track's "
                            "2D joint positions to reduce per-frame jitter. "
                            "Value is the sigma (standard deviation) in frames. "
                            "0 = disabled. "
                            "1-2 = light smoothing. "
                            "3-5 = strong smoothing (may blur fast motion)."
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
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_tracking(tracker, images_nchw, measurements):
        """
        Forward pass: run PHALP tracking over every frame and return a
        per-frame snapshot of every track (tentative and confirmed alike).

        Returns
        -------
        snapshots : list[dict]
            snapshots[t] maps track_id -> {
                'joints_2d':         np.ndarray (90,) or None,
                'is_confirmed':      bool,
                'time_since_update': int,
            }
        """
        B, C, img_h, img_w = images_nchw.shape
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

    @staticmethod
    def _apply_smoothing(snapshots, eventually_confirmed, img_h, img_w,
                         clip_boundary, sigma):
        """
        Build per-track joint time series, smooth, and write back into
        snapshots in-place.  Only smooths frames with actual detections
        (time_since_update == 0); predicted/occluded entries are left as-is.
        """
        B = len(snapshots)

        for tid in eventually_confirmed:
            # Collect frames where this track has a real detection
            detected_frames = [
                t for t in range(B)
                if tid in snapshots[t]
                and snapshots[t][tid]["joints_2d"] is not None
                and snapshots[t][tid]["time_since_update"] == 0
            ]
            if len(detected_frames) < 3:
                continue

            # Convert raw joints → OpenPose pixel coords for each detection
            kp_series = np.stack([
                _joints_to_openpose(
                    snapshots[t][tid]["joints_2d"], img_h, img_w, clip_boundary
                )
                for t in detected_frames
            ])  # (N_detected, 25, 3)

            smoothed = _smooth_track_joints(kp_series, sigma)

            # Store smoothed keypoints back — we use a special key so the
            # render pass can use them directly without re-converting.
            for i, t in enumerate(detected_frames):
                snapshots[t][tid]["smoothed_kp"] = smoothed[i]

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def render_pose(self, images, phalp, clip_boundary, draw_predicted,
                    retroactive_fill, smooth_sigma):
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

        use_two_pass = retroactive_fill or (smooth_sigma > 0)

        if use_two_pass:
            # ── Two-pass mode ─────────────────────────────────────────
            snapshots = self._run_tracking(tracker, images_nchw, measurements)
            pbar.update(B // 2)

            eventually_confirmed = {
                tid
                for frame_snap in snapshots
                for tid, tdata in frame_snap.items()
                if tdata["is_confirmed"]
            }

            if smooth_sigma > 0:
                self._apply_smoothing(
                    snapshots, eventually_confirmed,
                    img_h, img_w, clip_boundary, smooth_sigma
                )

            pose_images = []
            for t, frame_snap in enumerate(snapshots):
                canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                for tid, tdata in frame_snap.items():
                    confirmed    = tdata["is_confirmed"]
                    since_update = tdata["time_since_update"]

                    if confirmed:
                        if since_update > 0 and not draw_predicted:
                            continue
                    elif tid in eventually_confirmed:
                        if since_update > 0:
                            continue  # pure Kalman prediction before confirm
                    else:
                        continue  # never confirmed — false positive

                    # Use pre-smoothed keypoints when available, otherwise
                    # convert raw joints on the fly.
                    if "smoothed_kp" in tdata:
                        kp = tdata["smoothed_kp"]
                    elif tdata["joints_2d"] is not None:
                        kp = _joints_to_openpose(
                            tdata["joints_2d"], img_h, img_w, clip_boundary
                        )
                    else:
                        continue

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
