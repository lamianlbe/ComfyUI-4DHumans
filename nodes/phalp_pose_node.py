import numpy as np
import torch
import comfy.model_management
import comfy.utils

from ..humans4d.hmr2.utils.render_openpose import render_openpose
from ..humans4d.hmr2.utils.render_openpose_wholebody import render_wholebody_openpose
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


def _run_smplestx_on_bbox(img_np_rgb, bbox_xyxy, model, cfg):
    """
    Run SMPLest-X on a single detection bbox crop.

    Parameters
    ----------
    img_np_rgb : (H, W, 3) uint8 RGB image
    bbox_xyxy  : (4,) array [x1, y1, x2, y2] in pixel coords
    model      : DataParallel SMPLest-X model (eval mode, on CUDA)
    cfg        : SMPLest-X Config

    Returns
    -------
    dict with:
        "kp2d": (137, 3) float32 array (x, y, confidence) in original image coords
        "joint_cam": (137, 3) float32 array, root-relative 3D joints in camera space
        "inv_trans": (2, 3) affine to map input_img_shape → original image coords
    or None if the bbox is invalid.
    """
    import torchvision.transforms as transforms
    from ..smplestx.utils.data_utils import process_bbox, generate_patch_image

    h, w = img_np_rgb.shape[:2]

    # xyxy → xywh
    bbox_xywh = np.array([
        bbox_xyxy[0], bbox_xyxy[1],
        bbox_xyxy[2] - bbox_xyxy[0],
        bbox_xyxy[3] - bbox_xyxy[1],
    ], dtype=np.float32)

    bbox = process_bbox(
        bbox_xywh, w, h,
        cfg.model.input_img_shape,
        ratio=getattr(cfg.data, "bbox_ratio", 1.25),
    )
    if bbox is None:
        return None

    img_float = img_np_rgb.astype(np.float32)
    img_patch, _trans, inv_trans = generate_patch_image(
        img_float, bbox, 1.0, 0.0, False, cfg.model.input_img_shape,
    )

    transform = transforms.ToTensor()
    img_tensor = transform(img_patch.astype(np.float32)) / 255.0
    img_tensor = img_tensor.cuda()[None, :, :, :]

    with torch.no_grad():
        out = model({"img": img_tensor}, {}, {}, "test")

    # joint_proj: (B, 137, 2) in output_hm_shape space
    jp = out["smplx_joint_proj"][0].cpu().numpy()  # (137, 2)
    # joint_cam: (B, 137, 3) root-relative 3D in camera space
    joint_cam = out["smplx_joint_cam"][0].cpu().numpy()  # (137, 3)
    # root_cam: (B, 1, 3) absolute root position in camera space
    root_cam = out["root_cam"][0, 0].cpu().numpy()       # (3,)

    output_hm = cfg.model.output_hm_shape   # (D=16, H=16, W=12)
    input_img = cfg.model.input_img_shape    # (H=512, W=384)

    # Heatmap space → input_img_shape pixel space
    jp[:, 0] = jp[:, 0] / output_hm[2] * input_img[1]   # x: ×(384/12)=×32
    jp[:, 1] = jp[:, 1] / output_hm[1] * input_img[0]   # y: ×(512/16)=×32

    # Apply inv_trans to map back to original image coords
    ones = np.ones((137, 1), dtype=np.float32)
    jp_homo = np.concatenate([jp, ones], axis=1)          # (137, 3)
    jp_orig = (inv_trans @ jp_homo.T).T                   # (137, 2)

    confidence = np.ones((137, 1), dtype=np.float32)
    kp2d = np.concatenate([jp_orig, confidence], axis=1).astype(np.float32)

    return {
        "kp2d": kp2d,
        "joint_cam": joint_cam.astype(np.float32),
        "root_cam": root_cam.astype(np.float32),
        "inv_trans": inv_trans.astype(np.float32),
    }


def _project_face_3d_to_2d(joint_cam_face, cfg, inv_trans, root_cam):
    """
    Project face 3D joints (root-relative) to original image 2D coords.

    Parameters
    ----------
    joint_cam_face : (N, 3) float32 – root-relative 3D joints
    cfg            : SMPLest-X Config
    inv_trans      : (2, 3) affine – maps input_img_shape pixel coords → original image coords
    root_cam       : (3,) float32 – absolute root (pelvis) position in camera space

    Returns
    -------
    (N, 2) float32 – pixel coords in original image space
    """
    # Convert root-relative → absolute camera-space coords
    joints_abs = joint_cam_face + root_cam[None, :]

    focal = cfg.model.focal          # (fx, fy)
    princpt = cfg.model.princpt      # (cx, cy)
    input_body = cfg.model.input_body_shape  # (H=256, W=192)
    input_img = cfg.model.input_img_shape    # (H=512, W=384)

    # Perspective projection: 3D absolute → input_body pixel space
    z = joints_abs[:, 2] + 1e-4
    x_body = joints_abs[:, 0] / z * focal[0] + princpt[0]
    y_body = joints_abs[:, 1] / z * focal[1] + princpt[1]

    # input_body pixel space → input_img pixel space
    x_px = x_body / input_body[1] * input_img[1]
    y_px = y_body / input_body[0] * input_img[0]

    # Apply inv_trans to map to original image coords
    pts = np.stack([x_px, y_px, np.ones_like(x_px)], axis=1)  # (N, 3)
    pts_orig = (inv_trans @ pts.T).T  # (N, 2)
    return pts_orig.astype(np.float32)


def _gaussian_kernel(sigma):
    """Build a 1D Gaussian kernel for temporal smoothing."""
    half = max(1, int(3 * sigma))
    k = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel, half


def _smooth_1d(series, kernel, half, T):
    """Convolve a 1D signal with edge-padded Gaussian kernel."""
    padded = np.pad(series, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:T]


def _smooth_track_joints(joint_series, sigma):
    """
    Apply Gaussian temporal smoothing to a track's 2D joint time series.

    Parameters
    ----------
    joint_series : np.ndarray  (T, J, 3)  – (x, y, confidence) per frame
    sigma        : float       – Gaussian sigma in frames; 0 = no-op
    """
    if sigma <= 0 or len(joint_series) < 3:
        return joint_series

    T = len(joint_series)
    n_joints = joint_series.shape[1]
    kernel, half = _gaussian_kernel(sigma)

    smoothed = joint_series.copy()
    for j in range(n_joints):
        for c in range(2):  # x and y; leave confidence alone
            smoothed[:, j, c] = _smooth_1d(joint_series[:, j, c], kernel, half, T)
    return smoothed


def _smooth_3d_series(series_3d, sigma):
    """
    Apply Gaussian temporal smoothing to a (T, N, 3) xyz series.
    """
    if sigma <= 0 or len(series_3d) < 3:
        return series_3d

    T = len(series_3d)
    N = series_3d.shape[1]
    kernel, half = _gaussian_kernel(sigma)

    smoothed = series_3d.copy()
    for j in range(N):
        for c in range(3):  # x, y, z
            smoothed[:, j, c] = _smooth_1d(series_3d[:, j, c], kernel, half, T)
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

    Optionally accepts a SMPLest-X model for whole-body 137-joint estimation
    (body + hands + face) instead of PHALP's default 25 body joints.

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
            },
            "optional": {
                "smplestx": ("SMPLESTX",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    # ------------------------------------------------------------------
    # pass-1: run tracker and collect snapshots
    # ------------------------------------------------------------------

    @staticmethod
    def _run_tracking(tracker, images_nchw, measurements, smplestx=None):
        """
        Run PHALP tracking on all frames.  Optionally run SMPLest-X on each
        detection to produce 137-joint keypoints per frame.

        Returns a list of per-frame snapshot dicts.  When smplestx is provided
        each snapshot also contains a ``"__smplestx"`` key with a list of
        ``(score, kp137)`` tuples — one per detection.
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

            # ── Optional SMPLest-X inference on each detected bbox ────────
            frame_smplestx = []
            if smplestx is not None and len(pred_bbox) > 0:
                sx_model = smplestx["model"]
                sx_cfg   = smplestx["cfg"]
                for i in range(len(pred_bbox)):
                    result = _run_smplestx_on_bbox(
                        img_np, pred_bbox[i], sx_model, sx_cfg,
                    )
                    score = float(pred_scores[i]) if i < len(pred_scores) else 0.0
                    frame_smplestx.append((score, result))

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

            if frame_smplestx:
                frame_snap["__smplestx"] = frame_smplestx

            snapshots.append(frame_snap)

        return snapshots

    # ------------------------------------------------------------------
    # single-person mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_single_person_timeline(snapshots, eventually_confirmed,
                                      img_h, img_w, clip_boundary,
                                      use_smplestx=False):
        """
        Build a per-frame keypoint timeline for single-person mode.

        When *use_smplestx* is True, uses the highest-scoring SMPLest-X
        detection per frame (137 joints) instead of PHALP's 25-joint output.

        Returns (timeline, timeline_3d):
            timeline   : list of (137,3) or (25,3) 2D keypoints (or None)
            timeline_3d: list of dicts {"joint_cam": (137,3), "inv_trans": (2,3)} or None
        """
        B = len(snapshots)
        timeline = [None] * B
        timeline_3d = [None] * B

        for t in range(B):
            snap = snapshots[t]

            if use_smplestx and "__smplestx" in snap:
                # Pick the highest-confidence detection's SMPLest-X result
                best_result = None
                best_score = -1.0
                for score, result in snap["__smplestx"]:
                    if result is not None and score > best_score:
                        best_score = score
                        best_result = result
                if best_result is not None:
                    best_kp = best_result["kp2d"].copy()
                    # Apply clip_boundary to 137-joint keypoints
                    if clip_boundary >= 0:
                        lo_x, hi_x = -clip_boundary, img_w + clip_boundary
                        lo_y, hi_y = -clip_boundary, img_h + clip_boundary
                        oob = (
                            (best_kp[:, 0] < lo_x) | (best_kp[:, 0] > hi_x) |
                            (best_kp[:, 1] < lo_y) | (best_kp[:, 1] > hi_y)
                        )
                        best_kp[oob, 2] = 0.0
                    timeline[t] = best_kp
                    timeline_3d[t] = {
                        "joint_cam": best_result["joint_cam"],
                        "root_cam": best_result["root_cam"],
                        "inv_trans": best_result["inv_trans"],
                    }
            else:
                # Fallback: use PHALP 25-joint output
                best_kp = None
                best_su = float("inf")

                for tid, tdata in snap.items():
                    if isinstance(tid, str):
                        continue  # skip __smplestx key
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

        return timeline, timeline_3d

    @staticmethod
    def _smooth_timeline(timeline, sigma, timeline_3d=None, cfg=None):
        """
        Gaussian temporal smoothing on all joints in 2D pixel space.
        (3D is only used for interpolation, not smoothing, because per-frame
        projection parameters inv_trans/root_cam are themselves noisy.)
        """
        detected = [(t, kp) for t, kp in enumerate(timeline) if kp is not None]
        if sigma <= 0 or len(detected) < 3:
            return
        kp_series = np.stack([kp for _, kp in detected])
        smoothed = _smooth_track_joints(kp_series, sigma)
        for i, (t, _) in enumerate(detected):
            timeline[t] = smoothed[i]

    @staticmethod
    def _interpolate_timeline(timeline, timeline_3d=None, cfg=None):
        """
        Linearly interpolate gaps in the timeline. When timeline_3d is provided,
        face joints (65-136) are interpolated in 3D and re-projected to 2D.
        """
        detected_times = [t for t in range(len(timeline)) if timeline[t] is not None]
        if len(detected_times) < 2:
            return

        n_joints = timeline[detected_times[0]].shape[0]
        use_3d_face = (timeline_3d is not None and cfg is not None
                       and n_joints == 137)

        t_first, t_last = detected_times[0], detected_times[-1]
        for t in range(t_first + 1, t_last):
            if timeline[t] is not None:
                continue
            t0 = max(dt for dt in detected_times if dt < t)
            t1 = min(dt for dt in detected_times if dt > t)
            alpha = (t - t0) / (t1 - t0)

            # Body+hand: linear interpolation in 2D (always)
            timeline[t] = timeline[t0] + alpha * (timeline[t1] - timeline[t0])

            if use_3d_face and timeline_3d[t0] is not None and timeline_3d[t1] is not None:
                # Face: interpolate in 3D, then project to 2D
                cam0 = timeline_3d[t0]["joint_cam"][65:137]
                cam1 = timeline_3d[t1]["joint_cam"][65:137]
                face_3d_interp = cam0 + alpha * (cam1 - cam0)

                # Use inv_trans and root_cam from nearest detected frame
                src = timeline_3d[t0] if alpha < 0.5 else timeline_3d[t1]
                inv_trans = src["inv_trans"]
                root_cam = src["root_cam"]
                face_2d = _project_face_3d_to_2d(face_3d_interp, cfg, inv_trans, root_cam)
                timeline[t][65:137, :2] = face_2d

                # Also interpolate the 3D record for potential downstream use
                root_cam_interp = (timeline_3d[t0]["root_cam"]
                                   + alpha * (timeline_3d[t1]["root_cam"]
                                              - timeline_3d[t0]["root_cam"]))
                timeline_3d[t] = {
                    "joint_cam": timeline_3d[t0]["joint_cam"] + alpha * (
                        timeline_3d[t1]["joint_cam"] - timeline_3d[t0]["joint_cam"]),
                    "root_cam": root_cam_interp,
                    "inv_trans": inv_trans,
                }

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
    # rendering helper
    # ------------------------------------------------------------------

    @staticmethod
    def _render_kp(canvas, kp, use_wholebody):
        """Draw keypoints on canvas using the appropriate renderer."""
        if use_wholebody:
            return render_wholebody_openpose(canvas, kp)
        return render_openpose(canvas, kp)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def render_pose(self, images, phalp, single_person_mode, clip_boundary,
                    smooth_sigma, debug, double_frame, smplestx=None):
        if not _ensure_phalp_importable():
            raise RuntimeError("phalp package not found. See 'Load PHALP' node.")

        tracker = phalp["tracker"]
        tracker.setup_deepsort()

        use_smplestx = smplestx is not None
        use_wholebody = use_smplestx  # 137 joints → wholebody renderer

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

        # Always use two-pass when smoothing, single-person, or SMPLest-X
        use_two_pass = single_person_mode or smooth_sigma > 0 or use_smplestx

        if use_two_pass:
            # ── Pass 1: full tracking ──────────────────────────────────
            snapshots = self._run_tracking(
                tracker, images_nchw, measurements,
                smplestx=smplestx,
            )
            pbar.update(B // 2)

            eventually_confirmed = {
                tid
                for frame_snap in snapshots
                for tid, tdata in frame_snap.items()
                if not isinstance(tid, str) and tdata["is_confirmed"]
            }

            if single_person_mode:
                # ── Single-person pipeline ─────────────────────────────
                timeline, timeline_3d = self._build_single_person_timeline(
                    snapshots, eventually_confirmed,
                    img_h, img_w, clip_boundary,
                    use_smplestx=use_smplestx,
                )

                sx_cfg = smplestx["cfg"] if use_smplestx else None

                if smooth_sigma > 0:
                    self._smooth_timeline(timeline, smooth_sigma,
                                          timeline_3d=timeline_3d, cfg=sx_cfg)

                self._interpolate_timeline(timeline,
                                           timeline_3d=timeline_3d, cfg=sx_cfg)
                self._fill_nearest(timeline)

                if double_frame:
                    timeline = _double_frames_kp(timeline)

                pose_images = []
                n_out = len(timeline)
                for t in range(n_out):
                    if debug:
                        src_t = min(t * B // n_out, B - 1) if double_frame else t
                        bg = (images_nchw[src_t].permute(1, 2, 0) * 255).byte().numpy()
                        canvas = bg.copy()
                    else:
                        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                    if timeline[t] is not None:
                        canvas = self._render_kp(canvas, timeline[t], use_wholebody)

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

                    # If SMPLest-X is active, draw all detections' wholebody poses
                    if use_smplestx and "__smplestx" in frame_snap:
                        for _score, result in frame_snap["__smplestx"]:
                            if result is not None:
                                canvas = self._render_kp(canvas, result["kp2d"], True)
                    else:
                        for tid, tdata in frame_snap.items():
                            if isinstance(tid, str):
                                continue

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
