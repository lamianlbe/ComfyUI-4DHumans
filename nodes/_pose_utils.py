"""Shared utilities for pose estimation nodes."""

import numpy as np
import torch


def run_smplestx_on_bbox(img_np_rgb, bbox_xyxy, model, cfg):
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
        "root_cam": (3,) float32 array, absolute root position
        "inv_trans": (2, 3) affine to map input_img_shape -> original image coords
    or None if the bbox is invalid.
    """
    import torchvision.transforms as transforms
    from ..smplestx.utils.data_utils import process_bbox, generate_patch_image

    h, w = img_np_rgb.shape[:2]

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

    jp = out["smplx_joint_proj"][0].cpu().numpy()
    joint_cam = out["smplx_joint_cam"][0].cpu().numpy()
    root_cam = out["root_cam"][0, 0].cpu().numpy()

    output_hm = cfg.model.output_hm_shape
    input_img = cfg.model.input_img_shape

    jp[:, 0] = jp[:, 0] / output_hm[2] * input_img[1]
    jp[:, 1] = jp[:, 1] / output_hm[1] * input_img[0]

    ones = np.ones((137, 1), dtype=np.float32)
    jp_homo = np.concatenate([jp, ones], axis=1)
    jp_orig = (inv_trans @ jp_homo.T).T

    confidence = np.ones((137, 1), dtype=np.float32)
    kp2d = np.concatenate([jp_orig, confidence], axis=1).astype(np.float32)

    return {
        "kp2d": kp2d,
        "joint_cam": joint_cam.astype(np.float32),
        "root_cam": root_cam.astype(np.float32),
        "inv_trans": inv_trans.astype(np.float32),
    }


def project_face_3d_to_2d(joint_cam_face, cfg, inv_trans, root_cam):
    """
    Project face 3D joints (root-relative) to original image 2D coords.

    Parameters
    ----------
    joint_cam_face : (N, 3) float32 -- root-relative 3D joints
    cfg            : SMPLest-X Config
    inv_trans      : (2, 3) affine
    root_cam       : (3,) float32

    Returns
    -------
    (N, 2) float32 -- pixel coords in original image space
    """
    joints_abs = joint_cam_face + root_cam[None, :]

    focal = cfg.model.focal
    princpt = cfg.model.princpt
    input_body = cfg.model.input_body_shape
    input_img = cfg.model.input_img_shape

    z = joints_abs[:, 2] + 1e-4
    x_body = joints_abs[:, 0] / z * focal[0] + princpt[0]
    y_body = joints_abs[:, 1] / z * focal[1] + princpt[1]

    x_px = x_body / input_body[1] * input_img[1]
    y_px = y_body / input_body[0] * input_img[0]

    pts = np.stack([x_px, y_px, np.ones_like(x_px)], axis=1)
    pts_orig = (inv_trans @ pts.T).T
    return pts_orig.astype(np.float32)


def interpolate_timeline(timeline, timeline_3d=None, cfg=None):
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

        timeline[t] = timeline[t0] + alpha * (timeline[t1] - timeline[t0])

        if use_3d_face and timeline_3d[t0] is not None and timeline_3d[t1] is not None:
            cam0 = timeline_3d[t0]["joint_cam"][65:137]
            cam1 = timeline_3d[t1]["joint_cam"][65:137]
            face_3d_interp = cam0 + alpha * (cam1 - cam0)

            src = timeline_3d[t0] if alpha < 0.5 else timeline_3d[t1]
            inv_trans = src["inv_trans"]
            root_cam = src["root_cam"]
            face_2d = project_face_3d_to_2d(face_3d_interp, cfg, inv_trans, root_cam)
            timeline[t][65:137, :2] = face_2d

            root_cam_interp = (timeline_3d[t0]["root_cam"]
                               + alpha * (timeline_3d[t1]["root_cam"]
                                          - timeline_3d[t0]["root_cam"]))
            timeline_3d[t] = {
                "joint_cam": timeline_3d[t0]["joint_cam"] + alpha * (
                    timeline_3d[t1]["joint_cam"] - timeline_3d[t0]["joint_cam"]),
                "root_cam": root_cam_interp,
                "inv_trans": inv_trans,
            }


def resample_keypoints(timeline, fps_in, target_fps=30.0):
    """
    Resample a single-person keypoint timeline from fps_in to target_fps
    using linear interpolation between adjacent frames.

    Parameters
    ----------
    timeline : list of (K, 3) arrays or None
    fps_in : float
    target_fps : float

    Returns
    -------
    resampled : list of (K, 3) arrays or None
    src_indices : list of int, nearest source frame index per output frame
    """
    n_in = len(timeline)
    if n_in < 2:
        return list(timeline), list(range(n_in))

    duration = (n_in - 1) / fps_in
    n_out = max(1, int(round(duration * target_fps)) + 1)

    resampled = []
    src_indices = []
    for i in range(n_out):
        t_sec = i / target_fps
        t_in = t_sec * fps_in

        j0 = min(int(t_in), n_in - 1)
        j1 = min(j0 + 1, n_in - 1)
        alpha = t_in - j0

        src_indices.append(min(int(round(t_in)), n_in - 1))

        if j0 == j1 or alpha < 1e-6:
            resampled.append(
                timeline[j0].copy() if timeline[j0] is not None else None)
        elif timeline[j0] is not None and timeline[j1] is not None:
            resampled.append(
                timeline[j0] * (1 - alpha) + timeline[j1] * alpha)
        elif timeline[j0] is not None:
            resampled.append(timeline[j0].copy())
        elif timeline[j1] is not None:
            resampled.append(timeline[j1].copy())
        else:
            resampled.append(None)

    return resampled, src_indices


def compute_resampled_indices(n_in, fps_in, target_fps=30.0):
    """
    Compute nearest source frame indices for resampling from fps_in to
    target_fps. Works for both upsampling and downsampling.

    Returns list of source frame indices (length = output frame count).
    """
    if n_in < 2:
        return list(range(n_in))

    duration = (n_in - 1) / fps_in
    n_out = max(1, int(round(duration * target_fps)) + 1)

    indices = []
    for i in range(n_out):
        t_sec = i / target_fps
        j = min(int(round(t_sec * fps_in)), n_in - 1)
        indices.append(j)

    return indices


def fill_nearest(timeline):
    """Fill remaining None gaps with nearest-neighbour copy."""
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
