"""
SMPLest-X Human Pose Tracking node.

Single-person mode (no PHALP): runs SMPLest-X on the full image per frame,
with temporal interpolation and optional SCAIL 3D rendering.

Multi-person mode (PHALP connected): uses PHALP for detection/tracking,
runs SMPLest-X on each detected person.
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.render_openpose_wholebody import render_wholebody_openpose
from ..humans4d.hmr2.utils.render_openpose_scail import render_scail_pose_batch
from ._pose_utils import (
    run_smplestx_on_bbox,
    interpolate_timeline,
    fill_nearest,
)
from .load_phalp_node import _ensure_phalp_importable


class SMPLestXPoseNode:
    """SMPLest-X 137-joint whole-body pose estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "smplestx": ("SMPLESTX",),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Overlay skeletons on the original frame "
                            "instead of a black canvas."
                        ),
                    },
                ),
                "scail_pose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Render SCAIL-style pose images instead of "
                            "OpenPose style. Uses 3D cylinder rendering."
                        ),
                    },
                ),
            },
            "optional": {
                "phalp": ("PHALP",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render_pose"
    CATEGORY = "4dhumans"

    def render_pose(self, images, smplestx, debug, scail_pose, phalp=None):
        images_nchw = images.permute(0, 3, 1, 2)
        B, _C, img_h, img_w = images_nchw.shape

        if phalp is None:
            return self._single_person(
                images_nchw, B, img_h, img_w, smplestx, debug, scail_pose)
        else:
            return self._multi_person(
                images_nchw, B, img_h, img_w, smplestx, phalp, debug, scail_pose)

    # ------------------------------------------------------------------
    # Single-person mode (no PHALP)
    # ------------------------------------------------------------------

    def _single_person(self, images_nchw, B, img_h, img_w,
                       smplestx, debug, scail_pose):
        whole_bbox = np.array([0, 0, img_w, img_h], dtype=np.float32)
        sx_model = smplestx["model"]
        sx_cfg = smplestx["cfg"]

        pbar = comfy.utils.ProgressBar(2 * B)

        # Pass 1: inference
        timeline = [None] * B
        timeline_3d = [None] * B
        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
            result = run_smplestx_on_bbox(img_np, whole_bbox, sx_model, sx_cfg)
            if result is not None:
                timeline[t] = result["kp2d"].copy()
                timeline_3d[t] = {
                    "joint_cam": result["joint_cam"],
                    "root_cam": result["root_cam"],
                    "inv_trans": result["inv_trans"],
                }
            pbar.update(1)

        # Temporal interpolation + fill
        interpolate_timeline(timeline, timeline_3d, sx_cfg)
        fill_nearest(timeline)

        # Pass 2: render
        if scail_pose:
            scail_frames = render_scail_pose_batch(
                timeline, timeline_3d, img_h, img_w, cfg=sx_cfg)
            pose_images = []
            for t in range(B):
                if debug:
                    canvas = (images_nchw[t].permute(1, 2, 0)
                              * 255).byte().numpy().copy()
                    scail_img = scail_frames[t]
                    mask = scail_img > 0
                    canvas[mask] = scail_img[mask]
                else:
                    canvas = scail_frames[t]
                pose_images.append(
                    torch.from_numpy(canvas.astype(np.float32) / 255.0))
                pbar.update(1)
        else:
            pose_images = []
            for t in range(B):
                if debug:
                    canvas = (images_nchw[t].permute(1, 2, 0)
                              * 255).byte().numpy().copy()
                else:
                    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                if timeline[t] is not None:
                    canvas = render_wholebody_openpose(canvas, timeline[t])
                pose_images.append(
                    torch.from_numpy(canvas.astype(np.float32) / 255.0))
                pbar.update(1)

        return (torch.stack(pose_images),)

    # ------------------------------------------------------------------
    # Multi-person mode (with PHALP)
    # ------------------------------------------------------------------

    def _multi_person(self, images_nchw, B, img_h, img_w,
                      smplestx, phalp, debug, scail_pose):
        if not _ensure_phalp_importable():
            raise RuntimeError("phalp package not found. See 'Load PHALP' node.")

        tracker = phalp["tracker"]
        tracker.setup_deepsort()
        sx_model = smplestx["model"]
        sx_cfg = smplestx["cfg"]

        new_size = max(img_h, img_w)
        left = (new_size - img_w) // 2
        top = (new_size - img_h) // 2
        measurements = [img_h, img_w, new_size, left, top]

        pbar = comfy.utils.ProgressBar(2 * B)

        # Pass 1: detection + inference
        frame_results = []  # per-frame list of SMPLest-X results
        for t in range(B):
            img_np = (images_nchw[t].permute(1, 2, 0) * 255).byte().numpy()
            frame_name = str(t)

            (pred_bbox, pred_bbox_pad, pred_masks,
             pred_scores, pred_classes,
             gt_tids, gt_annots) = tracker.get_detections(
                img_np, frame_name, t, {}, measurements)

            sx_results = []
            for i in range(len(pred_bbox)):
                result = run_smplestx_on_bbox(
                    img_np, pred_bbox[i], sx_model, sx_cfg)
                if result is not None:
                    sx_results.append(result)

            extra_data = tracker.run_additional_models(
                img_np, pred_bbox, pred_masks, pred_scores,
                pred_classes, frame_name, t, measurements, gt_tids, gt_annots)
            detections = tracker.get_human_features(
                img_np, pred_masks, pred_bbox, pred_bbox_pad,
                pred_scores, frame_name, pred_classes, t,
                measurements, gt_tids, gt_annots, extra_data)
            tracker.tracker.predict()
            tracker.tracker.update(detections, t, frame_name, shot=0)

            frame_results.append(sx_results)
            pbar.update(1)

        # Pass 2: render
        if scail_pose:
            # Flatten all person-frames into one batch for SCAIL rendering
            flat_kp2d = []
            flat_3d = []
            flat_frame_idx = []  # which original frame each entry belongs to
            for t in range(B):
                for result in frame_results[t]:
                    flat_kp2d.append(result["kp2d"])
                    flat_3d.append({
                        "joint_cam": result["joint_cam"],
                        "root_cam": result["root_cam"],
                        "inv_trans": result["inv_trans"],
                    })
                    flat_frame_idx.append(t)

            # Single batch render call for all person-frames
            if flat_kp2d:
                scail_all = render_scail_pose_batch(
                    flat_kp2d, flat_3d, img_h, img_w, cfg=sx_cfg)
            else:
                scail_all = []

            # Composite per-person SCAIL images back onto per-frame canvases
            pose_images = []
            scail_by_frame = [[] for _ in range(B)]
            for i, t in enumerate(flat_frame_idx):
                scail_by_frame[t].append(scail_all[i])

            for t in range(B):
                if debug:
                    canvas = (images_nchw[t].permute(1, 2, 0)
                              * 255).byte().numpy().copy()
                else:
                    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                for scail_img in scail_by_frame[t]:
                    mask = scail_img > 0
                    canvas[mask] = scail_img[mask]

                pose_images.append(
                    torch.from_numpy(canvas.astype(np.float32) / 255.0))
                pbar.update(1)
        else:
            pose_images = []
            for t in range(B):
                if debug:
                    canvas = (images_nchw[t].permute(1, 2, 0)
                              * 255).byte().numpy().copy()
                else:
                    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

                for result in frame_results[t]:
                    canvas = render_wholebody_openpose(canvas, result["kp2d"])

                pose_images.append(
                    torch.from_numpy(canvas.astype(np.float32) / 255.0))
                pbar.update(1)

        return (torch.stack(pose_images),)
