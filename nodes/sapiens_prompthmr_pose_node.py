"""
Sapiens PromptHMR Human Pose node.

Merged node that runs both PromptHMR (3D) and Sapiens (2D) on masked
per-person images.  Outputs a unified POSES dict containing all pose
data with per-person visibility flags.

Inputs:
  - images: RGB image batch (B, H, W, 3)
  - masks: frame-grouped masks (B*N, H, W)
  - prompthmr: loaded PromptHMR model
  - sapiens: loaded Sapiens model
  - fps: source video FPS

Output:
  - poses (POSES): unified dict with 2D + 3D data for all persons
"""

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox


def _mask_to_bbox_torch(mask_np):
    """Convert a binary mask (H, W) to a [x1, y1, x2, y2, score] bbox tensor."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return torch.tensor([[x1, y1, x2, y2, 1.0]], dtype=torch.float32)


def _mask_to_bbox_np(mask_np):
    """Convert a binary mask (H, W) to a [x1, y1, x2, y2] bbox array."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


class SapiensPromptHMRPoseNode:
    """Unified PromptHMR 3D + Sapiens 2D pose estimation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "prompthmr": ("PROMPTHMR",),
                "sapiens": ("SAPIENS",),
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
            },
        }

    RETURN_TYPES = ("POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "estimate_pose"
    CATEGORY = "4dhumans"

    def estimate_pose(self, images, masks, prompthmr, sapiens, fps):
        from .load_prompthmr_node import _ensure_lib_importable
        _ensure_lib_importable()

        from torch.amp import autocast
        from prompt_hmr.models.inference import prepare_batch

        model = prompthmr["model"]
        img_size = prompthmr["img_size"]

        B, img_h, img_w, C = images.shape
        rgb = images[..., :3]  # (B, H, W, 3)

        # masks: (M, H, W) frame-grouped
        if masks.dim() == 4:
            masks = masks[..., 0]
        M = masks.shape[0]
        if M % B != 0:
            raise ValueError(
                f"Sapiens PromptHMR Human Pose: mask count ({M}) must be a "
                f"multiple of frame count ({B}). Got {M} masks for {B} frames."
            )
        n_persons = M // B
        masks = masks.reshape(B, n_persons, masks.shape[-2], masks.shape[-1])

        pbar = comfy.utils.ProgressBar(B * 2)  # PromptHMR pass + Sapiens pass

        # Per-frame camera info
        cam_int_per_frame = [None] * B
        scale_per_frame = [None] * B
        offset_per_frame = [None] * B

        # Per-person storage
        persons = []
        for _ in range(n_persons):
            persons.append({
                "visible": True,
                "body_joints2d": [None] * B,
                "body_joints": [None] * B,
                "smpl_j3d": [None] * B,
                "keypoints": [None] * B,
            })

        # ----- Pass 1: PromptHMR 3D -----
        for t in range(B):
            img_np = (rgb[t] * 255).byte().cpu().numpy()

            bboxes = []
            masks_uint8 = []
            person_indices = []

            for p_idx in range(n_persons):
                mask_frame = masks[t, p_idx].cpu().numpy()
                bbox = _mask_to_bbox_torch(mask_frame)
                if bbox is None:
                    continue
                mask_u8 = (mask_frame > 0.5).astype(np.uint8) * 255
                bboxes.append(bbox)
                masks_uint8.append(mask_u8)
                person_indices.append(p_idx)

            if not bboxes:
                pbar.update(1)
                continue

            boxes_cat = torch.cat(bboxes, dim=0)
            masks_arr = np.array(masks_uint8)

            inputs = [{
                "image_cv": img_np,
                "boxes": boxes_cat,
                "masks": masks_arr,
                "text": None,
            }]

            with torch.no_grad(), autocast("cuda"):
                batch = prepare_batch(inputs, img_size=img_size, interaction=False)
                output = model(batch, use_mean_hands=True)[0]

            joints_2d = output["body_joints2d"].detach().cpu().numpy()
            joints_3d = output["body_joints"].detach().cpu().numpy()
            smpl_j3d = output["smpl_j3d"].detach().cpu().numpy()

            scale_val = batch[0]["image_scale"]
            offset_val = batch[0]["image_offset"]
            if isinstance(offset_val, torch.Tensor):
                offset_val = offset_val.numpy()
            offset_val = np.array(offset_val)

            cam_int_val = output["cam_int"].detach().cpu().numpy()
            cam_int_per_frame[t] = cam_int_val[0]
            scale_per_frame[t] = float(scale_val)
            offset_per_frame[t] = offset_val.copy()

            joints_2d_orig = (joints_2d - offset_val[None, None, :]) / scale_val

            for i, p_idx in enumerate(person_indices):
                persons[p_idx]["body_joints2d"][t] = joints_2d_orig[i]
                persons[p_idx]["body_joints"][t] = joints_3d[i]
                persons[p_idx]["smpl_j3d"][t] = smpl_j3d[i]

            pbar.update(1)

        # ----- Pass 2: Sapiens 2D -----
        for t in range(B):
            img_np = (rgb[t] * 255).byte().cpu().numpy()

            for p_idx in range(n_persons):
                mask_frame = masks[t, p_idx].cpu().numpy()
                bbox = _mask_to_bbox_np(mask_frame)
                if bbox is None:
                    continue

                mask_bool = mask_frame > 0.5
                masked_img = img_np.copy()
                masked_img[~mask_bool] = 0

                result = run_sapiens_on_bbox(masked_img, bbox, sapiens)
                if result is not None:
                    persons[p_idx]["keypoints"][t] = result["pixel_kp"]

            pbar.update(1)

        # ---------------------------------------------------------------
        # Temporal outlier filtering on Sapiens 2D keypoints
        # ---------------------------------------------------------------
        from ._pose_utils import temporal_filter_keypoints

        total_repaired = 0
        for p_idx in range(n_persons):
            kp_timeline = persons[p_idx]["keypoints"]
            filtered, n_rep = temporal_filter_keypoints(
                kp_timeline,
                velocity_threshold=3.0,
                window_size=5,
                smooth_sigma=0.0,
            )
            persons[p_idx]["keypoints"] = filtered
            total_repaired += n_rep

        if total_repaired > 0:
            print(f"[SapiensPromptHMRPose] Temporal filter: "
                  f"repaired {total_repaired} outlier frames across "
                  f"{n_persons} persons")

        poses = {
            "n_persons": n_persons,
            "n_frames": B,
            "img_h": img_h,
            "img_w": img_w,
            "fps": float(fps),
            "persons": persons,
            "cam_int": cam_int_per_frame,
            "scale": scale_per_frame,
            "offset": offset_per_frame,
        }

        return (poses,)
