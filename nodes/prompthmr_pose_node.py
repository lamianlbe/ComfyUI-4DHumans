"""
PromptHMR 3D Human Pose Estimation node.

Accepts an RGB image batch and a mask batch.  Masks are frame-grouped:
for B frames and N persons the mask tensor has shape (B*N, H, W) with
frame 1's N person masks first, then frame 2's, etc.

Runs PromptHMR to recover 3D human mesh (SMPL-X) and outputs raw
3D pose data as a POSE_3D type for downstream processing.
"""

import numpy as np
import torch
import comfy.utils


def _mask_to_bbox(mask_np):
    """Convert a binary mask (H, W) to a [x1, y1, x2, y2, score] bbox tensor."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return torch.tensor([[x1, y1, x2, y2, 1.0]], dtype=torch.float32)


class PromptHMRPoseNode:
    """PromptHMR 3D pose estimation from RGB images + per-person masks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompthmr": ("PROMPTHMR",),
                "masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("POSE_3D",)
    RETURN_NAMES = ("pose_3d",)
    FUNCTION = "estimate_pose"
    CATEGORY = "4dhumans"

    def estimate_pose(self, images, prompthmr, masks):
        from .load_prompthmr_node import _ensure_lib_importable
        _ensure_lib_importable()

        from torch.amp import autocast
        from prompt_hmr.models.inference import prepare_batch

        model = prompthmr["model"]
        img_size = prompthmr["img_size"]

        B, img_h, img_w, C = images.shape
        rgb = images[..., :3]  # (B, H, W, 3)

        # masks: (M, H, W) frame-grouped: frame1's N persons, frame2's N, ...
        M = masks.shape[0]
        if M % B != 0:
            raise ValueError(
                f"PromptHMR 3D Human Pose: mask count ({M}) must be a "
                f"multiple of frame count ({B}). Got {M} masks for {B} frames."
            )
        n_persons = M // B
        # Reshape to (B, n_persons, H, W)
        if masks.dim() == 4:
            masks = masks[..., 0]  # drop trailing channel dim if present
        masks = masks.reshape(B, n_persons, masks.shape[-2], masks.shape[-1])

        pbar = comfy.utils.ProgressBar(B)

        # Per-frame camera info (for 3D coordinate transforms)
        cam_int_per_frame = [None] * B    # (3, 3) modified cam intrinsic
        scale_per_frame = [None] * B      # scalar
        offset_per_frame = [None] * B     # (2,) array

        # Per-person storage: list of n_persons lists, each with B frames
        persons = []
        for _ in range(n_persons):
            persons.append({
                "body_joints2d": [None] * B,  # (25, 2) per frame
                "body_joints": [None] * B,    # (25, 3) per frame
                "smpl_j3d": [None] * B,       # (24, 3) per frame – SMPL 24-joint 3D
            })

        for t in range(B):
            img_np = (rgb[t] * 255).byte().cpu().numpy()

            # Gather valid person masks for this frame
            bboxes = []
            masks_uint8 = []
            person_indices = []  # which person index each valid detection belongs to

            for p_idx in range(n_persons):
                mask_frame = masks[t, p_idx].cpu().numpy()

                bbox = _mask_to_bbox(mask_frame)
                if bbox is None:
                    continue

                mask_u8 = (mask_frame > 0.5).astype(np.uint8) * 255
                bboxes.append(bbox)
                masks_uint8.append(mask_u8)
                person_indices.append(p_idx)

            if not bboxes:
                pbar.update(1)
                continue

            # Stack all persons for this frame into one batch call
            boxes_cat = torch.cat(bboxes, dim=0)  # (N, 5)
            masks_arr = np.array(masks_uint8)       # (N, H, W)

            inputs = [{
                "image_cv": img_np,
                "boxes": boxes_cat,
                "masks": masks_arr,
                "text": None,
            }]

            with torch.no_grad(), autocast("cuda"):
                batch = prepare_batch(inputs, img_size=img_size, interaction=False)
                output = model(batch, use_mean_hands=True)[0]

            # Extract results
            joints_2d = output["body_joints2d"].detach().cpu().numpy()  # (N, 25, 2)
            joints_3d = output["body_joints"].detach().cpu().numpy()    # (N, 25, 3)
            smpl_j3d = output["smpl_j3d"].detach().cpu().numpy()        # (N, 24, 3)

            # Camera parameters for this frame
            scale_val = batch[0]["image_scale"]
            offset_val = batch[0]["image_offset"]
            if isinstance(offset_val, torch.Tensor):
                offset_val = offset_val.numpy()
            offset_val = np.array(offset_val)

            cam_int_val = output["cam_int"].detach().cpu().numpy()  # (N, 3, 3)
            cam_int_per_frame[t] = cam_int_val[0]  # (3, 3) – same for all persons
            scale_per_frame[t] = float(scale_val)
            offset_per_frame[t] = offset_val.copy()

            # Unscale 2D joints from padded/scaled model space to original image coords
            joints_2d_orig = (joints_2d - offset_val[None, None, :]) / scale_val

            # Store per-person results
            for i, p_idx in enumerate(person_indices):
                persons[p_idx]["body_joints2d"][t] = joints_2d_orig[i]  # (25, 2)
                persons[p_idx]["body_joints"][t] = joints_3d[i]          # (25, 3)
                persons[p_idx]["smpl_j3d"][t] = smpl_j3d[i]              # (24, 3)

            pbar.update(1)

        pose_3d = {
            "n_persons": n_persons,
            "n_frames": B,
            "img_h": img_h,
            "img_w": img_w,
            "persons": persons,
            "cam_int": cam_int_per_frame,    # (3, 3) per frame
            "scale": scale_per_frame,        # scalar per frame
            "offset": offset_per_frame,      # (2,) per frame
        }

        return (pose_3d,)
