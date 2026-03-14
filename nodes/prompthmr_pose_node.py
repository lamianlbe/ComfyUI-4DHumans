"""
PromptHMR 3D Human Pose Estimation node.

Accepts an RGB image batch and 1-8 mask batches (one per person).
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
                "mask_1": ("MASK",),
            },
            "optional": {
                "mask_2": ("MASK",),
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
                "mask_5": ("MASK",),
                "mask_6": ("MASK",),
                "mask_7": ("MASK",),
                "mask_8": ("MASK",),
            },
        }

    RETURN_TYPES = ("POSE_3D",)
    RETURN_NAMES = ("pose_3d",)
    FUNCTION = "estimate_pose"
    CATEGORY = "4dhumans"

    def estimate_pose(self, images, prompthmr, mask_1,
                      mask_2=None, mask_3=None, mask_4=None,
                      mask_5=None, mask_6=None, mask_7=None,
                      mask_8=None):
        from .load_prompthmr_node import _ensure_lib_importable
        _ensure_lib_importable()

        from torch.amp import autocast
        from prompt_hmr.models.inference import prepare_batch

        model = prompthmr["model"]
        img_size = prompthmr["img_size"]

        # Collect all connected masks
        all_masks = [mask_1]
        for m in [mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8]:
            if m is not None:
                all_masks.append(m)
        n_persons = len(all_masks)

        B, img_h, img_w, C = images.shape
        rgb = images[..., :3]  # (B, H, W, 3)

        pbar = comfy.utils.ProgressBar(B)

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

            for p_idx, mask_batch in enumerate(all_masks):
                # mask_batch shape: (B, H, W) or (B, H, W, 1)
                if mask_batch.dim() == 4:
                    mask_frame = mask_batch[t, ..., 0].cpu().numpy()
                else:
                    mask_frame = mask_batch[t].cpu().numpy()

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

            # Unscale 2D joints from padded/scaled model space to original image coords
            scale = batch[0]["image_scale"]
            offset = batch[0]["image_offset"]
            if isinstance(offset, torch.Tensor):
                offset = offset.numpy()
            offset = np.array(offset)

            joints_2d_orig = (joints_2d - offset[None, None, :]) / scale

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
        }

        return (pose_3d,)
