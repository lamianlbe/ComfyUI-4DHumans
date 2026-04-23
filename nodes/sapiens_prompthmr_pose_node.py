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

import logging
import time

import numpy as np
import torch
import comfy.utils

from ..humans4d.hmr2.utils.sapiens_inference import run_sapiens_on_bbox

_logger = logging.getLogger(__name__)


def _cuda_sync():
    """Force GPU sync so timing measurements are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _linear_interp_timeline(timeline):
    """In-place fill None entries by linearly interpolating neighbours.

    `timeline` is a list of np.ndarray or None. Gaps between valid
    entries are filled with linear interpolation.  Leading/trailing
    Nones are left as-is (no extrapolation).
    """
    n = len(timeline)
    valid = [i for i, v in enumerate(timeline) if v is not None]
    if len(valid) < 2:
        return
    for a, b in zip(valid[:-1], valid[1:]):
        if b - a <= 1:
            continue
        va = timeline[a]
        vb = timeline[b]
        for i in range(a + 1, b):
            alpha = (i - a) / (b - a)
            timeline[i] = va * (1.0 - alpha) + vb * alpha


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
                "pose_3d_model": ("POSE3D",),
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
                "batch_size": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": (
                            "Cross-frame batch size, shared by PromptHMR "
                            "and Sapiens. Higher = faster on large GPUs; "
                            "reduce if OOM."
                        ),
                    },
                ),
                "pose_fps": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": (
                            "Target sampling rate for both the 3D "
                            "backbone (PromptHMR / NLF) and the 2D "
                            "Sapiens pass. Frames between samples are "
                            "filled by linear interpolation. Lower = "
                            "faster but slightly coarser motion. Set "
                            ">= source fps to disable (run every frame)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("POSES",)
    RETURN_NAMES = ("poses",)
    FUNCTION = "estimate_pose"
    CATEGORY = "4dhumans"

    def estimate_pose(self, images, masks, pose_3d_model, sapiens, fps,
                      batch_size=16, pose_fps=15.0, **kwargs):
        # Backward-compat: older serialised workflows may still send the
        # legacy param names pose_3d_fps or prompthmr_fps.
        if "pose_3d_fps" in kwargs:
            pose_fps = kwargs.pop("pose_3d_fps")
        if "prompthmr_fps" in kwargs:
            pose_fps = kwargs.pop("prompthmr_fps")
        backend = pose_3d_model.get("backend", "prompthmr")
        if backend == "prompthmr":
            from .load_prompthmr_node import _ensure_lib_importable
            _ensure_lib_importable()
            from prompt_hmr.models.inference import prepare_batch  # noqa: F401

        from torch.amp import autocast

        model = pose_3d_model["model"]
        img_size = pose_3d_model.get("img_size", 896)
        # Autocast dtype for forward pass. Default is float16 to match the
        # historical autocast("cuda") behaviour when no dtype was set.
        phmr_dtype = pose_3d_model.get("torch_dtype", torch.float16)

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

        pbar = comfy.utils.ProgressBar(B)

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

        # Sapiens model/preprocessor (for batched inference)
        sap_model = sapiens["model"]
        sap_preproc = sapiens["preprocessor"]
        sap_device = sapiens["device"]
        sap_dtype = sapiens["dtype"]

        # ----- Pre-convert all images and masks once (no per-frame .cpu()) -----
        images_np = (rgb * 255).byte().cpu().numpy()          # (B, H, W, 3)
        masks_np = masks.cpu().numpy()                         # (B, N, H, W)

        # Collect Sapiens requests across all frames for cross-frame batching.
        # Each entry: (frame_idx, person_idx, preproc_tensor, (x1,y1,w,h))
        sap_requests = []

        # Timing accumulators (GPU-synchronised).
        phmr_time_s = 0.0
        sapiens_time_s = 0.0

        # ----- Determine sampled frames (shared by 3D and 2D) -----
        # Stride ≥ 1; if target >= source, stride=1 (no skipping).
        if pose_fps > 0 and pose_fps < fps:
            phmr_stride = max(1, int(round(float(fps) / float(pose_fps))))
        else:
            phmr_stride = 1
        sampled_frames_set = set(range(0, B, phmr_stride))
        # Always include the last frame so interpolation has an anchor
        # at the end of the clip.
        if B > 0:
            sampled_frames_set.add(B - 1)

        # ----- Phase 1a: Gather per-frame 3D inputs + Sapiens crops -----
        # Both passes only process sampled frames; interpolation fills
        # the rest after inference.
        phmr_frame_inputs = [None] * B
        for t in range(B):
            img_np = images_np[t]
            masks_frame_np = masks_np[t]
            is_sampled = t in sampled_frames_set

            bboxes_phmr = []
            masks_uint8 = []
            person_indices = []

            for p_idx in range(n_persons):
                mask_frame = masks_frame_np[p_idx]
                ys, xs = np.where(mask_frame > 0.5)
                if len(xs) == 0:
                    continue

                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())

                if not is_sampled:
                    continue

                bboxes_phmr.append(
                    torch.tensor(
                        [[x1, y1, x2, y2, 1.0]], dtype=torch.float32
                    )
                )
                masks_uint8.append(
                    (mask_frame > 0.5).astype(np.uint8) * 255
                )
                person_indices.append(p_idx)

                # Sapiens crop for Pass 2 (same sampled frames only)
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(img_w, x2)
                y2c = min(img_h, y2)
                cropped = img_np[y1c:y2c, x1c:x2c].copy()
                crop_mask = mask_frame[y1c:y2c, x1c:x2c] > 0.5
                cropped[~crop_mask] = 0
                sap_requests.append((
                    t, p_idx,
                    sap_preproc(cropped),
                    (x1c, y1c, x2c - x1c, y2c - y1c),
                ))

            if not person_indices:
                continue

            phmr_frame_inputs[t] = {
                "input": {
                    "image_cv": img_np,
                    "boxes": torch.cat(bboxes_phmr, dim=0),
                    "masks": np.array(masks_uint8),
                    "text": None,
                },
                "person_indices": person_indices,
            }

        # ----- Phase 1b: 3D pose cross-frame batched inference -----
        valid_frames = [t for t in range(B) if phmr_frame_inputs[t] is not None]

        if backend == "prompthmr":
            for chunk_start in range(0, len(valid_frames), batch_size):
                chunk_frames = valid_frames[chunk_start:chunk_start + batch_size]
                inputs = [phmr_frame_inputs[t]["input"] for t in chunk_frames]

                _cuda_sync()
                phmr_start = time.perf_counter()
                with torch.no_grad(), autocast("cuda", dtype=phmr_dtype):
                    batch = prepare_batch(
                        inputs, img_size=img_size, interaction=False
                    )
                    outputs = model(batch, use_mean_hands=True)
                _cuda_sync()
                phmr_time_s += time.perf_counter() - phmr_start

                for chunk_i, t in enumerate(chunk_frames):
                    output = outputs[chunk_i]
                    person_indices = phmr_frame_inputs[t]["person_indices"]

                    joints_2d = output["body_joints2d"].detach().cpu().numpy()
                    joints_3d = output["body_joints"].detach().cpu().numpy()
                    smpl_j3d = output["smpl_j3d"].detach().cpu().numpy()

                    scale_val = batch[chunk_i]["image_scale"]
                    offset_val = batch[chunk_i]["image_offset"]
                    if isinstance(offset_val, torch.Tensor):
                        offset_val = offset_val.numpy()
                    offset_val = np.array(offset_val)

                    cam_int_val = output["cam_int"].detach().cpu().numpy()
                    cam_int_per_frame[t] = cam_int_val[0]
                    scale_per_frame[t] = float(scale_val)
                    offset_per_frame[t] = offset_val.copy()

                    joints_2d_orig = (
                        joints_2d - offset_val[None, None, :]
                    ) / scale_val

                    for i, p_idx in enumerate(person_indices):
                        persons[p_idx]["body_joints2d"][t] = joints_2d_orig[i]
                        persons[p_idx]["body_joints"][t] = joints_3d[i]
                        persons[p_idx]["smpl_j3d"][t] = smpl_j3d[i]

                    pbar.update(1)

                # Skipped frames (no valid persons) still need pbar updates
                # when the chunk is shorter than batch_size at the tail.
            skipped = B - len(valid_frames)
            for _ in range(skipped):
                pbar.update(1)
        elif backend == "nlf":
            from ._nlf_inference import run_nlf_inference
            phmr_time_s += run_nlf_inference(
                model=model,
                images_np=images_np,
                masks_np=masks_np,
                valid_frames=valid_frames,
                phmr_frame_inputs=phmr_frame_inputs,
                persons=persons,
                cam_int_per_frame=cam_int_per_frame,
                scale_per_frame=scale_per_frame,
                offset_per_frame=offset_per_frame,
                batch_size=batch_size,
                dtype=phmr_dtype,
                img_h=img_h,
                img_w=img_w,
                pbar=pbar,
                B=B,
            )
        else:
            raise ValueError(
                f"Unknown 3D pose backend: {backend!r}. "
                "Expected 'prompthmr' or 'nlf'."
            )

        # ----- Phase 1c: Fill non-sampled frames by linear interpolation -----
        # (only runs when pose_fps < fps)
        if phmr_stride > 1:
            # Per-person 3D fields
            for p_idx in range(n_persons):
                person = persons[p_idx]
                _linear_interp_timeline(person["body_joints2d"])
                _linear_interp_timeline(person["body_joints"])
                _linear_interp_timeline(person["smpl_j3d"])

            # Global per-frame camera parameters (same for all persons)
            _linear_interp_timeline(cam_int_per_frame)
            # scale is a list of floats; wrap to arrays for interp, then unwrap
            scale_arr = [
                np.array([v], dtype=np.float64) if v is not None else None
                for v in scale_per_frame
            ]
            _linear_interp_timeline(scale_arr)
            scale_per_frame[:] = [
                float(v[0]) if v is not None else None for v in scale_arr
            ]
            _linear_interp_timeline(offset_per_frame)

        # ----- Pass 2: Sapiens cross-frame batched inference -----
        if sap_requests:
            n_total = len(sap_requests)
            sap_pbar = comfy.utils.ProgressBar(n_total)

            K_sap = hm_h = hm_w = None

            for chunk_start in range(0, n_total, batch_size):
                chunk = sap_requests[chunk_start:chunk_start + batch_size]
                tensors = [r[2] for r in chunk]

                sap_batch = torch.stack(tensors, dim=0).to(sap_device).to(sap_dtype)

                _cuda_sync()
                sap_start = time.perf_counter()
                with torch.inference_mode():
                    heatmaps_batch = sap_model(sap_batch).to(torch.float32)
                    N_b, K_sap, hm_h, hm_w = heatmaps_batch.shape

                    # GPU-side argmax — only transfer (N, K) indices back.
                    hm_flat = heatmaps_batch.reshape(N_b, K_sap, -1)
                    max_vals, max_idxs = hm_flat.max(dim=2)
                    y_hm_t = (max_idxs // hm_w).to(torch.float32)
                    x_hm_t = (max_idxs % hm_w).to(torch.float32)

                    y_hm = y_hm_t.cpu().numpy()
                    x_hm = x_hm_t.cpu().numpy()
                    confs = max_vals.cpu().numpy()
                _cuda_sync()
                sapiens_time_s += time.perf_counter() - sap_start

                for i, (frame_t, p_idx, _, (bx, by, bw, bh)) in enumerate(chunk):
                    pixel_kp = np.empty((K_sap, 3), dtype=np.float32)
                    pixel_kp[:, 0] = x_hm[i] * bw / hm_w + bx
                    pixel_kp[:, 1] = y_hm[i] * bh / hm_h + by
                    pixel_kp[:, 2] = confs[i]
                    persons[p_idx]["keypoints"][frame_t] = pixel_kp
                    sap_pbar.update(1)

        # Interpolate Sapiens keypoints on skipped frames (same stride
        # as the 3D backbone).  keypoints are confidence-aware (133, 3):
        # the linear interpolation averages confidences too, which is a
        # reasonable "between-sample" estimate.
        if phmr_stride > 1:
            for p_idx in range(n_persons):
                _linear_interp_timeline(persons[p_idx]["keypoints"])

        # Store raw keypoints for non-destructive editing in Pose Editor.
        # keypoints_raw is the original Sapiens output; keypoints is the
        # "active" version that downstream nodes consume.  The Pose Editor
        # can apply temporal filtering and always re-derives from raw.
        for p_idx in range(n_persons):
            persons[p_idx]["keypoints_raw"] = [
                kp.copy() if kp is not None else None
                for kp in persons[p_idx]["keypoints"]
            ]

        n_sap = len(sap_requests)
        n_phmr = len(valid_frames)
        effective_phmr_fps = fps / phmr_stride
        _logger.info(
            "SapiensPromptHMR Human Pose: %d frames, %d persons, batch=%d | "
            "%s %.2f s (%d frames @ %.1f fps stride=%d, %.1f ms/frame) | "
            "Sapiens %.2f s (%d crops, %.1f ms/crop)",
            B, n_persons, batch_size,
            backend.upper(),
            phmr_time_s, n_phmr, effective_phmr_fps, phmr_stride,
            1000.0 * phmr_time_s / max(1, n_phmr),
            sapiens_time_s, n_sap,
            1000.0 * sapiens_time_s / max(1, n_sap),
        )

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
