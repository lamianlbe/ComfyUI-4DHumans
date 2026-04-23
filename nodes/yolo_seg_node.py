"""
YOLO Instance Segmentation nodes (Ultralytics backend).

LoadYOLO loads a YOLO segmentation checkpoint (e.g. yolo11x-seg.pt,
yolo26x-seg.pt) from ``models/yolo/``.

YOLOInstanceSegmentation runs per-frame segmentation on an image batch
with built-in ByteTrack / BoT-SORT tracking, filters detections to the
requested COCO class IDs (default: 0 = person), and outputs masks in
the same ``(B * N, H, W)`` frame-grouped layout that the SAM3 nodes
emit — so downstream nodes (PromptHMR, Sapiens) accept it with no
changes.
"""

import logging
import os

import numpy as np
import torch

import comfy.utils
from folder_paths import models_dir

# Hardcoded checkpoint path: <ComfyUI models/>/reface/yolo26x-seg.pt
YOLO_CKPT_PATH = os.path.join(models_dir, "reface", "yolo26x-seg.pt")

_logger = logging.getLogger(__name__)


class LoadYOLONode:
    """Load the YOLO26-x segmentation model (Ultralytics backend).

    Checkpoint is hardcoded to ``models/reface/yolo26x-seg.pt``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 256,
                        "max": 1280,
                        "step": 32,
                        "tooltip": (
                            "Inference resolution. 640 is YOLO's default; "
                            "higher captures small objects better."
                        ),
                    },
                ),
                "half": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use FP16 inference (faster on modern GPUs).",
                    },
                ),
                "conf": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Detection confidence threshold.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("YOLO",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, imgsz, half, conf):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics is required for YOLO. Install with:\n"
                "  pip install -U ultralytics"
            ) from e

        path = YOLO_CKPT_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"YOLO checkpoint not found: {path}\n"
                f"Please place yolo26x-seg.pt at that location."
            )

        model = YOLO(path)
        _logger.info(
            "Loaded YOLO26-x: %s (imgsz=%d, half=%s)",
            path, imgsz, half,
        )
        return ({
            "model": model,
            "imgsz": imgsz,
            "half": half,
            "conf": conf,
            "checkpoint_path": path,
        },)


class YOLOInstanceSegmentationNode:
    """Run YOLO instance segmentation + tracking, filtered by class IDs.

    Output masks have shape ``(B * N, H, W)`` frame-grouped, where N is
    the number of unique tracked objects across the whole video.  Every
    frame has exactly N slots; objects not visible in a frame get zero
    masks.  This format is identical to the SAM3 nodes, so downstream
    PromptHMR / Sapiens nodes work without modification.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "yolo": ("YOLO",),
                "class_ids": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": (
                            "Comma-separated COCO class IDs to keep. "
                            "0=person, 16=dog, 17=cat. "
                            "Default 0 keeps only people."
                        ),
                    },
                ),
                "tracker": (
                    ["bytetrack.yaml", "botsort.yaml"],
                    {
                        "default": "bytetrack.yaml",
                        "tooltip": (
                            "Multi-object tracker. ByteTrack is faster; "
                            "BoT-SORT is more robust to occlusion."
                        ),
                    },
                ),
                "iou": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "NMS IoU threshold for suppressing duplicate boxes.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    # Per-chunk frame count — tuned to cap peak VRAM on long HD videos.
    # At 720p / bf16, a chunk of 128 uses well under 10 GB including
    # activations, tracker state, and per-frame mask accumulators.
    _CHUNK_SIZE = 128

    def segment(self, images, yolo, class_ids, tracker, iou):
        chunk_size = self._CHUNK_SIZE
        model = yolo["model"]

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # Parse comma-separated class IDs
        try:
            class_list = [
                int(c.strip()) for c in class_ids.split(",") if c.strip()
            ]
        except ValueError:
            raise ValueError(
                f"Invalid class_ids '{class_ids}'. "
                "Expected comma-separated integers (e.g. '0' or '0,16')."
            )
        if not class_list:
            class_list = [0]  # default to person

        # Convert to BCHW RGB float32 [0, 1].  Keep on CPU — we slice
        # it per chunk and let ultralytics move each chunk to GPU.
        # (Sending the whole (B, 3, H, W) tensor in one call makes
        # ultralytics allocate ~B * H * W * 12 bytes of VRAM up front.)
        source_cpu = images.permute(0, 3, 1, 2).contiguous().float()

        pbar = comfy.utils.ProgressBar(B)

        # Collect per-frame masks indexed by global track_id
        per_frame = [{} for _ in range(B)]
        all_track_ids = set()
        fallback_id = 10_000_000  # for frames where tracker didn't assign IDs
        logged_shape = False

        try:
            # Ensure model is on GPU (previous run offloaded to CPU)
            if torch.cuda.is_available():
                try:
                    if hasattr(model, "model") and model.model is not None:
                        model.model.to("cuda")
                except Exception as e:
                    _logger.warning("YOLO restore-to-CUDA failed: %s", e)

            # Process the video in chunks to bound peak VRAM.  Use
            # persist=True so ByteTrack / BoT-SORT carry their state
            # across chunk boundaries — track IDs stay consistent end
            # to end as if it were a single stream.
            t = 0
            for chunk_start in range(0, B, chunk_size):
                chunk_end = min(chunk_start + chunk_size, B)
                chunk = source_cpu[chunk_start:chunk_end]

                results_iter = model.track(
                    source=chunk,
                    stream=True,
                    imgsz=yolo["imgsz"],
                    conf=yolo["conf"],
                    iou=iou,
                    half=yolo["half"],
                    classes=class_list,
                    tracker=tracker,
                    persist=(chunk_start > 0),
                    verbose=False,
                )

                for r in results_iter:
                    if t >= B:
                        break

                    if r.masks is None or r.boxes is None:
                        t += 1
                        pbar.update(1)
                        continue

                    masks_data = r.masks.data      # (N, H', W')
                    ids = r.boxes.id                # (N,) or None

                    if not logged_shape:
                        _logger.info(
                            "YOLO mask shape: %s, dtype: %s, range: [%.2f, %.2f], "
                            "image shape: (%d, %d), chunk_size=%d",
                            tuple(masks_data.shape), masks_data.dtype,
                            float(masks_data.min()), float(masks_data.max()),
                            H, W, chunk_size,
                        )
                        logged_shape = True

                    # Tracker may skip IDs on noisy detections; assign
                    # one-shot placeholder IDs so those masks still
                    # appear in their own slot rather than being merged.
                    if ids is None:
                        ids = torch.arange(masks_data.shape[0]) + fallback_id
                        fallback_id += masks_data.shape[0]

                    # Resize masks back to original (H, W) if needed
                    if masks_data.shape[-2:] != (H, W):
                        masks_data = torch.nn.functional.interpolate(
                            masks_data.unsqueeze(1).float(),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)

                    masks_np = (masks_data > 0.5).cpu().numpy().astype(np.float32)
                    ids_np = ids.cpu().numpy().astype(int)

                    for i, tid in enumerate(ids_np):
                        per_frame[t][int(tid)] = masks_np[i]
                        all_track_ids.add(int(tid))

                    t += 1
                    pbar.update(1)

                # Force-release per-chunk intermediate tensors (Results
                # objects, heatmaps, resized masks still on GPU) before
                # starting the next chunk.
                del results_iter
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            # Offload YOLO to CPU so downstream nodes have VRAM available
            try:
                if hasattr(model, "model") and model.model is not None:
                    model.model.to("cpu")
            except Exception as e:
                _logger.warning("YOLO VRAM offload failed: %s", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_track_ids:
            _logger.warning(
                "YOLO detected no objects matching classes=%s. "
                "Returning single empty mask per frame.",
                class_list,
            )
            return (torch.zeros(B, H, W),)

        # Map track IDs → contiguous slot indices (0, 1, 2, ...)
        sorted_ids = sorted(all_track_ids)
        id_to_slot = {tid: i for i, tid in enumerate(sorted_ids)}
        n_persons = len(sorted_ids)

        # Build aligned output tensor — same layout as SAM3
        masks_out = torch.zeros(B, n_persons, H, W)
        for t in range(B):
            for tid, mask_np in per_frame[t].items():
                slot = id_to_slot[tid]
                masks_out[t, slot] = torch.from_numpy(mask_np)

        masks_out = masks_out.reshape(B * n_persons, H, W)

        _logger.info(
            "YOLO segmentation: %d frames, classes=%s, %d tracked objects, "
            "output shape %s",
            B, class_list, n_persons, tuple(masks_out.shape),
        )

        return (masks_out,)
