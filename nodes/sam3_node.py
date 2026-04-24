"""
SAM3 Video Segmentation nodes (Ultralytics backend).

LoadSAM3 loads the SAM3 model via Ultralytics' SAM3VideoSemanticPredictor.
SAM3VideoSegmentation runs text-prompted video segmentation and outputs
frame-aligned masks suitable for PromptHMR / Sapiens downstream nodes.

Output masks have shape (B * N, H, W) where B = number of frames and
N = number of tracked persons across the whole video.  Every frame has
exactly N mask slots; frames where a person is not visible get a zero
mask.

Model checkpoint (``sam3.pt``) is loaded from ``models/sam3/``.
"""

import glob
import logging
import os

import numpy as np
from ._mask_utils import build_debug_overlay, pack_mask, unpack_mask
import torch

import comfy.utils
from folder_paths import models_dir

# Checkpoint directory under ComfyUI models/
SAM3_CKPT_DIR = os.path.join(models_dir, "sam3")
os.makedirs(SAM3_CKPT_DIR, exist_ok=True)

_logger = logging.getLogger(__name__)


class TensorVideoLoader:
    """Make a BCHW tensor batch look like an ultralytics video dataset.

    Ultralytics' built-in LoadTensor hardcodes mode="image" and yields
    all frames in a single batch.  SAM3VideoSemanticPredictor.init_state
    asserts dataset.mode == "video" and iterates frame-by-frame with
    dataset.frame / dataset.frames bookkeeping, so neither file-path
    loaders nor LoadTensor fit.

    This loader implements the minimal interface SAM3 video needs:
        - mode == "video"
        - frames: total frame count
        - frame:  current frame index (updated during iteration)
        - __iter__ / __next__ yielding one frame as
          ([path], [im0_hwc_bgr_uint8], [info_str]) — matching what
          LoadImagesAndVideos yields from cv2.VideoCapture.read().
          Ultralytics' preprocess() then runs letterbox + normalize +
          to-tensor, which is what SAM3's positional encodings expect.
    """

    def __init__(self, tensor, fps=30):
        # tensor: (B, C, H, W) float [0, 1], any device
        import numpy as _np
        # Pre-convert once to HWC RGB uint8 on CPU, then swap to BGR
        rgb_u8 = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()  # (B, C, H, W)
        # BCHW → BHWC
        hwc = _np.transpose(rgb_u8, (0, 2, 3, 1))
        # RGB → BGR (ultralytics convention from cv2)
        self.frames_bgr = hwc[..., ::-1].copy()

        self.bs = 1
        self.mode = "video"
        self.frames = int(tensor.shape[0])
        self.fps = fps
        self.frame = 0
        self.count = 0
        self.paths = [f"frame_{i}.jpg" for i in range(self.frames)]

    def __iter__(self):
        self.count = 0
        self.frame = 0
        return self

    def __next__(self):
        if self.count >= self.frames:
            raise StopIteration
        im0 = self.frames_bgr[self.count]   # (H, W, 3) BGR uint8
        path = self.paths[self.count]
        self.frame = self.count
        self.count += 1
        return [path], [im0], [""]

    def __len__(self):
        return self.frames


def _list_checkpoints():
    """Return basenames of available SAM3 checkpoint files."""
    patterns = ["*.pt", "*.pth", "*.safetensors"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(SAM3_CKPT_DIR, pat)))
    basenames = sorted(set(os.path.basename(f) for f in files))
    if not basenames:
        basenames = ["(no checkpoints found)"]
    return basenames


class LoadSAM3Node:
    """Load a SAM3 video segmentation model (Ultralytics backend).

    Checkpoint files should be placed in ``models/sam3/``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_list_checkpoints(),),
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 256,
                        "max": 1024,
                        "step": 32,
                        "tooltip": (
                            "Inference resolution. Lower = faster but less "
                            "accurate. SAM3 is internally fixed at 1008 in "
                            "the official code; ultralytics exposes this."
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

    RETURN_TYPES = ("SAM3",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, checkpoint, imgsz, half, conf):
        try:
            from ultralytics.models.sam.predict import (
                SAM3VideoSemanticPredictor,
            )
        except ImportError as e:
            raise ImportError(
                "Ultralytics is required for SAM3. Install with:\n"
                "  pip install -U ultralytics"
            ) from e

        path = os.path.join(SAM3_CKPT_DIR, checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {path}\n"
                f"Please place sam3.pt in {SAM3_CKPT_DIR}/"
            )

        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            imgsz=imgsz,
            model=path,
            half=half,
            save=False,
            verbose=False,
        )
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        _logger.info(
            "Loaded SAM3 (ultralytics): %s (imgsz=%d, half=%s)",
            checkpoint, imgsz, half,
        )
        return ({
            "predictor": predictor,
            "imgsz": imgsz,
            "half": half,
            "conf": conf,
            "checkpoint_path": path,  # for lazy image-predictor building
            "_image_predictor": None,
        },)


class SAM3VideoSegmentationNode:
    """Run SAM3 text-prompted video segmentation (Ultralytics backend).

    Outputs aligned masks (B * N, H, W) where every frame has exactly
    N object slots.  Uses ultralytics' built-in tracking IDs for
    cross-frame alignment — no manual slot mapping needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sam3": ("SAM3",),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "person",
                        "tooltip": (
                            "Text description of objects to segment. "
                            "Use comma to segment multiple types "
                            "(e.g. 'person, dog')."
                        ),
                    },
                ),
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Output a second IMAGE where each tracked "
                            "object's mask is color-coded and alpha-"
                            "blended onto the original frames, with a "
                            "matching bbox outline + tid text label. "
                            "Off by default (adds a CPU pass per "
                            "frame). When off, the debug_overlay "
                            "output is just the input images passed "
                            "through unchanged."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "debug_overlay")
    FUNCTION = "segment"
    CATEGORY = "4dhumans"

    def segment(self, images, sam3, text_prompt, debug_overlay=False):
        predictor = sam3["predictor"]

        # images: (B, H, W, 3) float [0, 1]
        B, H, W, _C = images.shape

        # Split comma-separated prompts
        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        if not prompts:
            prompts = ["person"]

        # Convert to BCHW RGB float32 [0, 1] for ultralytics
        source = images.permute(0, 3, 1, 2).contiguous().float()

        pbar = comfy.utils.ProgressBar(B)

        # ------------------------------------------------------------------
        # SAM3VideoSemanticPredictor requires dataset.mode == "video".
        # Ultralytics' default LoadTensor hardcodes "image", which trips
        # the assertion in init_state.  We monkey-patch setup_source on
        # this predictor to swap in our TensorVideoLoader after the
        # built-in path runs (so other predictor state stays initialized).
        # ------------------------------------------------------------------
        original_setup_source = predictor.setup_source

        def patched_setup_source(src):
            original_setup_source(src)
            if isinstance(src, torch.Tensor):
                predictor.dataset = TensorVideoLoader(src)
                predictor.bs = 1

        predictor.setup_source = patched_setup_source

        try:
            # Ensure model is on GPU (previous run offloaded to CPU)
            from .sam3_image_node import _restore_predictor_to_cuda
            _restore_predictor_to_cuda(predictor)
            results_iter = predictor(
                source=source,
                text=prompts,
                stream=True,
            )

            # Collect per-frame masks indexed by global track_id.
            per_frame = [{} for _ in range(B)]
            all_track_ids = set()

            frame_idx = 0
            logged_shape = False
            for r in results_iter:
                if frame_idx >= B:
                    break

                if r.masks is not None and r.boxes is not None:
                    masks_data = r.masks.data  # (N, H', W')
                    ids = r.boxes.id           # (N,) or None

                    # Log once so users can diagnose shape mismatches
                    if not logged_shape:
                        _logger.info(
                            "SAM3 mask shape: %s, dtype: %s, range: [%.2f, %.2f], "
                            "image shape: (%d, %d)",
                            tuple(masks_data.shape), masks_data.dtype,
                            float(masks_data.min()), float(masks_data.max()),
                            H, W,
                        )
                        logged_shape = True

                    if ids is None:
                        ids = torch.arange(masks_data.shape[0])

                    # Resize masks back to original (H, W) if needed.
                    # Use bilinear here to avoid nearest-neighbour blockiness
                    # when ultralytics returns masks at the letterbox size.
                    if masks_data.shape[-2:] != (H, W):
                        masks_data = torch.nn.functional.interpolate(
                            masks_data.unsqueeze(1).float(),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)

                    # Bit-pack masks: 1/32 the RAM of float32, 1/8 the
                    # RAM of plain bool.  Long HD videos with many
                    # tracked persons can easily OOM without this.
                    masks_np = (masks_data > 0.5).cpu().numpy()  # bool
                    ids_np = ids.cpu().numpy().astype(int)

                    for i, tid in enumerate(ids_np):
                        per_frame[frame_idx][int(tid)] = pack_mask(masks_np[i])
                        all_track_ids.add(int(tid))

                frame_idx += 1
                pbar.update(1)
        finally:
            # Restore original setup_source to avoid polluting the predictor
            predictor.setup_source = original_setup_source

            # Offload SAM3 model to CPU so downstream nodes
            # (PromptHMR + Sapiens) have VRAM available.
            from .sam3_image_node import _offload_sam3_to_cpu
            _offload_sam3_to_cpu(sam3)

        if not all_track_ids:
            _logger.warning(
                "SAM3 detected nothing for prompts %s. "
                "Returning single empty mask per frame.",
                prompts,
            )
            # Pass-through for the debug output so downstream graph
            # wiring stays valid when nothing is detected.
            return (torch.zeros(B, H, W), images)

        # Map track IDs → contiguous slot indices (0, 1, 2, ...)
        sorted_ids = sorted(all_track_ids)
        id_to_slot = {tid: i for i, tid in enumerate(sorted_ids)}
        n_persons = len(sorted_ids)

        # Build aligned output tensor — same layout as other seg nodes.
        # Allocate the final float32 tensor once and unpack packed
        # masks directly into it.  Peak RAM ≈ sizeof(masks_out).
        masks_out = torch.zeros(B * n_persons, H, W, dtype=torch.float32)
        for t in range(B):
            for tid, packed in per_frame[t].items():
                slot = id_to_slot[tid]
                mask_bool = unpack_mask(packed, H, W)
                masks_out[t * n_persons + slot] = torch.from_numpy(
                    mask_bool.astype(np.float32)
                )

        # Build the colored-mask overlay before dropping per_frame.
        # When the toggle is off, pass the input images through so the
        # output signature stays consistent for downstream wiring.
        if debug_overlay:
            per_frame_items = [
                list(per_frame[t].items()) for t in range(B)
            ]
            overlay_out, legend = build_debug_overlay(
                images=images,
                per_frame_items=per_frame_items,
                H=H, W=W,
            )
            _logger.info("SAM3 video debug overlay legend: %s", legend)
        else:
            overlay_out = images

        del per_frame

        _logger.info(
            "SAM3 (ultralytics): %d frames, %d prompts, %d tracked objects, "
            "output shape %s",
            B, len(prompts), n_persons, tuple(masks_out.shape),
        )

        return (masks_out, overlay_out)
