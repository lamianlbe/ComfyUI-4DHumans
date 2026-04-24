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

from ._mask_utils import pack_mask, unpack_mask

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
                "debug_overlay": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Output a second IMAGE where each person's "
                            "mask is color-coded and alpha-blended onto "
                            "the original frames, with a matching bbox "
                            "outline + track-id text label. Lets you "
                            "eyeball mask quality and track-ID stability "
                            "directly. Off by default (adds a CPU pass "
                            "per frame). When off, the debug_overlay "
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

    # Per-chunk frame count — tuned to cap peak VRAM on long HD videos.
    # At 720p / bf16, a chunk of 128 uses well under 10 GB including
    # activations, tracker state, and per-frame mask accumulators.
    _CHUNK_SIZE = 128

    # Debug palette (RGB, uint8). One color per tracked-person slot.
    # 12 distinct hues — more than enough for typical scenes; after
    # this we wrap so two tracks may share a color, but the tid= text
    # label still disambiguates them.
    _DEBUG_PALETTE_RGB = np.array([
        [255,  60,  60],   # red
        [ 60, 200,  60],   # green
        [ 60, 120, 255],   # blue
        [255, 200,  40],   # yellow-orange
        [255,  70, 200],   # magenta
        [ 60, 220, 220],   # cyan
        [220, 110, 255],   # purple
        [180, 180,  60],   # olive
        [255, 140,  40],   # orange
        [110, 255, 160],   # mint
        [255, 160, 200],   # pink
        [160, 200, 255],   # light blue
    ], dtype=np.uint8)

    def segment(self, images, yolo, class_ids, tracker, iou,
                debug_overlay=False):
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
            # Ensure model is on GPU (previous run offloaded to CPU).
            # Prefer the wrapper `.to()` so ultralytics' cached `.device`
            # attribute gets refreshed alongside the weight move.
            if torch.cuda.is_available():
                try:
                    if hasattr(model, "to"):
                        model.to("cuda")
                    elif hasattr(model, "model") and model.model is not None:
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

                track_kwargs = dict(
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
                if torch.cuda.is_available():
                    # Explicit override in case ultralytics' cached
                    # device selection is stale after a CPU offload.
                    track_kwargs["device"] = "cuda"

                results_iter = model.track(**track_kwargs)

                for r in results_iter:
                    if t >= B:
                        break

                    if r.masks is None or r.boxes is None:
                        _logger.info(
                            "  YOLO-seg frame %4d: NO masks/boxes (YOLO "
                            "found nothing for classes=%s)",
                            t, class_list,
                        )
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

                    # --- Per-frame tracker-output dump --------------------
                    # Logs every detection: YOLO class-conf + tracker ID
                    # (or 'NONE' if the tracker dropped it) + bbox + mask
                    # pixel count. The mask px is the SIGNAL we actually
                    # care about — a bbox can be huge because of one
                    # stray mask pixel in the corner; the pixel count
                    # tells us whether the mask is a real person-sized
                    # blob or a thin scattering of noise pixels.
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy() \
                        if r.boxes.xyxy is not None else None
                    confs = r.boxes.conf.cpu().numpy() \
                        if r.boxes.conf is not None else None
                    ids_cpu = ids.cpu().numpy().astype(int) \
                        if ids is not None else None

                    # Mask pixel count at YOLO's native mask resolution
                    # (pre-resize). masks_data is uint8 in {0, 1}.
                    mask_shape = tuple(masks_data.shape[-2:])  # (H', W')
                    per_det_px = masks_data.reshape(masks_data.shape[0], -1) \
                        .sum(dim=1).cpu().numpy().astype(int)
                    mask_total = int(mask_shape[0] * mask_shape[1])

                    det_lines = []
                    N_det = int(masks_data.shape[0])
                    for k in range(N_det):
                        conf = float(confs[k]) if confs is not None else -1.0
                        tid_str = (
                            f"tid={int(ids_cpu[k])}"
                            if ids_cpu is not None else "tid=NONE"
                        )
                        if boxes_xyxy is not None:
                            x1, y1, x2, y2 = boxes_xyxy[k]
                            bbox_str = (
                                f"bbox=({int(x1)},{int(y1)},"
                                f"{int(x2)},{int(y2)})"
                            )
                        else:
                            bbox_str = "bbox=(?)"
                        # Mask fill ratio = px / (H' * W'). Useful for
                        # spotting over-segmentation (e.g. >40% of frame
                        # is suspicious for a single person).
                        pct = (
                            100.0 * per_det_px[k] / mask_total
                            if mask_total > 0 else 0.0
                        )
                        det_lines.append(
                            f"{tid_str} conf={conf:.2f} {bbox_str} "
                            f"mask_px={int(per_det_px[k])} ({pct:.1f}%)"
                        )
                    _logger.info(
                        "  YOLO-seg frame %4d: %d det(s) | mask_shape=%s | %s",
                        t, N_det, mask_shape,
                        " | ".join(det_lines) if det_lines else "(none)",
                    )

                    # Tracker may skip IDs on noisy detections; assign
                    # one-shot placeholder IDs so those masks still
                    # appear in their own slot rather than being merged.
                    if ids is None:
                        _logger.warning(
                            "  YOLO-seg frame %4d: tracker returned "
                            "r.boxes.id=None for all %d detections — "
                            "injecting fallback IDs %d..%d. This creates "
                            "PHANTOM slots (one per detection, unmergeable "
                            "with real tracks) and is the #1 cause of "
                            "inflated n_persons when BoT-SORT hasn't yet "
                            "confirmed its tracks.",
                            t, N_det, fallback_id,
                            fallback_id + N_det - 1,
                        )
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

                    # Bit-pack masks: 1/32 the RAM of float32, 1/8 the
                    # RAM of plain bool. Long HD videos with many
                    # tracked persons easily blow past 16-32 GB of
                    # system RAM otherwise.
                    masks_np = (masks_data > 0.5).cpu().numpy()   # bool
                    ids_np = ids.cpu().numpy().astype(int)

                    for i, tid in enumerate(ids_np):
                        per_frame[t][int(tid)] = pack_mask(masks_np[i])
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
                if hasattr(model, "to"):
                    model.to("cpu")
                elif hasattr(model, "model") and model.model is not None:
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
            # Pass-through for the debug output so downstream graph
            # wiring doesn't need to know whether anything was detected.
            return (torch.zeros(B, H, W), images)

        # Map track IDs → contiguous slot indices (0, 1, 2, ...)
        sorted_ids = sorted(all_track_ids)
        id_to_slot = {tid: i for i, tid in enumerate(sorted_ids)}
        n_persons = len(sorted_ids)

        # Build aligned output — same layout as SAM3.
        # Allocate the final float32 tensor once and unpack each packed
        # mask directly into it — avoids a parallel intermediate bool
        # buffer. Peak RAM ≈ sizeof(masks_out).
        masks_out = torch.zeros(B * n_persons, H, W, dtype=torch.float32)
        for t in range(B):
            for tid, packed in per_frame[t].items():
                slot = id_to_slot[tid]
                mask_bool = unpack_mask(packed, H, W)
                masks_out[t * n_persons + slot] = torch.from_numpy(
                    mask_bool.astype(np.float32)
                )

        # Build the colored-mask overlay before dropping per_frame, so
        # we still have per-frame track lookups available. Returns the
        # original images unchanged when debug_overlay is off (keeps
        # the output signature consistent).
        overlay_out = self._build_debug_overlay(
            images=images,
            per_frame=per_frame,
            id_to_slot=id_to_slot,
            sorted_ids=sorted_ids,
            H=H, W=W,
        ) if debug_overlay else images

        del per_frame

        _logger.info(
            "YOLO segmentation: %d frames, classes=%s, %d tracked objects, "
            "output shape %s",
            B, class_list, n_persons, tuple(masks_out.shape),
        )

        return (masks_out, overlay_out)

    def _build_debug_overlay(self, images, per_frame, id_to_slot,
                             sorted_ids, H, W):
        """Render a color-coded mask overlay for visual inspection.

        Returns a ``(B, H, W, 3)`` float32 tensor in [0, 1] where each
        tracked person's mask is alpha-blended into the original image
        with a distinct color, plus a matching bbox outline and a
        ``tid=N slot=M`` text label. All work is on CPU / numpy — the
        overlay is a debug aid, not a hot path.
        """
        import cv2  # ultralytics ships cv2, so this is always available

        palette = self._DEBUG_PALETTE_RGB  # (P, 3) uint8 RGB
        alpha = 0.45
        B = int(images.shape[0])

        # Log the slot -> color legend once so reading the overlay is
        # unambiguous (the text label also prints slot, but eyeballing
        # color-to-person is what makes this fast to scan).
        color_names = [
            "red", "green", "blue", "yellow", "magenta", "cyan",
            "purple", "olive", "orange", "mint", "pink", "light-blue",
        ]
        legend_parts = []
        for slot, tid in enumerate(sorted_ids):
            c = color_names[slot % len(color_names)]
            legend_parts.append(f"slot{slot}(tid={tid})={c}")
        _logger.info(
            "YOLO-seg debug overlay legend: %s", " | ".join(legend_parts),
        )

        # Work in uint8 so cv2's drawing functions behave naturally.
        dbg = (images.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        dbg = np.ascontiguousarray(dbg)  # cv2 requires contiguous

        for t in range(B):
            frame = dbg[t]  # (H, W, 3) uint8, view (in-place edits OK)
            for tid, packed in per_frame[t].items():
                slot = id_to_slot[tid]
                color = palette[slot % len(palette)]            # (3,) uint8
                color_list = [int(c) for c in color.tolist()]   # for cv2

                mask_bool = unpack_mask(packed, H, W)           # (H, W) bool
                if not mask_bool.any():
                    continue

                # 1. Alpha-blend colored mask.
                # Vectorized blend only on mask pixels keeps things
                # cheap and avoids bleeding outside the silhouette.
                sel = mask_bool
                f = frame[sel].astype(np.float32)
                blended = f * (1.0 - alpha) + color.astype(np.float32) * alpha
                frame[sel] = np.clip(blended, 0, 255).astype(np.uint8)

                # 2. Bbox outline (from mask tight bounds).
                ys, xs = np.where(mask_bool)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color_list, thickness=2,
                )

                # 3. Text label: track ID + slot index.
                label = f"tid={tid} slot={slot}"
                ty = max(0, y1 - 6)
                cv2.putText(
                    frame, label, (x1 + 2, ty + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_list,
                    thickness=2, lineType=cv2.LINE_AA,
                )

        return torch.from_numpy(dbg.astype(np.float32) / 255.0)
