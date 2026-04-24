"""
Bit-packed mask utilities + debug overlay renderer.

Segmentation masks are binary, so storing them as float32 (4 bytes per
pixel) or even numpy bool (1 byte per pixel) wastes memory.  For long
videos at HD resolution with multiple people this can push system RAM
past its limits and trigger OOM-kill.

`pack_mask` / `unpack_mask` use numpy's bit-packing (8 pixels per byte)
for 1/32 the float32 cost — enough headroom for 1000-frame HD videos
with several tracked persons on ~16 GB cloud instances.

`build_debug_overlay` renders a color-coded alpha-blended overlay of
tracked masks onto the input frames — the same debug aid is used by
all segmentation nodes (YOLO-seg, SAM3 video, SAM3 image) so visual
output stays consistent no matter which backend produced the masks.
"""

import numpy as np
import torch


def pack_mask(mask_bool):
    """Compress a (H, W) bool mask into bit-packed bytes.

    Returns a 1-D uint8 array.  Pair with ``unpack_mask(buf, H, W)``.
    """
    return np.packbits(np.ascontiguousarray(mask_bool).reshape(-1))


def unpack_mask(packed, h, w):
    """Decode bit-packed bytes back into a (H, W) bool mask."""
    total = h * w
    return np.unpackbits(packed, count=total).reshape(h, w).astype(bool)


# Debug overlay palette (RGB, uint8). 12 distinct hues chosen for
# contrast against typical photographic content; beyond 12 we wrap, but
# the text tid=N label always disambiguates color collisions.
_DEBUG_PALETTE_RGB = np.array([
    [255,  60,  60],   # red
    [ 60, 200,  60],   # green
    [ 60, 120, 255],   # blue
    [255, 200,  40],   # yellow
    [255,  70, 200],   # magenta
    [ 60, 220, 220],   # cyan
    [220, 110, 255],   # purple
    [180, 180,  60],   # olive
    [255, 140,  40],   # orange
    [110, 255, 160],   # mint
    [255, 160, 200],   # pink
    [160, 200, 255],   # light-blue
], dtype=np.uint8)
_DEBUG_COLOR_NAMES = [
    "red", "green", "blue", "yellow", "magenta", "cyan",
    "purple", "olive", "orange", "mint", "pink", "light-blue",
]


def build_debug_overlay(images, per_frame_items, H, W, alpha=0.45):
    """Render a color-coded mask overlay onto input frames.

    Parameters
    ----------
    images : torch.Tensor
        ``(B, H, W, 3)`` float in [0, 1]. Only read; never modified.
    per_frame_items : list[list[tuple[int, np.ndarray]]]
        Length-B list; ``per_frame_items[t]`` is a list of
        ``(track_id, packed_mask_uint8)`` pairs for detections in
        frame t. Use an empty list for frames with no detections.
    H, W : int
        Target image height / width (match the images tensor).
    alpha : float
        Blend weight for the colored mask. 0.45 keeps enough of the
        underlying image visible while the mask is clearly visible.

    Returns
    -------
    overlay : torch.Tensor
        ``(B, H, W, 3)`` float32 in [0, 1]. Same dtype/range as a
        normal ComfyUI IMAGE — connect to PreviewImage / VHS.
    legend : str
        One-line ``slotX(tid=T)=colorName`` mapping so the log can
        explain which color means which tracked person.
    """
    import cv2  # ultralytics / opencv is always available here

    # Stable slot assignment: sort track IDs across all frames so the
    # legend is deterministic (first-seen id doesn't matter).
    all_tids = sorted({
        tid for ft in per_frame_items for tid, _ in ft
    })
    id_to_slot = {tid: i for i, tid in enumerate(all_tids)}

    palette = _DEBUG_PALETTE_RGB
    color_names = _DEBUG_COLOR_NAMES

    legend_parts = [
        f"slot{slot}(tid={tid})={color_names[slot % len(color_names)]}"
        for slot, tid in enumerate(all_tids)
    ]
    legend = " | ".join(legend_parts) if legend_parts else "(no tracks)"

    # Work in uint8 so cv2's drawing functions behave naturally, and
    # force contiguity (cv2 refuses views / strided arrays).
    dbg = (images.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    dbg = np.ascontiguousarray(dbg)
    B = int(dbg.shape[0])

    for t in range(B):
        frame = dbg[t]  # (H, W, 3) uint8 — in-place edits OK
        for tid, packed in per_frame_items[t]:
            slot = id_to_slot[tid]
            color = palette[slot % len(palette)]
            color_list = [int(c) for c in color.tolist()]

            mask_bool = unpack_mask(packed, H, W)
            if not mask_bool.any():
                continue

            # 1. Alpha-blend colored mask, only over mask pixels.
            sel = mask_bool
            f = frame[sel].astype(np.float32)
            blended = f * (1.0 - alpha) + color.astype(np.float32) * alpha
            frame[sel] = np.clip(blended, 0, 255).astype(np.uint8)

            # 2. Bbox outline from mask tight bounds.
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

    overlay = torch.from_numpy(dbg.astype(np.float32) / 255.0)
    return overlay, legend
