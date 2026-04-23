"""
Bit-packed mask utilities.

Segmentation masks are binary, so storing them as float32 (4 bytes per
pixel) or even numpy bool (1 byte per pixel) wastes memory.  For long
videos at HD resolution with multiple people this can push system RAM
past its limits and trigger OOM-kill.

`pack_mask` / `unpack_mask` use numpy's bit-packing (8 pixels per byte)
for 1/32 the float32 cost — enough headroom for 1000-frame HD videos
with several tracked persons on ~16 GB cloud instances.
"""

import numpy as np


def pack_mask(mask_bool):
    """Compress a (H, W) bool mask into bit-packed bytes.

    Returns a 1-D uint8 array.  Pair with ``unpack_mask(buf, H, W)``.
    """
    return np.packbits(np.ascontiguousarray(mask_bool).reshape(-1))


def unpack_mask(packed, h, w):
    """Decode bit-packed bytes back into a (H, W) bool mask."""
    total = h * w
    return np.unpackbits(packed, count=total).reshape(h, w).astype(bool)
