"""
Frame Rate Capper — drop frames at uniform stride so the effective
frame rate never exceeds a user-set maximum.

Use case: source videos at 30/60 fps fed into expensive per-frame
pose / 3D inference often don't NEED that many samples per second.
Capping to e.g. 16 fps roughly halves downstream cost with negligible
motion-fidelity loss for human pose. Keeps the workflow's "real fps"
metadata accurate so VHS / pose-saving / 3D-sampling stages all stay
in sync.

Algorithm:

    if original_fps <= max_fps:    → pass-through, no drop
    else:
        stride = ceil(original_fps / max_fps)
        keep frames [0, stride, 2*stride, ...]
        actual_fps = original_fps / stride

Using ceil guarantees ``actual_fps <= max_fps`` even when the ratio
isn't an integer (e.g. 30 / 12 → stride 3 → 10 fps ≤ 12). The output
fps the node reports is the TRUE achieved fps after dropping, not the
target — downstream nodes (BMPRTMWPose's fps input, NPZ saver, etc.)
should consume this value so timing stays accurate.

Frame-zero is always kept (so the first frame's timestamp = 0). Last
frame is included only when (n_frames - 1) % stride == 0; that's a
deliberate uniform-stride choice — uniform timing matters more than
"include the last frame" for downstream tracking / interpolation.
"""

import logging
import math


_logger = logging.getLogger(__name__)


class FrameRateCapperNode:
    """Drop frames so output fps ≤ max_fps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.1,
                        "tooltip": (
                            "Source video frame rate. Typically wired "
                            "from VHS_VideoInfoLoaded's frame_rate "
                            "output. Used as the dividend when "
                            "computing the drop stride."
                        ),
                    },
                ),
                "max_fps": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.1,
                        "tooltip": (
                            "Cap on the effective output frame rate. "
                            "When original_fps > max_fps, frames are "
                            "dropped at a uniform stride = "
                            "ceil(original_fps / max_fps), so the "
                            "actual output fps lands at "
                            "original_fps / stride (always ≤ max_fps). "
                            "Lower max_fps = faster downstream "
                            "inference. 16 is a reasonable default "
                            "for human-pose pipelines (matches "
                            "Pose3DUpgrade's typical pose_3d_fps)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "actual_fps")
    FUNCTION = "cap"
    CATEGORY = "4dhumans"

    def cap(self, images, original_fps, max_fps):
        # Defensive: invalid fps inputs pass through unchanged. Keeps
        # the node from crashing user workflows on edge inputs (e.g.
        # static-image batches where VHS reports fps=0).
        if original_fps <= 0.0 or max_fps <= 0.0:
            _logger.warning(
                "FrameRateCapper: invalid fps inputs "
                "(original=%.3f, max=%.3f) — passing %d frames through.",
                float(original_fps), float(max_fps),
                int(images.shape[0]),
            )
            return (images, float(original_fps))

        n_in = int(images.shape[0])

        # Already at or under the cap → no dropping.
        if original_fps <= max_fps:
            _logger.info(
                "FrameRateCapper: %d frames @ %.3f fps already ≤ "
                "max %.3f fps — pass-through.",
                n_in, float(original_fps), float(max_fps),
            )
            return (images, float(original_fps))

        # ceil so actual_fps is GUARANTEED ≤ max_fps, not just close.
        stride = max(1, math.ceil(float(original_fps) / float(max_fps)))
        # Tensor slicing — zero-copy view, O(1) memory.
        capped = images[::stride]
        actual_fps = float(original_fps) / float(stride)
        n_out = int(capped.shape[0])

        _logger.info(
            "FrameRateCapper: %d frames @ %.3f fps → %d frames @ "
            "%.3f fps (stride=%d, target ≤ %.3f fps).",
            n_in, float(original_fps),
            n_out, actual_fps,
            stride, float(max_fps),
        )

        return (capped, actual_fps)
