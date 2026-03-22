"""
Wan Animate Face Preprocess node.

Ensures each frame has at most one visible person's face for Wan 2.2
Animate's face video input.

Rules:
  - 0 visible persons in a frame → entire frame blacked out
  - 1 visible person, no invisible persons → pass through unchanged
  - 1 visible person, with invisible persons → keep visible person's
    face region (expanded bbox), black out everything else
  - >1 visible person → error (must use Pose Editor to select one)

Uses head keypoints (nose, eyes, ears, shoulders) from the POSES dict
to compute a face bounding box, then expands it to include surrounding
context.
"""

import numpy as np
import torch
import comfy.utils


# OpenPose 25 indices for head/upper-body landmarks
_NOSE = 0
_NECK = 1
_R_SHOULDER = 2
_L_SHOULDER = 5
_R_EYE = 15
_L_EYE = 16
_R_EAR = 17
_L_EAR = 18
_HEAD_INDICES = [_NOSE, _R_EYE, _L_EYE, _R_EAR, _L_EAR]
# Fallback: use shoulders/neck to estimate head position
_UPPER_BODY_INDICES = [_NECK, _R_SHOULDER, _L_SHOULDER]


def _face_bbox_from_keypoints(kp2d, img_w, img_h, expand_ratio=2.0):
    """
    Compute an expanded face bounding box from OpenPose 25 keypoints.

    Uses head keypoints (nose, eyes, ears) primarily. If fewer than 2
    head points are available, falls back to shoulders/neck to estimate
    head position. If only 1 point is available, uses a fixed-size bbox
    based on image dimensions.

    Parameters
    ----------
    kp2d : (25, 2) or (25, 3) ndarray
    img_w, img_h : int
    expand_ratio : float
        How much to expand the tight bbox around head keypoints.

    Returns
    -------
    (x1, y1, x2, y2) or None if absolutely no keypoints detected
    """
    def _valid_pt(idx):
        x, y = float(kp2d[idx, 0]), float(kp2d[idx, 1])
        if abs(x) < 1e-3 and abs(y) < 1e-3:
            return None
        if x < 0 or y < 0 or x > img_w or y > img_h:
            return None
        return (x, y)

    # Collect head points
    head_pts = []
    for idx in _HEAD_INDICES:
        pt = _valid_pt(idx)
        if pt is not None:
            head_pts.append(pt)

    if len(head_pts) >= 2:
        # Good case: multiple head points available
        xs = [p[0] for p in head_pts]
        ys = [p[1] for p in head_pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        size = max(max(xs) - min(xs), max(ys) - min(ys))
        # Ensure minimum size (at least 5% of image)
        size = max(size, min(img_w, img_h) * 0.05)
    elif len(head_pts) == 1:
        # Only one head point: use it as center with estimated size
        cx, cy = head_pts[0]
        # Estimate head size as ~15% of image height
        size = img_h * 0.15
    else:
        # No head points: try shoulders to estimate head position
        upper_pts = []
        for idx in _UPPER_BODY_INDICES:
            pt = _valid_pt(idx)
            if pt is not None:
                upper_pts.append(pt)

        if len(upper_pts) == 0:
            return None

        # Head is roughly above the midpoint of shoulders
        xs = [p[0] for p in upper_pts]
        ys = [p[1] for p in upper_pts]
        cx = sum(xs) / len(xs)
        # Head center is above shoulders by roughly shoulder width
        shoulder_span = max(xs) - min(xs) if len(xs) > 1 else img_w * 0.15
        cy = min(ys) - shoulder_span * 0.5
        cy = max(0, cy)
        size = shoulder_span

    # Expand around center
    half = size * expand_ratio / 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(img_w, cx + half))
    y2 = int(min(img_h, cy + half))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


class WanAnimateFacePreprocessNode:
    """Preprocess video for Wan 2.2 Animate face input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "poses": ("POSES",),
                "expand_ratio": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": (
                            "How much to expand the face region. "
                            "1.0 = tight crop around head keypoints, "
                            "2.0 = double the size for more context."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preprocess"
    CATEGORY = "4dhumans"

    def preprocess(self, images, poses, expand_ratio):
        B, img_h, img_w, C = images.shape
        n_persons = poses["n_persons"]
        n_frames = poses["n_frames"]

        if B != n_frames:
            raise ValueError(
                f"Wan Animate Face Preprocess: image batch size ({B}) "
                f"does not match pose frame count ({n_frames})."
            )

        # --- Pre-check: no frame should have >1 visible person ---
        for t in range(B):
            n_visible = 0
            for p_idx in range(n_persons):
                person = poses["persons"][p_idx]
                if person.get("visible", True):
                    j2d = person["body_joints2d"][t]
                    if j2d is not None:
                        n_visible += 1
            if n_visible > 1:
                raise ValueError(
                    f"Wan Animate Face Preprocess: frame {t} has {n_visible} "
                    f"visible persons. Use Pose Editor to set all but one "
                    f"person to invisible before using this node."
                )

        # --- Count total persons (visible + invisible) with detections ---
        pbar = comfy.utils.ProgressBar(B)
        output = []

        for t in range(B):
            # Find visible and total detected persons
            visible_person = None  # (p_idx, j2d)
            has_invisible = False

            for p_idx in range(n_persons):
                person = poses["persons"][p_idx]
                j2d = person["body_joints2d"][t]
                if j2d is None:
                    continue

                if person.get("visible", True):
                    visible_person = (p_idx, j2d)
                else:
                    has_invisible = True

            if visible_person is None:
                # No visible person → black frame
                output.append(torch.zeros_like(images[t]))

            elif not has_invisible:
                # One visible, no invisible → pass through unchanged
                output.append(images[t].clone())

            else:
                # One visible + invisible persons → mask to face region
                _, j2d = visible_person
                bbox = _face_bbox_from_keypoints(
                    j2d, img_w, img_h, expand_ratio
                )

                if bbox is None:
                    # Can't locate face → black frame as safety
                    output.append(torch.zeros_like(images[t]))
                else:
                    x1, y1, x2, y2 = bbox
                    frame = torch.zeros_like(images[t])
                    frame[y1:y2, x1:x2] = images[t][y1:y2, x1:x2]
                    output.append(frame)

            pbar.update(1)

        return (torch.stack(output),)
