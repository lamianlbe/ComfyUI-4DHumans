"""
Wan Animate Face Preprocess node.

Ensures each frame has at most one visible person's face for Wan 2.2
Animate's face video input.  If multiple persons are visible, only the
visible one's face region is kept; the rest is blacked out.  If no
person is visible, the entire frame is blacked out.

Uses head keypoints (nose, eyes, ears, shoulders) from the POSES dict
to compute a face bounding box, then expands it to include surrounding
context.
"""

import numpy as np
import torch
import comfy.utils


# OpenPose 25 indices for head/upper-body landmarks
_NOSE = 0
_R_EYE = 15
_L_EYE = 16
_R_EAR = 17
_L_EAR = 18
_R_SHOULDER = 2
_L_SHOULDER = 5
_HEAD_INDICES = [_NOSE, _R_EYE, _L_EYE, _R_EAR, _L_EAR, _R_SHOULDER, _L_SHOULDER]


def _face_bbox_from_keypoints(kp2d, img_w, img_h, expand_ratio=1.8):
    """
    Compute an expanded face bounding box from OpenPose 25 keypoints.

    Parameters
    ----------
    kp2d : (25, 2) or (25, 3) ndarray
    img_w, img_h : int
    expand_ratio : float
        How much to expand the tight bbox around head keypoints.
        1.0 = tight, 2.0 = double the size in each direction.

    Returns
    -------
    (x1, y1, x2, y2) or None if no valid head keypoints
    """
    pts = []
    for idx in _HEAD_INDICES:
        x, y = float(kp2d[idx, 0]), float(kp2d[idx, 1])
        # Skip zero/invalid points
        if abs(x) < 1e-3 and abs(y) < 1e-3:
            continue
        if x < 0 or y < 0 or x > img_w or y > img_h:
            continue
        pts.append((x, y))

    if len(pts) < 2:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # Tight bbox
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    w = x2 - x1
    h = y2 - y1
    size = max(w, h)

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

        pbar = comfy.utils.ProgressBar(B)
        output = []

        for t in range(B):
            # Find visible persons in this frame
            visible_persons = []
            for p_idx in range(n_persons):
                person = poses["persons"][p_idx]
                if not person.get("visible", True):
                    continue
                j2d = person["body_joints2d"][t]
                if j2d is not None:
                    visible_persons.append((p_idx, j2d))

            if len(visible_persons) == 0:
                # No visible person → black frame
                output.append(torch.zeros_like(images[t]))

            elif len(visible_persons) == 1:
                # Exactly one visible person → pass through unchanged
                output.append(images[t].clone())

            else:
                # Multiple visible persons → keep only the first visible
                # person's face region, black out everything else
                _, j2d = visible_persons[0]
                bbox = _face_bbox_from_keypoints(
                    j2d, img_w, img_h, expand_ratio
                )

                if bbox is None:
                    # Can't locate face → black frame
                    output.append(torch.zeros_like(images[t]))
                else:
                    x1, y1, x2, y2 = bbox
                    frame = torch.zeros_like(images[t])
                    frame[y1:y2, x1:x2] = images[t][y1:y2, x1:x2]
                    output.append(frame)

            pbar.update(1)

        return (torch.stack(output),)
