"""
Renderer for Sapiens Goliath 308-keypoint format.

Goliath layout (after teeth removal):
  0-14   : body (nose, eyes, ears, shoulders, elbows, hips, knees, ankles)
  15-20  : feet (L/R big_toe, small_toe, heel)
  21-41  : right hand (20 finger joints + wrist at 41)
  42-62  : left hand  (20 finger joints + wrist at 62)
  63-69  : extra body (olecranon, cubital_fossa, acromion, neck)
  70-307 : face (238 dense landmarks – drawn as dots, no lines)
"""

import cv2
import numpy as np

CONF_THRESHOLD = 0.3

# Face keypoint range (after teeth removal)
_FACE_START = 70
_FACE_END = 308  # exclusive

# ---------------------------------------------------------------------------
# Skeleton: (idx_a, idx_b, color)
# ---------------------------------------------------------------------------

# Body links (wrists live inside hand arrays: R=41, L=62)
_BODY_SKELETON = [
    # legs
    (13, 11, (0, 255, 0)),      # L ankle → L knee
    (11, 9,  (0, 255, 0)),      # L knee → L hip
    (14, 12, (255, 128, 0)),    # R ankle → R knee
    (12, 10, (255, 128, 0)),    # R knee → R hip
    # torso
    (9,  10, (51, 153, 255)),   # L hip → R hip
    (5,  9,  (51, 153, 255)),   # L shoulder → L hip
    (6,  10, (51, 153, 255)),   # R shoulder → R hip
    (5,  6,  (51, 153, 255)),   # L shoulder → R shoulder
    # arms
    (5,  7,  (0, 255, 0)),      # L shoulder → L elbow
    (6,  8,  (255, 128, 0)),    # R shoulder → R elbow
    (7,  62, (0, 255, 0)),      # L elbow → L wrist
    (8,  41, (255, 128, 0)),    # R elbow → R wrist
    # head
    (1,  2,  (51, 153, 255)),   # L eye → R eye
    (0,  1,  (51, 153, 255)),   # nose → L eye
    (0,  2,  (51, 153, 255)),   # nose → R eye
    (1,  3,  (51, 153, 255)),   # L eye → L ear
    (2,  4,  (51, 153, 255)),   # R eye → R ear
    (3,  5,  (51, 153, 255)),   # L ear → L shoulder
    (4,  6,  (51, 153, 255)),   # R ear → R shoulder
]

# Feet links
_FEET_SKELETON = [
    (13, 15, (0, 255, 0)),      # L ankle → L big toe
    (13, 16, (0, 255, 0)),      # L ankle → L small toe
    (13, 17, (0, 255, 0)),      # L ankle → L heel
    (14, 18, (255, 128, 0)),    # R ankle → R big toe
    (14, 19, (255, 128, 0)),    # R ankle → R small toe
    (14, 20, (255, 128, 0)),    # R ankle → R heel
]

# Right hand: wrist=41, fingers tip→base order in groups of 4
_RHAND_SKELETON = []
_rhand_wrist = 41
for _finger_base in [24, 28, 32, 36, 40]:  # thumb, index, middle, ring, pinky
    _color = {24: (255, 128, 0), 28: (255, 153, 255), 32: (102, 178, 255),
              36: (255, 51, 51), 40: (0, 255, 0)}[_finger_base]
    _RHAND_SKELETON.append((_rhand_wrist, _finger_base, _color))
    _RHAND_SKELETON.append((_finger_base, _finger_base - 1, _color))
    _RHAND_SKELETON.append((_finger_base - 1, _finger_base - 2, _color))
    _RHAND_SKELETON.append((_finger_base - 2, _finger_base - 3, _color))

# Left hand: wrist=62, fingers tip→base order in groups of 4
_LHAND_SKELETON = []
_lhand_wrist = 62
for _finger_base in [45, 49, 53, 57, 61]:  # thumb, index, middle, ring, pinky
    _color = {45: (255, 128, 0), 49: (255, 153, 255), 53: (102, 178, 255),
              57: (255, 51, 51), 61: (0, 255, 0)}[_finger_base]
    _LHAND_SKELETON.append((_lhand_wrist, _finger_base, _color))
    _LHAND_SKELETON.append((_finger_base, _finger_base - 1, _color))
    _LHAND_SKELETON.append((_finger_base - 1, _finger_base - 2, _color))
    _LHAND_SKELETON.append((_finger_base - 2, _finger_base - 3, _color))

_ALL_SKELETON = _BODY_SKELETON + _FEET_SKELETON + _RHAND_SKELETON + _LHAND_SKELETON


def render_goliath(canvas, kp, img_h, img_w, show_face=True,
                   conf_thr=CONF_THRESHOLD):
    """
    Render Goliath 308-keypoint skeleton on *canvas*.

    Parameters
    ----------
    canvas : ndarray (H, W, 3) uint8
    kp : ndarray (308, 3) pixel coords (x, y, confidence)
    img_h, img_w : image dimensions
    show_face : bool – whether to draw face landmark dots
    conf_thr : float – minimum confidence to draw

    Returns
    -------
    canvas : ndarray (H, W, 3) uint8
    """
    num_kp = kp.shape[0]
    line_thick = max(int(min(img_h, img_w) / 300), 2)
    dot_radius = max(int(min(img_h, img_w) / 200), 2)
    face_radius = max(int(min(img_h, img_w) / 400), 1)

    # Draw skeleton lines
    for a, b, color in _ALL_SKELETON:
        if a >= num_kp or b >= num_kp:
            continue
        xa, ya, ca = kp[a]
        xb, yb, cb = kp[b]
        if ca < conf_thr or cb < conf_thr:
            continue
        cv2.line(canvas,
                 (int(xa), int(ya)), (int(xb), int(yb)),
                 color, line_thick, cv2.LINE_AA)

    # Draw body/hand/feet keypoint dots
    for i in range(_FACE_START):
        if i >= num_kp:
            break
        x, y, c = kp[i]
        if c < conf_thr:
            continue
        cv2.circle(canvas, (int(x), int(y)), dot_radius,
                   (255, 255, 255), -1, cv2.LINE_AA)

    # Draw face landmark dots (no lines)
    if show_face:
        for i in range(_FACE_START, min(_FACE_END, num_kp)):
            x, y, c = kp[i]
            if c < conf_thr:
                continue
            cv2.circle(canvas, (int(x), int(y)), face_radius,
                       (255, 255, 255), -1, cv2.LINE_AA)

    return canvas
