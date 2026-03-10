"""
Render SCAIL-style pose images from SMPLest-X 137-joint keypoints.

Visual style matches WAN-SCAIL ControlNet expectations:
  - Body limbs: thick ellipse-fill strokes with SCAIL color scheme
    (warm=right side, cool=left side), drawn at 60% opacity
  - Hands: DWPose-style HSV rainbow thin lines + red dots
  - Face: white dots (optimized subset: eyebrows + eyes/nose/mouth)
"""
import math
import cv2
import numpy as np

# ── SMPLest-X 25-body → SCAIL limb mapping ──────────────────────────────────
# Our body joints: 0=Pelvis 1=L.Hip 2=R.Hip 3=L.Knee 4=R.Knee 5=L.Ankle
# 6=R.Ankle 7=Neck 8=L.Shoulder 9=R.Shoulder 10=L.Elbow 11=R.Elbow
# 12=L.Wrist 13=R.Wrist 14-16=L.Foot 17-19=R.Foot 20=L.Ear 21=R.Ear
# 22=L.Eye 23=R.Eye 24=Nose

SCAIL_LIMBS = [
    (7, 9),    # 0  Neck → R.Shoulder
    (7, 8),    # 1  Neck → L.Shoulder
    (9, 11),   # 2  R.Shoulder → R.Elbow
    (11, 13),  # 3  R.Elbow → R.Wrist
    (8, 10),   # 4  L.Shoulder → L.Elbow
    (10, 12),  # 5  L.Elbow → L.Wrist
    (7, 2),    # 6  Neck → R.Hip
    (2, 4),    # 7  R.Hip → R.Knee
    (4, 6),    # 8  R.Knee → R.Ankle
    (7, 1),    # 9  Neck → L.Hip
    (1, 3),    # 10 L.Hip → L.Knee
    (3, 5),    # 11 L.Knee → L.Ankle
    (7, 24),   # 12 Neck → Nose
    (24, 23),  # 13 Nose → R.Eye
    (23, 21),  # 14 R.Eye → R.Ear
    (24, 22),  # 15 Nose → L.Eye
    (22, 20),  # 16 L.Eye → L.Ear
]

# SCAIL color scheme (RGB): warm=right, cool=left
SCAIL_LIMB_COLORS = [
    (255, 0, 0),       # 0  Red
    (0, 255, 255),     # 1  Cyan
    (255, 85, 0),      # 2  Orange
    (255, 170, 0),     # 3  Golden Orange
    (0, 170, 255),     # 4  Sky Blue
    (0, 85, 255),      # 5  Medium Blue
    (180, 255, 0),     # 6  Yellow-Green
    (0, 255, 0),       # 7  Bright Green
    (0, 255, 85),      # 8  Light Green-Blue
    (0, 0, 255),       # 9  Pure Blue
    (85, 0, 255),      # 10 Purple-Blue
    (170, 0, 255),     # 11 Medium Purple
    (150, 150, 150),   # 12 Grey
    (255, 0, 170),     # 13 Pink-Magenta
    (50, 0, 255),      # 14 Dark Violet
    (255, 0, 170),     # 15 Pink-Magenta
    (50, 0, 255),      # 16 Dark Violet
]

# Hand edges (wrist → 4 joints per finger × 5 fingers = 20 edges)
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]

# SMPLest-X face keypoints that correspond to DWPose "optimized" subset
# DWPose 68-face: 17-27 = eyebrows, 36-47 = eyes, 48-67 = nose+mouth
# SMPLest-X 72-face (indices 65-136 in 137 layout) has a similar ordering
# We draw all 72 face points as white dots (small radius)

def _hsv_to_rgb(h, s, v):
    """Convert a single HSV value to RGB tuple (0-255)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_scail_pose(img, keypoints, threshold=0.1):
    """
    Render SCAIL-style pose on image from 137-joint SMPLest-X keypoints.

    Args:
        img: (H, W, 3) uint8 canvas (typically black).
        keypoints: (137, 3) array with (x, y, confidence) per joint.
        threshold: Minimum confidence to draw.

    Returns:
        (H, W, 3) uint8 image with SCAIL-style skeleton.
    """
    img = np.ascontiguousarray(img.copy())
    h, w = img.shape[:2]
    stickwidth = max(2, int(min(h, w) / 200))

    def _valid(idx):
        return keypoints[idx, 2] > threshold

    def _pt(idx):
        return (int(round(keypoints[idx, 0])), int(round(keypoints[idx, 1])))

    def _xy(idx):
        return keypoints[idx, 0], keypoints[idx, 1]

    # ── Body limbs (ellipse-fill, SCAIL colors, 60% opacity) ────────────
    body_layer = np.zeros_like(img)
    for i, (a, b) in enumerate(SCAIL_LIMBS):
        if not (_valid(a) and _valid(b)):
            continue
        x1, y1 = _xy(a)
        x2, y2 = _xy(b)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if length < 1:
            continue
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        polygon = cv2.ellipse2Poly(
            (int(mx), int(my)),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1,
        )
        cv2.fillConvexPoly(body_layer, polygon, SCAIL_LIMB_COLORS[i])

    # Apply 60% opacity (matches DWPose body style)
    body_layer = (body_layer * 0.6).astype(np.uint8)

    # Body joint dots
    body_dot_indices = [24, 7, 9, 11, 13, 8, 10, 12, 2, 4, 6, 1, 3, 5, 23, 22, 21, 20]
    for idx in body_dot_indices:
        if _valid(idx):
            cv2.circle(body_layer, _pt(idx), max(2, stickwidth - 1),
                       SCAIL_LIMB_COLORS[min(body_dot_indices.index(idx),
                                             len(SCAIL_LIMB_COLORS) - 1)],
                       thickness=-1)

    # Merge body layer onto canvas
    mask = body_layer > 0
    img[mask] = body_layer[mask]

    # ── Hands (DWPose HSV rainbow style) ────────────────────────────────
    hand_thick = max(1, int(min(h, w) / 300))
    hand_dot_rad = max(1, hand_thick)

    for hand_start, wrist_idx in [(25, 12), (45, 13)]:  # left, right
        # Build 21-joint array: [wrist, 20 finger joints]
        hand_pts = np.zeros((21, 3), dtype=np.float32)
        hand_pts[0] = keypoints[wrist_idx]
        hand_pts[1:] = keypoints[hand_start:hand_start + 20]

        for ie, (ea, eb) in enumerate(_HAND_EDGES):
            if hand_pts[ea, 2] > threshold and hand_pts[eb, 2] > threshold:
                p1 = (int(round(hand_pts[ea, 0])), int(round(hand_pts[ea, 1])))
                p2 = (int(round(hand_pts[eb, 0])), int(round(hand_pts[eb, 1])))
                color = _hsv_to_rgb(ie / len(_HAND_EDGES), 1.0, 1.0)
                cv2.line(img, p1, p2, color, hand_thick, cv2.LINE_AA)

        for j in range(21):
            if hand_pts[j, 2] > threshold:
                pt = (int(round(hand_pts[j, 0])), int(round(hand_pts[j, 1])))
                cv2.circle(img, pt, hand_dot_rad, (0, 0, 255), thickness=-1)

    # ── Face (white dots, small radius) ─────────────────────────────────
    face_rad = max(1, int(min(h, w) / 500))
    for j in range(65, min(137, keypoints.shape[0])):
        if _valid(j):
            cv2.circle(img, _pt(j), face_rad, (255, 255, 255), -1, cv2.LINE_AA)

    return img
