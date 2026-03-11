"""
Render whole-body OpenPose-style keypoints (137 joints) from SMPLest-X.

Joint layout (SMPLest-X 137 format):
  0-24:   Body (25 joints)
  25-44:  Left hand (20 joints)
  45-64:  Right hand (20 joints)
  65-136: Face (72 keypoints)
"""
import cv2
import numpy as np

# ── Skeleton definitions ──────────────────────────────────────────────────────

BODY_PAIRS = [
    (0, 1), (0, 2),          # pelvis → hips
    (1, 3), (2, 4),          # hips → knees
    (3, 5), (4, 6),          # knees → ankles
    (0, 7),                   # pelvis → neck
    (7, 8), (7, 9),          # neck → shoulders
    (8, 10), (9, 11),        # shoulders → elbows
    (10, 12), (11, 13),      # elbows → wrists
    (5, 14), (5, 15), (5, 16),  # L ankle → toes/heel
    (6, 17), (6, 18), (6, 19),  # R ankle → toes/heel
    (7, 24),                  # neck → nose
    (24, 22), (24, 23),      # nose → eyes
    (22, 20), (23, 21),      # eyes → ears
]

LHAND_PAIRS = [
    (12, 25), (25, 26), (26, 27), (27, 28),   # thumb
    (12, 29), (29, 30), (30, 31), (31, 32),   # index
    (12, 33), (33, 34), (34, 35), (35, 36),   # middle
    (12, 37), (37, 38), (38, 39), (39, 40),   # ring
    (12, 41), (41, 42), (42, 43), (43, 44),   # pinky
]

RHAND_PAIRS = [
    (13, 45), (45, 46), (46, 47), (47, 48),   # thumb
    (13, 49), (49, 50), (50, 51), (51, 52),   # index
    (13, 53), (53, 54), (54, 55), (55, 56),   # middle
    (13, 57), (57, 58), (58, 59), (59, 60),   # ring
    (13, 61), (61, 62), (62, 63), (63, 64),   # pinky
]

# ── Colors ────────────────────────────────────────────────────────────────────
# Body: OpenPose-style rainbow per limb (BGR)
BODY_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 170),
    (170, 0, 255), (255, 0, 255), (85, 0, 255), (0, 0, 255),
    (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255),
    (0, 255, 255),
]

# Hand finger colors (gradient per finger)
_FINGER_COLORS_L = [
    (0, 0, 255), (0, 60, 255), (0, 120, 255), (0, 180, 255),   # thumb (red)
    (0, 0, 230), (0, 50, 230), (0, 100, 230), (0, 150, 230),   # index
    (0, 0, 200), (0, 50, 200), (0, 100, 200), (0, 150, 200),   # middle
    (0, 0, 170), (0, 50, 170), (0, 100, 170), (0, 150, 170),   # ring
    (0, 0, 140), (0, 50, 140), (0, 100, 140), (0, 150, 140),   # pinky
]

_FINGER_COLORS_R = [
    (255, 0, 0), (255, 60, 0), (255, 120, 0), (255, 180, 0),   # thumb (blue)
    (230, 0, 0), (230, 50, 0), (230, 100, 0), (230, 150, 0),   # index
    (200, 0, 0), (200, 50, 0), (200, 100, 0), (200, 150, 0),   # middle
    (170, 0, 0), (170, 50, 0), (170, 100, 0), (170, 150, 0),   # ring
    (140, 0, 0), (140, 50, 0), (140, 100, 0), (140, 150, 0),   # pinky
]

FACE_COLOR = (255, 255, 255)  # white dots


def render_wholebody_openpose(img, keypoints, threshold=0.1):
    """
    Render 137-joint whole-body skeleton on image.

    Args:
        img: (H, W, 3) uint8 canvas.
        keypoints: (137, 3) array with (x, y, confidence) per joint.
        threshold: Minimum confidence to draw a joint/limb.

    Returns:
        (H, W, 3) uint8 image with skeleton drawn.
    """
    img = np.ascontiguousarray(img.copy())
    h, w = img.shape[:2]
    area = h * w
    line_thick = max(1, int(round(area ** 0.5 / 200)))
    circle_rad = max(1, line_thick + 1)
    hand_thick = max(1, line_thick // 2)
    hand_rad   = max(1, circle_rad // 2)

    def _valid(idx):
        return keypoints[idx, 2] > threshold

    def _pt(idx):
        return (int(round(keypoints[idx, 0])), int(round(keypoints[idx, 1])))

    # ── Body limbs ────────────────────────────────────────────────────────
    for i, (a, b) in enumerate(BODY_PAIRS):
        if _valid(a) and _valid(b):
            color = BODY_COLORS[b % len(BODY_COLORS)]
            cv2.line(img, _pt(a), _pt(b), color, line_thick, cv2.LINE_AA)

    # Body joint circles
    for j in range(25):
        if _valid(j):
            color = BODY_COLORS[j % len(BODY_COLORS)]
            cv2.circle(img, _pt(j), circle_rad, color, -1, cv2.LINE_AA)

    # ── Left hand ─────────────────────────────────────────────────────────
    for i, (a, b) in enumerate(LHAND_PAIRS):
        if _valid(a) and _valid(b):
            color = _FINGER_COLORS_L[i % len(_FINGER_COLORS_L)]
            cv2.line(img, _pt(a), _pt(b), color, hand_thick, cv2.LINE_AA)

    for j in range(25, 45):
        if _valid(j):
            cv2.circle(img, _pt(j), hand_rad, (0, 0, 255), -1, cv2.LINE_AA)

    # ── Right hand ────────────────────────────────────────────────────────
    for i, (a, b) in enumerate(RHAND_PAIRS):
        if _valid(a) and _valid(b):
            color = _FINGER_COLORS_R[i % len(_FINGER_COLORS_R)]
            cv2.line(img, _pt(a), _pt(b), color, hand_thick, cv2.LINE_AA)

    for j in range(45, 65):
        if _valid(j):
            cv2.circle(img, _pt(j), hand_rad, (255, 0, 0), -1, cv2.LINE_AA)

    # ── Face ──────────────────────────────────────────────────────────────
    face_rad = max(1, hand_rad)
    for j in range(65, min(137, keypoints.shape[0])):
        if _valid(j):
            cv2.circle(img, _pt(j), face_rad, FACE_COLOR, -1, cv2.LINE_AA)

    return img
