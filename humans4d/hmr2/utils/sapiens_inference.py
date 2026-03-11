"""
Sapiens COCO-WholeBody pose estimation: inference + conversion to DWPose format.

Uses the COCO-WholeBody 133-keypoint model:
  body 0-16, feet 17-22, face 23-90 (iBUG 68),
  left hand 91-111, right hand 112-132.

Converts to DWPose-compatible dict format so that the existing
``draw_pose()`` renderer (from ``scail/draw_pose_utils.py``) can be reused.
"""

import logging

import numpy as np
import torch

_logger = logging.getLogger(__name__)

# Minimum confidence to consider a keypoint valid
CONF_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_sapiens_on_bbox(img_np, bbox, sapiens_dict):
    """
    Run Sapiens pose estimation on a single person crop.

    Parameters
    ----------
    img_np : ndarray (H, W, 3) uint8 RGB
    bbox : array-like [x1, y1, x2, y2]
    sapiens_dict : dict from LoadSapiensNode

    Returns
    -------
    dict with:
        "pixel_kp" : ndarray (N, 3)  image-space (x, y, conf)
    or None on failure.
    """
    model = sapiens_dict["model"]
    preprocessor = sapiens_dict["preprocessor"]
    device = sapiens_dict["device"]
    dtype = sapiens_dict["dtype"]

    img_h, img_w = img_np.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    cropped = img_np[y1:y2, x1:x2]
    crop_h, crop_w = cropped.shape[:2]

    tensor = preprocessor(cropped).unsqueeze(0).to(device).to(dtype)
    heatmaps = model(tensor).to(torch.float32)  # (1, K, hm_h, hm_w)
    heatmaps = heatmaps[0].cpu().numpy()         # (K, hm_h, hm_w)

    num_kp, hm_h, hm_w = heatmaps.shape
    pixel_kp = np.zeros((num_kp, 3), dtype=np.float32)
    for i in range(num_kp):
        hm = heatmaps[i]
        y_hm, x_hm = np.unravel_index(np.argmax(hm), hm.shape)
        pixel_kp[i] = (
            float(x_hm) * crop_w / hm_w + x1,
            float(y_hm) * crop_h / hm_h + y1,
            float(hm[y_hm, x_hm]),
        )

    return {"pixel_kp": pixel_kp}


# ---------------------------------------------------------------------------
# COCO-WholeBody 133 → DWPose conversion
# ---------------------------------------------------------------------------

# COCO-17 body → DWPose-18 body (neck is synthesized)
_COCO17_TO_DW18 = [
    0,    # DW 0  = COCO 0  nose
    None, # DW 1  = neck (synthesized from shoulders)
    6,    # DW 2  = COCO 6  R_shoulder
    8,    # DW 3  = COCO 8  R_elbow
    10,   # DW 4  = COCO 10 R_wrist
    5,    # DW 5  = COCO 5  L_shoulder
    7,    # DW 6  = COCO 7  L_elbow
    9,    # DW 7  = COCO 9  L_wrist
    12,   # DW 8  = COCO 12 R_hip
    14,   # DW 9  = COCO 14 R_knee
    16,   # DW 10 = COCO 16 R_ankle
    11,   # DW 11 = COCO 11 L_hip
    13,   # DW 12 = COCO 13 L_knee
    15,   # DW 13 = COCO 15 L_ankle
    2,    # DW 14 = COCO 2  R_eye
    1,    # DW 15 = COCO 1  L_eye
    4,    # DW 16 = COCO 4  R_ear
    3,    # DW 17 = COCO 3  L_ear
]


def coco_wb_to_dwpose(pixel_kp, img_h, img_w, conf_thr=CONF_THRESHOLD):
    """
    Convert COCO-WholeBody 133-keypoint pixel coords to DWPose dict format.

    Parameters
    ----------
    pixel_kp : ndarray (133, 3) with (x, y, confidence) in image pixel coords
    img_h, img_w : output image dimensions
    conf_thr : minimum confidence threshold

    Returns
    -------
    dict compatible with ``draw_pose()``:
        bodies:  {candidate: [[18×2]], subset: [[18 indices]]}
        faces:   [[68×2]]
        hands:   [[21×2], [21×2]]  (left, right)
    All coordinates are normalised to [0, 1].
    """
    def _get(idx):
        if idx >= pixel_kp.shape[0]:
            return -1.0, -1.0, 0.0
        x, y, c = pixel_kp[idx]
        if c < conf_thr:
            return -1.0, -1.0, 0.0
        return x / img_w, y / img_h, float(c)

    # --- Body (DWPose 18) ---
    candidate = []
    subset_row = []
    for dw_idx, coco_idx in enumerate(_COCO17_TO_DW18):
        if coco_idx is None:
            # Synthesize neck as midpoint of L_shoulder (5) and R_shoulder (6)
            lx, ly, lc = _get(5)
            rx, ry, rc = _get(6)
            if lc >= conf_thr and rc >= conf_thr:
                nx, ny = (lx + rx) / 2, (ly + ry) / 2
                candidate.append([nx, ny])
                subset_row.append(dw_idx)
            else:
                candidate.append([-1.0, -1.0])
                subset_row.append(-1)
        else:
            nx, ny, c = _get(coco_idx)
            candidate.append([nx, ny])
            subset_row.append(dw_idx if c >= conf_thr else -1)

    # --- Face (iBUG 68) — COCO-WB indices 23-90, already in iBUG order ---
    face = []
    for i in range(68):
        nx, ny, c = _get(23 + i)
        face.append([nx, ny])

    # --- Hands (21 joints each) — already in standard order ---
    left_hand = []
    for i in range(21):
        nx, ny, c = _get(91 + i)
        left_hand.append([nx, ny])

    right_hand = []
    for i in range(21):
        nx, ny, c = _get(112 + i)
        right_hand.append([nx, ny])

    return {
        "bodies": {
            "candidate": [candidate],
            "subset": [subset_row],
        },
        "faces": [face],
        "hands": [left_hand, right_hand],
    }


# ---------------------------------------------------------------------------
# Sapiens 2D + SMPLest-X 3D fusion
#
# When a body joint is missing from Sapiens (low confidence = clipped at
# screen edge), fill it in from SMPLest-X's 3D→2D projection.
# ---------------------------------------------------------------------------

# COCO-WB body index → SMPLest-X 137-joint index
_COCO_WB_BODY_TO_SMPLESTX = {
    0: 24,   # nose
    1: 22,   # L_eye
    2: 23,   # R_eye
    3: 20,   # L_ear
    4: 21,   # R_ear
    5: 8,    # L_shoulder
    6: 9,    # R_shoulder
    7: 10,   # L_elbow
    8: 11,   # R_elbow
    9: 12,   # L_wrist
    10: 13,  # R_wrist
    11: 1,   # L_hip
    12: 2,   # R_hip
    13: 3,   # L_knee
    14: 4,   # R_knee
    15: 5,   # L_ankle
    16: 6,   # R_ankle
}

# COCO-WB left hand (91-111) → SMPLest-X: wrist(12) + hand(25-44)
# COCO-WB right hand (112-132) → SMPLest-X: wrist(13) + hand(45-64)
_COCO_WB_LHAND_TO_SMPLESTX = [12] + list(range(25, 45))  # 21 joints
_COCO_WB_RHAND_TO_SMPLESTX = [13] + list(range(45, 65))  # 21 joints


def fuse_sapiens_smplestx(sapiens_kp, sx_kp2d, img_h, img_w,
                          conf_thr=CONF_THRESHOLD, edge_margin=0.02,
                          outlier_fraction=0.15, frame_idx=None):
    """
    Merge Sapiens 2D keypoints with SMPLest-X 2D projections.

    A Sapiens keypoint is considered *unreliable* when:
    - its confidence is below *conf_thr*, OR
    - it falls within *edge_margin* (fraction) of the image boundary
      (Sapiens heatmaps clamp off-screen joints to the edge with high conf).

    Additionally, a Sapiens keypoint is considered an *outlier* when its
    distance from the corresponding SMPLest-X prediction exceeds
    *outlier_fraction* of the image diagonal.

    Unreliable/outlier body/hand joints are replaced with SMPLest-X's
    3D→2D projection.

    Parameters
    ----------
    sapiens_kp : ndarray (133, 3) – Sapiens pixel coords (x, y, confidence)
    sx_kp2d    : ndarray (137, 3) – SMPLest-X pixel coords (x, y, confidence)
    img_h, img_w : image dimensions
    conf_thr   : float – confidence threshold
    edge_margin : float – fraction of image size to consider as edge zone
    outlier_fraction : float – max allowed distance as fraction of image diagonal
    frame_idx  : int or None – frame number for logging

    Returns
    -------
    merged : ndarray (133, 3) – merged keypoints in COCO-WB format.
    substituted : set – COCO-WB indices that were replaced.
    """
    merged = sapiens_kp.copy()

    # Edge zone: keypoints within this margin of image boundary are suspect
    mx = img_w * edge_margin
    my = img_h * edge_margin

    # Outlier distance threshold (pixels)
    diag = np.sqrt(img_h ** 2 + img_w ** 2)
    outlier_dist = diag * outlier_fraction

    def _is_unreliable(idx):
        x, y, c = merged[idx]
        if c < conf_thr:
            return True
        # Near image boundary — likely clamped from off-screen
        if x <= mx or x >= img_w - mx or y <= my or y >= img_h - my:
            return True
        return False

    def _is_outlier(coco_idx, sx_idx):
        """Check if Sapiens keypoint is too far from SMPLest-X prediction."""
        sx, sy, sc = sx_kp2d[sx_idx]
        if sc <= 0:
            return False  # no SMPLest-X reference to compare against
        sap_x, sap_y, sap_c = sapiens_kp[coco_idx]
        if sap_c < conf_thr:
            return False  # already unreliable, will be handled by _substitute
        dist = np.sqrt((sap_x - sx) ** 2 + (sap_y - sy) ** 2)
        if dist > outlier_dist:
            frame_str = f"frame {frame_idx}" if frame_idx is not None else "frame ?"
            _logger.warning(
                "Sapiens outlier rejected: %s, kp=%d, conf=%.3f, "
                "sapiens=(%.1f, %.1f), smplestx=(%.1f, %.1f), dist=%.1f",
                frame_str, coco_idx, sap_c, sap_x, sap_y, sx, sy, dist)
            return True
        return False

    substituted = set()  # track which COCO-WB indices were substituted

    def _substitute(coco_idx, sx_idx):
        unreliable = _is_unreliable(coco_idx)
        outlier = (not unreliable) and _is_outlier(coco_idx, sx_idx)
        if not unreliable and not outlier:
            return
        sx, sy, sc = sx_kp2d[sx_idx]
        if sc <= 0:
            if outlier:
                # No SMPLest-X fallback — zero out the outlier keypoint
                merged[coco_idx, 2] = 0.0
            return
        merged[coco_idx, 0] = sx
        merged[coco_idx, 1] = sy
        merged[coco_idx, 2] = conf_thr  # just above threshold so it renders
        substituted.add(coco_idx)

    # Body joints (0-16)
    for coco_idx, sx_idx in _COCO_WB_BODY_TO_SMPLESTX.items():
        _substitute(coco_idx, sx_idx)

    # Left hand (COCO-WB 91-111)
    for i, sx_idx in enumerate(_COCO_WB_LHAND_TO_SMPLESTX):
        _substitute(91 + i, sx_idx)

    # Right hand (COCO-WB 112-132)
    for i, sx_idx in enumerate(_COCO_WB_RHAND_TO_SMPLESTX):
        _substitute(112 + i, sx_idx)

    # Face: skip fusion (Sapiens COCO-WB face is already high quality,
    # and SMPLest-X face uses FLAME ordering which needs complex mapping)

    return merged, substituted
