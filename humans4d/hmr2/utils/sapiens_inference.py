"""
Sapiens Goliath pose estimation: inference + conversion to DWPose format.

Goliath outputs 308 keypoints (body + hands + dense face).
We convert them to DWPose-compatible dict format so that the existing
``draw_pose()`` renderer (from ``scail/draw_pose_utils.py``) can be reused.
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Goliath 344-keypoint names (subset we actually use for mapping)
# Full list at: ComfyUI_Sapiens/Sapiens_Pytorch/pose_classes_and_palettes.py
# We only store the names needed for building index lookup.
# ---------------------------------------------------------------------------

# fmt: off
GOLIATH_KEYPOINTS = [
    "nose",                          # 0
    "left_eye", "right_eye",         # 1-2
    "left_ear", "right_ear",         # 3-4
    "left_shoulder", "right_shoulder",# 5-6
    "left_elbow", "right_elbow",     # 7-8
    "left_hip", "right_hip",         # 9-10
    "left_knee", "right_knee",       # 11-12
    "left_ankle", "right_ankle",     # 13-14
    "left_big_toe", "left_small_toe", "left_heel",   # 15-17
    "right_big_toe", "right_small_toe", "right_heel", # 18-20
    # Right hand (tip→base per finger, 20 joints)
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",       # 21-24
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint", # 25-28
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint", # 29-32
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint", # 33-36
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint", # 37-40
    "right_wrist",                   # 41
    # Left hand (tip→base per finger, 20 joints)
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",           # 42-45
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",     # 46-49
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint", # 50-53
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint", # 54-57
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint", # 58-61
    "left_wrist",                    # 62
    # Extra body
    "left_olecranon", "right_olecranon",         # 63-64
    "left_cubital_fossa", "right_cubital_fossa", # 65-66
    "left_acromion", "right_acromion",           # 67-68
    "neck",                                      # 69
    # Face
    "center_of_glabella",            # 70
    "center_of_nose_root",           # 71
    "tip_of_nose_bridge",            # 72
    "midpoint_1_of_nose_bridge",     # 73
    "midpoint_2_of_nose_bridge",     # 74
    "midpoint_3_of_nose_bridge",     # 75
    "center_of_labiomental_groove",  # 76
    "tip_of_chin",                   # 77
    # Right eyebrow
    "upper_startpoint_of_r_eyebrow", # 78
    "lower_startpoint_of_r_eyebrow", # 79
    "end_of_r_eyebrow",             # 80
    "upper_midpoint_1_of_r_eyebrow", # 81
    "lower_midpoint_1_of_r_eyebrow", # 82
    "upper_midpoint_2_of_r_eyebrow", # 83
    "upper_midpoint_3_of_r_eyebrow", # 84
    "lower_midpoint_2_of_r_eyebrow", # 85
    "lower_midpoint_3_of_r_eyebrow", # 86
    # Left eyebrow
    "upper_startpoint_of_l_eyebrow", # 87
    "lower_startpoint_of_l_eyebrow", # 88
    "end_of_l_eyebrow",             # 89
    "upper_midpoint_1_of_l_eyebrow", # 90
    "lower_midpoint_1_of_l_eyebrow", # 91
    "upper_midpoint_2_of_l_eyebrow", # 92
    "upper_midpoint_3_of_l_eyebrow", # 93
    "lower_midpoint_2_of_l_eyebrow", # 94
    "lower_midpoint_3_of_l_eyebrow", # 95
    # Left eye - upper lash
    "l_inner_end_of_upper_lash_line",       # 96
    "l_outer_end_of_upper_lash_line",       # 97
    "l_centerpoint_of_upper_lash_line",     # 98
    "l_midpoint_2_of_upper_lash_line",      # 99
    "l_midpoint_1_of_upper_lash_line",      # 100
    "l_midpoint_6_of_upper_lash_line",      # 101
    "l_midpoint_5_of_upper_lash_line",      # 102
    "l_midpoint_4_of_upper_lash_line",      # 103
    "l_midpoint_3_of_upper_lash_line",      # 104
    # Left eye - upper eyelid (skip for now)
    "l_outer_end_of_upper_eyelid_line",     # 105
    "l_midpoint_6_of_upper_eyelid_line",    # 106
    "l_midpoint_2_of_upper_eyelid_line",    # 107
    "l_midpoint_5_of_upper_eyelid_line",    # 108
    "l_centerpoint_of_upper_eyelid_line",   # 109
    "l_midpoint_4_of_upper_eyelid_line",    # 110
    "l_midpoint_1_of_upper_eyelid_line",    # 111
    "l_midpoint_3_of_upper_eyelid_line",    # 112
    # Left eye - upper crease
    "l_midpoint_6_of_upper_crease_line",    # 113
    "l_midpoint_2_of_upper_crease_line",    # 114
    "l_midpoint_5_of_upper_crease_line",    # 115
    "l_centerpoint_of_upper_crease_line",   # 116
    "l_midpoint_4_of_upper_crease_line",    # 117
    "l_midpoint_1_of_upper_crease_line",    # 118
    "l_midpoint_3_of_upper_crease_line",    # 119
    # Right eye - upper lash
    "r_inner_end_of_upper_lash_line",       # 120
    "r_outer_end_of_upper_lash_line",       # 121
    "r_centerpoint_of_upper_lash_line",     # 122
    "r_midpoint_1_of_upper_lash_line",      # 123
    "r_midpoint_2_of_upper_lash_line",      # 124
    "r_midpoint_3_of_upper_lash_line",      # 125
    "r_midpoint_4_of_upper_lash_line",      # 126
    "r_midpoint_5_of_upper_lash_line",      # 127
    "r_midpoint_6_of_upper_lash_line",      # 128
    # Right eye - upper eyelid
    "r_outer_end_of_upper_eyelid_line",     # 129
    "r_midpoint_3_of_upper_eyelid_line",    # 130
    "r_midpoint_1_of_upper_eyelid_line",    # 131
    "r_midpoint_4_of_upper_eyelid_line",    # 132
    "r_centerpoint_of_upper_eyelid_line",   # 133
    "r_midpoint_5_of_upper_eyelid_line",    # 134
    "r_midpoint_2_of_upper_eyelid_line",    # 135
    "r_midpoint_6_of_upper_eyelid_line",    # 136
    # Right eye - upper crease
    "r_midpoint_3_of_upper_crease_line",    # 137
    "r_midpoint_1_of_upper_crease_line",    # 138
    "r_midpoint_4_of_upper_crease_line",    # 139
    "r_centerpoint_of_upper_crease_line",   # 140
    "r_midpoint_5_of_upper_crease_line",    # 141
    "r_midpoint_2_of_upper_crease_line",    # 142
    "r_midpoint_6_of_upper_crease_line",    # 143
    # Left eye - lower lash
    "l_inner_end_of_lower_lash_line",       # 144
    "l_outer_end_of_lower_lash_line",       # 145
    "l_centerpoint_of_lower_lash_line",     # 146
    "l_midpoint_2_of_lower_lash_line",      # 147
    "l_midpoint_1_of_lower_lash_line",      # 148
    "l_midpoint_6_of_lower_lash_line",      # 149
    "l_midpoint_5_of_lower_lash_line",      # 150
    "l_midpoint_4_of_lower_lash_line",      # 151
    "l_midpoint_3_of_lower_lash_line",      # 152
    # Left eye - lower eyelid
    "l_outer_end_of_lower_eyelid_line",     # 153
    "l_midpoint_6_of_lower_eyelid_line",    # 154
    "l_midpoint_2_of_lower_eyelid_line",    # 155
    "l_midpoint_5_of_lower_eyelid_line",    # 156
    "l_centerpoint_of_lower_eyelid_line",   # 157
    "l_midpoint_4_of_lower_eyelid_line",    # 158
    "l_midpoint_1_of_lower_eyelid_line",    # 159
    "l_midpoint_3_of_lower_eyelid_line",    # 160
    # Right eye - lower lash
    "r_inner_end_of_lower_lash_line",       # 161
    "r_outer_end_of_lower_lash_line",       # 162
    "r_centerpoint_of_lower_lash_line",     # 163
    "r_midpoint_1_of_lower_lash_line",      # 164
    "r_midpoint_2_of_lower_lash_line",      # 165
    "r_midpoint_3_of_lower_lash_line",      # 166
    "r_midpoint_4_of_lower_lash_line",      # 167
    "r_midpoint_5_of_lower_lash_line",      # 168
    "r_midpoint_6_of_lower_lash_line",      # 169
    # Right eye - lower eyelid
    "r_outer_end_of_lower_eyelid_line",     # 170
    "r_midpoint_3_of_lower_eyelid_line",    # 171
    "r_midpoint_1_of_lower_eyelid_line",    # 172
    "r_midpoint_4_of_lower_eyelid_line",    # 173
    "r_centerpoint_of_lower_eyelid_line",   # 174
    "r_midpoint_5_of_lower_eyelid_line",    # 175
    "r_midpoint_2_of_lower_eyelid_line",    # 176
    "r_midpoint_6_of_lower_eyelid_line",    # 177
    # Nose detail
    "tip_of_nose",                          # 178
    "bottom_center_of_nose",                # 179
    "r_outer_corner_of_nose",               # 180
    "l_outer_corner_of_nose",               # 181
    "inner_corner_of_r_nostril",            # 182
    "outer_corner_of_r_nostril",            # 183
    "upper_corner_of_r_nostril",            # 184
    "inner_corner_of_l_nostril",            # 185
    "outer_corner_of_l_nostril",            # 186
    "upper_corner_of_l_nostril",            # 187
    # Mouth - outer
    "r_outer_corner_of_mouth",              # 188
    "l_outer_corner_of_mouth",              # 189
    "center_of_cupid_bow",                  # 190
    "center_of_lower_outer_lip",            # 191
    "midpoint_1_of_upper_outer_lip",        # 192
    "midpoint_2_of_upper_outer_lip",        # 193
    "midpoint_1_of_lower_outer_lip",        # 194
    "midpoint_2_of_lower_outer_lip",        # 195
    "midpoint_3_of_upper_outer_lip",        # 196
    "midpoint_4_of_upper_outer_lip",        # 197
    "midpoint_5_of_upper_outer_lip",        # 198
    "midpoint_6_of_upper_outer_lip",        # 199
    "midpoint_3_of_lower_outer_lip",        # 200
    "midpoint_4_of_lower_outer_lip",        # 201
    "midpoint_5_of_lower_outer_lip",        # 202
    "midpoint_6_of_lower_outer_lip",        # 203
    # Mouth - inner
    "r_inner_corner_of_mouth",              # 204
    "l_inner_corner_of_mouth",              # 205
    "center_of_upper_inner_lip",            # 206
    "center_of_lower_inner_lip",            # 207
    "midpoint_1_of_upper_inner_lip",        # 208
    "midpoint_2_of_upper_inner_lip",        # 209
    "midpoint_1_of_lower_inner_lip",        # 210
    "midpoint_2_of_lower_inner_lip",        # 211
    "midpoint_3_of_upper_inner_lip",        # 212
    "midpoint_4_of_upper_inner_lip",        # 213
    "midpoint_5_of_upper_inner_lip",        # 214
    "midpoint_6_of_upper_inner_lip",        # 215
    "midpoint_3_of_lower_inner_lip",        # 216
    "midpoint_4_of_lower_inner_lip",        # 217
    "midpoint_5_of_lower_inner_lip",        # 218
    "midpoint_6_of_lower_inner_lip",        # 219
    # Ears (skipped for iBUG mapping, listed for completeness)
    # ... (220-267: ear landmarks, iris, pupil)
]
# fmt: on

_NAME2IDX = {name: i for i, name in enumerate(GOLIATH_KEYPOINTS)}

# ---------------------------------------------------------------------------
# Goliath → COCO-18 body mapping (OpenPose / DWPose body format)
# DWPose body: 0=nose, 1=neck, 2=R_shoulder, 3=R_elbow, 4=R_wrist,
#   5=L_shoulder, 6=L_elbow, 7=L_wrist, 8=R_hip, 9=R_knee, 10=R_ankle,
#   11=L_hip, 12=L_knee, 13=L_ankle, 14=R_eye, 15=L_eye, 16=R_ear, 17=L_ear
# ---------------------------------------------------------------------------
_GOLIATH_TO_COCO18 = [
    0,   # 0  nose
    69,  # 1  neck
    6,   # 2  R_shoulder
    8,   # 3  R_elbow
    41,  # 4  R_wrist
    5,   # 5  L_shoulder
    7,   # 6  L_elbow
    62,  # 7  L_wrist
    10,  # 8  R_hip
    12,  # 9  R_knee
    14,  # 10 R_ankle
    9,   # 11 L_hip
    11,  # 12 L_knee
    13,  # 13 L_ankle
    2,   # 14 R_eye
    1,   # 15 L_eye
    4,   # 16 R_ear
    3,   # 17 L_ear
]

# ---------------------------------------------------------------------------
# Goliath → DWPose 21-joint hand mapping
# DWPose hand: 0=wrist, 1-4=thumb(base→tip), 5-8=index, 9-12=middle,
#              13-16=ring, 17-20=pinky
# Goliath stores fingers tip→base, so we reverse each finger.
# ---------------------------------------------------------------------------
_GOLIATH_RIGHT_HAND_TO_DW21 = [
    41,  # 0  wrist
    24, 23, 22, 21,  # 1-4   thumb: base→tip
    28, 27, 26, 25,  # 5-8   index
    32, 31, 30, 29,  # 9-12  middle
    36, 35, 34, 33,  # 13-16 ring
    40, 39, 38, 37,  # 17-20 pinky
]

_GOLIATH_LEFT_HAND_TO_DW21 = [
    62,  # 0  wrist
    45, 44, 43, 42,  # 1-4   thumb
    49, 48, 47, 46,  # 5-8   index
    53, 52, 51, 50,  # 9-12  middle
    57, 56, 55, 54,  # 13-16 ring
    61, 60, 59, 58,  # 17-20 pinky
]

# ---------------------------------------------------------------------------
# Goliath → iBUG 68-point face mapping
# Only eyebrows (17-26), eyes (36-47), nose (27-35), mouth (48-67) are used.
# Contour (0-16) is left as (-1,-1) since optimized_face skips them.
# ---------------------------------------------------------------------------
_GOLIATH_TO_IBUG68 = {
    # --- Right eyebrow (17-21) inner→outer ---
    17: 78,   # upper_startpoint_of_r_eyebrow
    18: 81,   # upper_midpoint_1_of_r_eyebrow
    19: 83,   # upper_midpoint_2_of_r_eyebrow
    20: 84,   # upper_midpoint_3_of_r_eyebrow
    21: 80,   # end_of_r_eyebrow
    # --- Left eyebrow (22-26) inner→outer ---
    22: 87,   # upper_startpoint_of_l_eyebrow
    23: 90,   # upper_midpoint_1_of_l_eyebrow
    24: 92,   # upper_midpoint_2_of_l_eyebrow
    25: 93,   # upper_midpoint_3_of_l_eyebrow
    26: 89,   # end_of_l_eyebrow
    # --- Nose bridge (27-30) top→bottom ---
    27: 71,   # center_of_nose_root
    28: 73,   # midpoint_1_of_nose_bridge
    29: 74,   # midpoint_2_of_nose_bridge
    30: 72,   # tip_of_nose_bridge
    # --- Nose bottom (31-35) left→right ---
    31: 181,  # l_outer_corner_of_nose
    32: 185,  # inner_corner_of_l_nostril
    33: 178,  # tip_of_nose
    34: 182,  # inner_corner_of_r_nostril
    35: 180,  # r_outer_corner_of_nose
    # --- Right eye (36-41) clockwise from inner corner ---
    36: 120,  # r_inner_end_of_upper_lash_line
    37: 124,  # r_midpoint_2_of_upper_lash_line
    38: 127,  # r_midpoint_5_of_upper_lash_line
    39: 121,  # r_outer_end_of_upper_lash_line
    40: 168,  # r_midpoint_5_of_lower_lash_line
    41: 165,  # r_midpoint_2_of_lower_lash_line
    # --- Left eye (42-47) clockwise from inner corner ---
    42: 96,   # l_inner_end_of_upper_lash_line
    43: 99,   # l_midpoint_2_of_upper_lash_line
    44: 102,  # l_midpoint_5_of_upper_lash_line
    45: 97,   # l_outer_end_of_upper_lash_line
    46: 150,  # l_midpoint_5_of_lower_lash_line
    47: 147,  # l_midpoint_2_of_lower_lash_line
    # --- Outer mouth (48-59) clockwise from right corner ---
    48: 188,  # r_outer_corner_of_mouth
    49: 192,  # midpoint_1_of_upper_outer_lip
    50: 196,  # midpoint_3_of_upper_outer_lip
    51: 190,  # center_of_cupid_bow
    52: 198,  # midpoint_5_of_upper_outer_lip
    53: 193,  # midpoint_2_of_upper_outer_lip
    54: 189,  # l_outer_corner_of_mouth
    55: 195,  # midpoint_2_of_lower_outer_lip
    56: 202,  # midpoint_5_of_lower_outer_lip
    57: 191,  # center_of_lower_outer_lip
    58: 200,  # midpoint_3_of_lower_outer_lip
    59: 194,  # midpoint_1_of_lower_outer_lip
    # --- Inner mouth (60-67) clockwise from right corner ---
    60: 204,  # r_inner_corner_of_mouth
    61: 208,  # midpoint_1_of_upper_inner_lip
    62: 206,  # center_of_upper_inner_lip
    63: 209,  # midpoint_2_of_upper_inner_lip
    64: 205,  # l_inner_corner_of_mouth
    65: 211,  # midpoint_2_of_lower_inner_lip
    66: 207,  # center_of_lower_inner_lip
    67: 210,  # midpoint_1_of_lower_inner_lip
}

# Minimum confidence to consider a keypoint valid
CONF_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_sapiens_on_bbox(img_np, bbox, sapiens_dict, bbox_expand=0.15):
    """
    Run Sapiens pose estimation on a single person crop.

    Parameters
    ----------
    img_np : ndarray (H, W, 3) uint8 RGB
    bbox : array-like [x1, y1, x2, y2]
    sapiens_dict : dict from LoadSapiensNode
    bbox_expand : float, expand bbox by this fraction on each side.
        Helps detect limbs at screen edges by providing more context.
        Out-of-image regions are zero-padded.

    Returns
    -------
    dict with:
        "goliath_kp"  : ndarray (N, 3)  heatmap-space (x, y, conf)
        "bbox"        : (x1, y1, x2, y2) int  (expanded bbox, clamped)
        "pixel_kp"    : ndarray (N, 3)  image-space (x, y, conf)
    or None on failure.
    """
    model = sapiens_dict["model"]
    preprocessor = sapiens_dict["preprocessor"]
    device = sapiens_dict["device"]
    dtype = sapiens_dict["dtype"]
    hm_w, hm_h = sapiens_dict["heatmap_size"]

    img_h, img_w = img_np.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])

    # Expand bbox to give the model more context at edges
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * bbox_expand)
    pad_y = int(bh * bbox_expand)
    ex1, ey1 = x1 - pad_x, y1 - pad_y
    ex2, ey2 = x2 + pad_x, y2 + pad_y

    # Clamp to image bounds and compute padding needed
    cx1 = max(0, ex1)
    cy1 = max(0, ey1)
    cx2 = min(img_w, ex2)
    cy2 = min(img_h, ey2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    cropped = img_np[cy1:cy2, cx1:cx2]

    # Zero-pad if expanded bbox went outside image
    pad_left = cx1 - ex1
    pad_top = cy1 - ey1
    pad_right = ex2 - cx2
    pad_bottom = ey2 - cy2
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant", constant_values=0,
        )

    # The effective crop region in image coords
    eff_x1, eff_y1 = ex1, ey1  # may be negative
    eff_w, eff_h = ex2 - ex1, ey2 - ey1
    if eff_w <= 0 or eff_h <= 0:
        return None

    tensor = preprocessor(cropped).unsqueeze(0).to(device).to(dtype)
    heatmaps = model(tensor).to(torch.float32)  # (1, K, hm_h, hm_w)
    heatmaps = heatmaps[0].cpu().numpy()         # (K, hm_h, hm_w)

    num_kp = heatmaps.shape[0]
    goliath_kp = np.zeros((num_kp, 3), dtype=np.float32)
    for i in range(num_kp):
        hm = heatmaps[i]
        y_hm, x_hm = np.unravel_index(np.argmax(hm), hm.shape)
        goliath_kp[i] = (float(x_hm), float(y_hm), float(hm[y_hm, x_hm]))

    # Scale heatmap coords to image pixel coords (relative to expanded bbox)
    pixel_kp = np.zeros_like(goliath_kp)
    pixel_kp[:, 0] = goliath_kp[:, 0] * eff_w / hm_w + eff_x1
    pixel_kp[:, 1] = goliath_kp[:, 1] * eff_h / hm_h + eff_y1
    pixel_kp[:, 2] = goliath_kp[:, 2]

    return {
        "goliath_kp": goliath_kp,
        "bbox": (max(0, ex1), max(0, ey1), min(img_w, ex2), min(img_h, ey2)),
        "pixel_kp": pixel_kp,
    }


# ---------------------------------------------------------------------------
# Conversion to DWPose dict format
# ---------------------------------------------------------------------------

def goliath_to_dwpose(pixel_kp, img_h, img_w, conf_thr=CONF_THRESHOLD):
    """
    Convert Sapiens Goliath pixel keypoints to DWPose dict format.

    Parameters
    ----------
    pixel_kp : ndarray (N, 3) with (x, y, confidence) in image pixel coords
    img_h, img_w : output image dimensions
    conf_thr : minimum confidence threshold

    Returns
    -------
    dict compatible with ``draw_pose()`` from ``scail/draw_pose_utils.py``:
        bodies:  {candidate: [[18×2]], subset: [[18 indices]]}
        faces:   [[68×2]]
        hands:   [[21×2], [21×2]]  (left, right)
    All coordinates are normalised to [0, 1].
    """
    num_kp = pixel_kp.shape[0]

    def _get(goliath_idx):
        """Get normalised (x/W, y/H) and confidence for a Goliath index."""
        if goliath_idx >= num_kp:
            return -1.0, -1.0, 0.0
        x, y, c = pixel_kp[goliath_idx]
        if c < conf_thr:
            return -1.0, -1.0, 0.0
        return x / img_w, y / img_h, float(c)

    # --- Body (COCO 18) ---
    candidate = []
    subset_row = []
    for coco_idx, goliath_idx in enumerate(_GOLIATH_TO_COCO18):
        nx, ny, c = _get(goliath_idx)
        candidate.append([nx, ny])
        subset_row.append(coco_idx if c >= conf_thr else -1)

    # --- Hands (21 joints each) ---
    left_hand = []
    for goliath_idx in _GOLIATH_LEFT_HAND_TO_DW21:
        nx, ny, c = _get(goliath_idx)
        left_hand.append([nx, ny])

    right_hand = []
    for goliath_idx in _GOLIATH_RIGHT_HAND_TO_DW21:
        nx, ny, c = _get(goliath_idx)
        right_hand.append([nx, ny])

    # --- Face (iBUG 68) ---
    face = []
    for ibug_idx in range(68):
        goliath_idx = _GOLIATH_TO_IBUG68.get(ibug_idx)
        if goliath_idx is not None:
            nx, ny, c = _get(goliath_idx)
            face.append([nx, ny])
        else:
            # Contour points (0-16) — not mapped, skip rendering
            face.append([-1.0, -1.0])

    return {
        "bodies": {
            "candidate": [candidate],
            "subset": [subset_row],
        },
        "faces": [face],
        "hands": [left_hand, right_hand],
    }


def goliath_pixel_kp_to_flat137(pixel_kp, conf_thr=CONF_THRESHOLD):
    """
    Convert Goliath pixel keypoints to a flat (137, 3) array for timeline
    storage and temporal smoothing.  Layout: 18 body + 21 left hand +
    21 right hand + 9 nose + 68 face = 137.

    This is NOT the same as SMPLest-X 137 joints — it's our own layout
    for Sapiens that's easy to convert back to DWPose dict later.

    Layout:
        [0:18]    COCO-18 body
        [18:39]   left hand (DWPose 21)
        [39:60]   right hand (DWPose 21)
        [60:128]  iBUG 68 face
        [128:137] reserved (zeros)
    """
    num_kp = pixel_kp.shape[0]
    out = np.zeros((137, 3), dtype=np.float32)

    def _safe(goliath_idx):
        if goliath_idx >= num_kp:
            return 0.0, 0.0, 0.0
        return pixel_kp[goliath_idx]

    # Body
    for i, gi in enumerate(_GOLIATH_TO_COCO18):
        out[i] = _safe(gi)

    # Left hand
    for i, gi in enumerate(_GOLIATH_LEFT_HAND_TO_DW21):
        out[18 + i] = _safe(gi)

    # Right hand
    for i, gi in enumerate(_GOLIATH_RIGHT_HAND_TO_DW21):
        out[39 + i] = _safe(gi)

    # Face
    for ibug_idx in range(68):
        gi = _GOLIATH_TO_IBUG68.get(ibug_idx)
        if gi is not None:
            out[60 + ibug_idx] = _safe(gi)

    return out


def flat137_to_dwpose(kp137, img_h, img_w, conf_thr=CONF_THRESHOLD):
    """
    Convert our flat 137-array back to DWPose dict format for rendering.
    """
    def _norm(idx):
        x, y, c = kp137[idx]
        if c < conf_thr:
            return [-1.0, -1.0]
        return [x / img_w, y / img_h]

    def _subset(idx):
        return idx if kp137[idx, 2] >= conf_thr else -1

    candidate = [_norm(i) for i in range(18)]
    subset_row = [_subset(i) for i in range(18)]
    left_hand = [_norm(18 + i) for i in range(21)]
    right_hand = [_norm(39 + i) for i in range(21)]
    face = [_norm(60 + i) for i in range(68)]

    return {
        "bodies": {
            "candidate": [candidate],
            "subset": [subset_row],
        },
        "faces": [face],
        "hands": [left_hand, right_hand],
    }
