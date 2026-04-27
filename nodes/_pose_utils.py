"""Shared utilities for pose estimation nodes."""

import numpy as np


# ---------------------------------------------------------------------------
# OpenPose 25 -> COCO WholeBody 133 joint mapping
# ---------------------------------------------------------------------------
# Maps OpenPose body index -> COCO-WB index.
# 23 of 25 OpenPose joints have a 1:1 mapping.
# OpenPose 1 (Neck) and 8 (MidHip) have no direct COCO equivalent.

OPENPOSE25_TO_COCO_WB = {
    0: 0,    # Nose
    # 1: Neck  -> no direct mapping
    2: 6,    # R_Shoulder
    3: 8,    # R_Elbow
    4: 10,   # R_Wrist
    5: 5,    # L_Shoulder
    6: 7,    # L_Elbow
    7: 9,    # L_Wrist
    # 8: MidHip -> no direct mapping
    9: 12,   # R_Hip
    10: 14,  # R_Knee
    11: 16,  # R_Ankle
    12: 11,  # L_Hip
    13: 13,  # L_Knee
    14: 15,  # L_Ankle
    15: 2,   # R_Eye
    16: 1,   # L_Eye
    17: 4,   # R_Ear
    18: 3,   # L_Ear
    19: 17,  # L_BigToe
    20: 18,  # L_SmallToe
    21: 19,  # L_Heel
    22: 20,  # R_BigToe
    23: 21,  # R_SmallToe
    24: 22,  # R_Heel
}


def openpose25_to_coco_wholebody(op_kp2d):
    """
    Convert OpenPose 25-joint body keypoints to COCO WholeBody 133 format.

    Only body (0-16) and feet (17-22) slots are filled.
    Face (23-90) and hands (91-132) remain zero.

    Parameters
    ----------
    op_kp2d : (25, 2) or (25, 3) ndarray
        OpenPose 2D keypoints. If 3 columns, col 2 is confidence.

    Returns
    -------
    coco_wb : (133, 3) float32 array  (x, y, confidence)
    """
    coco_wb = np.zeros((133, 3), dtype=np.float32)
    has_conf = op_kp2d.shape[1] >= 3

    for op_idx, coco_idx in OPENPOSE25_TO_COCO_WB.items():
        coco_wb[coco_idx, 0] = op_kp2d[op_idx, 0]
        coco_wb[coco_idx, 1] = op_kp2d[op_idx, 1]
        coco_wb[coco_idx, 2] = op_kp2d[op_idx, 2] if has_conf else 1.0

    return coco_wb


def fuse_3d_body_with_sapiens(op_kp2d, sapiens_kp,
                              show_face=True, show_hand_foot=True):
    """
    Fuse PromptHMR 3D body+feet with Sapiens face+hands.

    Body+feet (COCO-WB 0-22) come from PromptHMR's OpenPose 25 joints.
    Face (23-90) and hands (91-132) come from Sapiens if available.

    Parameters
    ----------
    op_kp2d : (25, 2) or (25, 3) ndarray
        OpenPose 25-joint 2D keypoints from PromptHMR.
    sapiens_kp : (133, 3) ndarray or None
        Sapiens COCO-WholeBody keypoints. If None, face/hands are zero.
    show_face : bool
        Include face keypoints (23-90) from Sapiens.
    show_hand_foot : bool
        Include hand keypoints (91-132) from Sapiens.

    Returns
    -------
    coco_wb : (133, 3) float32 array
    """
    # Start with 3D body+feet
    coco_wb = openpose25_to_coco_wholebody(op_kp2d)

    # Fill face + hands from Sapiens, and prefer Sapiens for the head
    # slots (nose/eyes/ears) of the body too.  Reason: 3D backbones
    # like NLF don't regress face joints (their SMPL skeleton only has
    # a Head joint near the crown), so these slots may be zero after
    # the OpenPose-25 -> COCO-WB mapping.  Sapiens is a 2D keypoint
    # model specialised for faces and always gives reliable nose/eye/
    # ear locations for visible persons.
    if sapiens_kp is not None:
        # COCO-WB head slots: 0 Nose, 1 L_Eye, 2 R_Eye, 3 L_Ear, 4 R_Ear.
        # Overwrite when Sapiens has confidence; otherwise keep whatever
        # came from the 3D backbone (which may be zero).
        for head_idx in (0, 1, 2, 3, 4):
            if sapiens_kp[head_idx, 2] > 0.1:
                coco_wb[head_idx] = sapiens_kp[head_idx]

        if show_face:
            coco_wb[23:91] = sapiens_kp[23:91]    # face (68 keypoints)
        if show_hand_foot:
            coco_wb[91:133] = sapiens_kp[91:133]  # hands (42 keypoints)

    return coco_wb


# ---------------------------------------------------------------------------
# Frame rate resampling utilities
# ---------------------------------------------------------------------------

def resample_keypoints(timeline, fps_in, target_fps=30.0):
    """
    Resample a single-person keypoint timeline from fps_in to target_fps
    using linear interpolation between adjacent frames.

    Parameters
    ----------
    timeline : list of (K, 3) arrays or None
    fps_in : float
    target_fps : float

    Returns
    -------
    resampled : list of (K, 3) arrays or None
    src_indices : list of int, nearest source frame index per output frame
    """
    n_in = len(timeline)
    if n_in < 2:
        return list(timeline), list(range(n_in))

    duration = (n_in - 1) / fps_in
    n_out = max(1, int(round(duration * target_fps)) + 1)

    resampled = []
    src_indices = []
    for i in range(n_out):
        t_sec = i / target_fps
        t_in = t_sec * fps_in

        j0 = min(int(t_in), n_in - 1)
        j1 = min(j0 + 1, n_in - 1)
        alpha = t_in - j0

        src_indices.append(min(int(round(t_in)), n_in - 1))

        if j0 == j1 or alpha < 1e-6:
            resampled.append(
                timeline[j0].copy() if timeline[j0] is not None else None)
        elif timeline[j0] is not None and timeline[j1] is not None:
            resampled.append(
                timeline[j0] * (1 - alpha) + timeline[j1] * alpha)
        elif timeline[j0] is not None:
            resampled.append(timeline[j0].copy())
        elif timeline[j1] is not None:
            resampled.append(timeline[j1].copy())
        else:
            resampled.append(None)

    return resampled, src_indices


# ---------------------------------------------------------------------------
# OpenPose 25 -> DWPose 18-body mapping (for SCAIL compatibility)
# ---------------------------------------------------------------------------
# DWPose body 18 joints: Nose, Neck, R_Shoulder, R_Elbow, R_Wrist,
# L_Shoulder, L_Elbow, L_Wrist, R_Hip, R_Knee, R_Ankle,
# L_Hip, L_Knee, L_Ankle, R_Eye, L_Eye, R_Ear, L_Ear

_OP25_TO_DW18 = {
    0: 0,    # Nose
    # 1: Neck -> synthesise from (R_Shoulder + L_Shoulder) / 2
    2: 2,    # R_Shoulder
    3: 3,    # R_Elbow
    4: 4,    # R_Wrist
    5: 5,    # L_Shoulder
    6: 6,    # L_Elbow
    7: 7,    # L_Wrist
    9: 8,    # R_Hip
    10: 9,   # R_Knee
    11: 10,  # R_Ankle
    12: 11,  # L_Hip
    13: 12,  # L_Knee
    14: 13,  # L_Ankle
    15: 14,  # R_Eye
    16: 15,  # L_Eye
    17: 16,  # R_Ear
    18: 17,  # L_Ear
}


def openpose25_to_dwpose_body(op_kp2d, img_w, img_h):
    """
    Convert OpenPose 25-joint 2D keypoints to DWPose 18-joint body format.

    Parameters
    ----------
    op_kp2d : (25, 2) or (25, 3) ndarray  – pixel coordinates
    img_w, img_h : int – image dimensions for normalisation

    Returns
    -------
    candidate : (18, 2) float32 – normalised [0, 1]
    subset : (18,) float32 – joint index if valid, -1 if missing
    """
    candidate = np.zeros((18, 2), dtype=np.float32)
    subset = np.full(18, -1.0, dtype=np.float32)

    for op_idx, dw_idx in _OP25_TO_DW18.items():
        x, y = op_kp2d[op_idx, 0], op_kp2d[op_idx, 1]
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            continue
        candidate[dw_idx, 0] = x / img_w
        candidate[dw_idx, 1] = y / img_h
        subset[dw_idx] = dw_idx

    # Neck = midpoint of R_Shoulder and L_Shoulder
    r_sh = op_kp2d[2]
    l_sh = op_kp2d[5]
    if not (abs(r_sh[0]) < 1e-6 and abs(r_sh[1]) < 1e-6) and \
       not (abs(l_sh[0]) < 1e-6 and abs(l_sh[1]) < 1e-6):
        candidate[1, 0] = (r_sh[0] + l_sh[0]) / 2 / img_w
        candidate[1, 1] = (r_sh[1] + l_sh[1]) / 2 / img_h
        subset[1] = 1.0

    return candidate, subset


def coco_wb133_to_dwpose_face_hands(coco_wb, img_w, img_h, conf_thr=0.3):
    """
    Extract face (68 pts) and hands (left 21 + right 21) from
    COCO WholeBody 133-format keypoints in DWPose format.

    Keypoints with confidence below *conf_thr* are zeroed out so the
    renderer's ``x > eps`` check will skip them (avoids spurious lines
    from noisy heatmap argmax positions).

    Parameters
    ----------
    coco_wb : (133, 3) ndarray – pixel coords + confidence
    img_w, img_h : int
    conf_thr : float – minimum confidence to keep a keypoint

    Returns
    -------
    face : (68, 2) float32 – normalised [0, 1]
    right_hand : (21, 2) float32 – normalised [0, 1]
    left_hand : (21, 2) float32 – normalised [0, 1]
    """
    # COCO-WB: face = 23..90 (68 pts), left hand = 91..111, right hand = 112..132
    face_slice = coco_wb[23:91]        # (68, 3)
    left_hand_slice = coco_wb[91:112]  # (21, 3)
    right_hand_slice = coco_wb[112:133]  # (21, 3)

    def _normalise(kp_slice, n, w, h):
        out = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            if kp_slice[i, 2] >= conf_thr:
                out[i, 0] = kp_slice[i, 0] / w
                out[i, 1] = kp_slice[i, 1] / h
        return out

    face = _normalise(face_slice, 68, img_w, img_h)
    left_hand = _normalise(left_hand_slice, 21, img_w, img_h)
    right_hand = _normalise(right_hand_slice, 21, img_w, img_h)

    return face, right_hand, left_hand


# ---------------------------------------------------------------------------
# Temporal outlier filtering for 2D keypoints
# ---------------------------------------------------------------------------

def temporal_filter_keypoints(timeline, velocity_threshold=3.0,
                              window_size=5, smooth_sigma=0.0):
    """
    Detect and repair outlier keypoints that jump to wrong positions in
    isolated frames.  Operates on a single-person timeline.

    Algorithm
    ---------
    1. For each joint, compute per-frame displacement (L2 pixel distance).
    2. Compute a rolling median displacement over *window_size* frames.
    3. If a frame's displacement exceeds *velocity_threshold* × rolling
       median, AND the next frame also shows a snap-back (high displacement),
       mark the frame as an outlier.
    4. Replace outlier frames with linear interpolation from nearest good
       neighbours.
    5. Optionally apply a light Gaussian smooth (sigma > 0).

    Parameters
    ----------
    timeline : list of (K, C) ndarray or None, length T
        Per-frame keypoints.  C >= 2 (x, y, ...).  None = missing frame.
    velocity_threshold : float
        Multiplier on rolling median displacement to flag outliers.
    window_size : int
        Window for rolling median (must be odd).
    smooth_sigma : float
        If > 0, apply Gaussian smoothing after outlier repair.  0 = no
        smoothing.

    Returns
    -------
    filtered : list of (K, C) ndarray or None, same length as input
    n_repaired : int – number of frames that were repaired
    """
    T = len(timeline)
    if T < 3:
        return [x.copy() if x is not None else None for x in timeline], 0

    # Find contiguous non-None segments
    data = [x.copy() if x is not None else None for x in timeline]
    K = None
    C = None
    for d in data:
        if d is not None:
            K, C = d.shape
            break
    if K is None:
        return data, 0

    # Build a (T, K, 2) array of xy positions; track valid mask
    xy = np.zeros((T, K, 2), dtype=np.float64)
    valid = np.zeros(T, dtype=bool)
    for t in range(T):
        if data[t] is not None:
            xy[t] = data[t][:, :2]
            valid[t] = True

    # Per-joint displacement between consecutive valid frames
    # disp[t] = ||xy[t] - xy[prev_valid]|| per joint
    disp = np.zeros((T, K), dtype=np.float64)
    prev_valid_idx = -1
    for t in range(T):
        if not valid[t]:
            continue
        if prev_valid_idx >= 0:
            diff = xy[t] - xy[prev_valid_idx]
            disp[t] = np.sqrt((diff ** 2).sum(axis=-1))
        prev_valid_idx = t

    # Mean displacement across joints per frame
    mean_disp = disp.mean(axis=1)  # (T,)

    # Rolling median of mean displacement
    half_w = window_size // 2
    rolling_med = np.zeros(T, dtype=np.float64)
    for t in range(T):
        lo = max(0, t - half_w)
        hi = min(T, t + half_w + 1)
        vals = mean_disp[lo:hi]
        vals = vals[vals > 1e-6]  # ignore zero entries
        rolling_med[t] = np.median(vals) if len(vals) > 0 else 0.0

    # Detect outliers: spike followed by snap-back
    outlier = np.zeros(T, dtype=bool)
    for t in range(1, T - 1):
        if not valid[t]:
            continue
        med = rolling_med[t]
        if med < 1e-6:
            continue
        # Current frame has abnormally high displacement
        if mean_disp[t] > velocity_threshold * med:
            # Check if next valid frame also has high displacement (snap-back)
            # or just mark this frame as suspicious
            next_t = -1
            for nt in range(t + 1, min(t + 4, T)):
                if valid[nt]:
                    next_t = nt
                    break
            if next_t > 0 and mean_disp[next_t] > velocity_threshold * med * 0.5:
                outlier[t] = True
            elif next_t > 0:
                # Even without snap-back, if displacement is very extreme, flag it
                if mean_disp[t] > velocity_threshold * 2 * med:
                    outlier[t] = True

    # Also check for isolated single-frame spikes where the frame before
    # and after are both good (and close to each other)
    for t in range(1, T - 1):
        if not valid[t] or outlier[t]:
            continue
        if not valid[t - 1] or not valid[t + 1]:
            continue
        # Displacement from t-1 to t+1 should be small if t is an outlier
        skip_disp = np.sqrt(((xy[t + 1] - xy[t - 1]) ** 2).sum(axis=-1)).mean()
        if skip_disp < mean_disp[t] * 0.3 and mean_disp[t] > velocity_threshold * rolling_med[t]:
            outlier[t] = True

    n_repaired = int(outlier.sum())

    # Replace outliers with linear interpolation
    for t in range(T):
        if not outlier[t]:
            continue
        # Find nearest good frame before and after
        before = -1
        for b in range(t - 1, -1, -1):
            if valid[b] and not outlier[b]:
                before = b
                break
        after = -1
        for a in range(t + 1, T):
            if valid[a] and not outlier[a]:
                after = a
                break

        if before >= 0 and after >= 0:
            alpha = (t - before) / (after - before)
            data[t] = data[before] * (1 - alpha) + data[after] * alpha
        elif before >= 0:
            data[t] = data[before].copy()
        elif after >= 0:
            data[t] = data[after].copy()
        # else: leave as-is

    # Optional Gaussian smoothing
    if smooth_sigma > 0 and n_repaired >= 0:
        from scipy.ndimage import gaussian_filter1d
        # Build array of valid frames
        arr = np.zeros((T, K, C), dtype=np.float64)
        vmask = np.zeros(T, dtype=bool)
        for t in range(T):
            if data[t] is not None:
                arr[t] = data[t]
                vmask[t] = True

        if vmask.sum() > 2:
            # Only smooth xy columns, preserve confidence
            for k in range(K):
                for c in range(min(2, C)):
                    vals = arr[:, k, c].copy()
                    # Only smooth where valid
                    if vmask.all():
                        vals = gaussian_filter1d(vals, sigma=smooth_sigma)
                    else:
                        # Interpolate gaps first, smooth, then mask back
                        valid_idx = np.where(vmask)[0]
                        if len(valid_idx) > 1:
                            vals_interp = np.interp(
                                np.arange(T), valid_idx, vals[valid_idx])
                            vals_interp = gaussian_filter1d(
                                vals_interp, sigma=smooth_sigma)
                            vals = vals_interp
                    arr[:, k, c] = vals

            for t in range(T):
                if data[t] is not None:
                    data[t] = arr[t].astype(data[t].dtype)

    return data, n_repaired


def compute_resampled_indices(n_in, fps_in, target_fps=30.0):
    """
    Compute nearest source frame indices for resampling from fps_in to
    target_fps. Works for both upsampling and downsampling.

    Returns list of source frame indices (length = output frame count).
    """
    if n_in < 2:
        return list(range(n_in))

    duration = (n_in - 1) / fps_in
    n_out = max(1, int(round(duration * target_fps)) + 1)

    indices = []
    for i in range(n_out):
        t_sec = i / target_fps
        j = min(int(round(t_sec * fps_in)), n_in - 1)
        indices.append(j)

    return indices


# ---------------------------------------------------------------------------
# Face visibility (for render-time face-slot gating)
# ---------------------------------------------------------------------------
# Three independent signals, tried in priority order:
#   1. 3D body normal (smpl_j3d shoulder × torso) — strongest. Works even
#      when the subject is upside down or in odd poses; just measures
#      "does the chest face the camera".
#   2. RTMW / ViTPose face mean confidence — strong, data-driven. Top-
#      down 2D estimators trained on diverse data drop face-landmark
#      confidence sharply on back/occluded faces (unlike top-down BODY
#      kpts which they hallucinate). FaRL has no per-landmark
#      confidence (output is hard-coded to 1.0), so this signal must be
#      captured upstream BEFORE FaRL overrides the face slots and
#      stashed in poses["_face_conf_2d"].
#   3. 2D head-kpt geometry (eye-ear sides, eye-vs-ear distance ratio,
#      nose between eyes, ear-eye conf asymmetry). Last-resort fallback
#      when the above are unavailable. Note: deliberately does NOT
#      include "nose above eyes" — scenes with inverted/upside-down
#      subjects would otherwise reject valid faces.

# SMPL-24 indices used for the 3D body-normal check.
_SMPL_PELVIS = 0
_SMPL_NECK = 12
_SMPL_L_SHOULDER = 16
_SMPL_R_SHOULDER = 17


def _is_face_visible_2d(head_kp, eye_conf_thresh=0.1):
    """Return True when 5 COCO-17 head keypoints suggest a forward-
    facing face. Returns False on geometry/confidence patterns typical
    of back views or hallucinated kpts. Used as the last-resort
    fallback in is_face_visible().

    Parameters
    ----------
    head_kp : (5, 2) or (5, 3) ndarray
        COCO-17 indices 0..4 (nose, subj L_eye, subj R_eye,
        subj L_ear, subj R_ear). The 3rd column (confidence) is
        used by signals 1 and 4 when available.
    eye_conf_thresh : float
        Minimum confidence to consider a keypoint "available" for
        signals that need conf. Defaults to 0.1.

    Notes
    -----
    Deliberately omits the historical "nose above eyes" check — it
    false-rejects upright valid faces in scenes with inverted /
    upside-down subjects. The remaining 4 signals are scale- and
    orientation-invariant.
    """
    if head_kp.shape[0] < 5:
        return True  # not enough info to reject
    has_conf = head_kp.shape[1] >= 3
    if has_conf:
        n_conf = float(head_kp[0, 2])
        le_c, re_c = float(head_kp[1, 2]), float(head_kp[2, 2])
        la_c, ra_c = float(head_kp[3, 2]), float(head_kp[4, 2])
        eyes_min_conf = min(le_c, re_c)
        ears_min_conf = min(la_c, ra_c)
        eyes_avg_conf = 0.5 * (le_c + re_c)
        ears_avg_conf = 0.5 * (la_c + ra_c)
    else:
        n_conf = 1.0
        eyes_min_conf = ears_min_conf = 1.0
        eyes_avg_conf = ears_avg_conf = 1.0

    nose = head_kp[0, :2]
    l_eye, r_eye = head_kp[1, :2], head_kp[2, :2]
    l_ear, r_ear = head_kp[3, :2], head_kp[4, :2]

    eye_dx = float(l_eye[0] - r_eye[0])
    ear_dx = float(l_ear[0] - r_ear[0])
    eye_dist = float(np.hypot(*(l_eye - r_eye)))
    ear_dist = float(np.hypot(*(l_ear - r_ear)))

    # Signal 1 — anatomical-side consistency. For a face that's actually
    # looking AT the camera, subj_L_eye and subj_L_ear lie on the same
    # side of subj_R_*; back views often invert this.
    if (
        ears_min_conf > eye_conf_thresh
        and ear_dist > 5.0
        and abs(eye_dx) > 1.0
        and (eye_dx * ear_dx < 0)
    ):
        return False

    # Signal 2 — eye distance vs ear distance. Real frontal faces have
    # inter-eye ≈ 0.30..0.50 × inter-ear; hallucinated back views
    # commonly inflate the ratio.
    if (
        ears_min_conf > eye_conf_thresh
        and ear_dist > 5.0
        and eye_dist > 0.0
        and eye_dist > 0.65 * ear_dist
    ):
        return False

    # Signal 3 — nose lies between the eyes in image x (with pad).
    eye_min_x = min(l_eye[0], r_eye[0])
    eye_max_x = max(l_eye[0], r_eye[0])
    if eye_dist > 5.0:
        pad = eye_dist * 0.35
        if nose[0] < eye_min_x - pad or nose[0] > eye_max_x + pad:
            return False

    # Signal 4 — confidence asymmetry: ears confident but eyes are not.
    # Top-down regressors often locate ears on back views (silhouette
    # is locatable from behind) but eye conf softens.
    if (
        ears_avg_conf > eyes_avg_conf + 0.20
        and ears_min_conf > 0.30
        and n_conf < 0.30
    ):
        return False

    return True


def is_face_visible(
    poses,
    p_idx,
    t,
    forward_z_threshold=0.1,
    face_conf_threshold=0.30,
):
    """Return True if the face at (person p_idx, frame t) appears to
    be visible to camera, False otherwise. Tries three signals in
    priority order; first conclusive one wins.

    1. 3D body normal — `smpl_j3d[t]` shoulder × torso → forward vector.
       Visible when normalised forward.z < forward_z_threshold (camera
       looks down +Z, so negative-z = chest faces camera). Threshold
       0.1 admits slight past-profile yaws but rejects clear back.
    2. RTMW / ViTPose face mean confidence stashed in
       `poses["_face_conf_2d"][p_idx][t]`. Visible when score >=
       face_conf_threshold. Captured upstream by BMPRTMWPose BEFORE
       FaRL overrides the face slots.
    3. 2D head-kpt geometry via _is_face_visible_2d().

    Returns True (permissive default) when all three signals are
    unavailable — caller's `show_face` toggle is the master kill
    switch when the user wants to force-hide all faces.
    """
    person = poses["persons"][p_idx]

    # -- Signal 1: 3D body normal -----------------------------------------
    smpl_list = person.get("smpl_j3d")
    if smpl_list is not None and 0 <= t < len(smpl_list):
        j = smpl_list[t]
        if j is not None and getattr(j, "shape", (0,))[0] >= _SMPL_R_SHOULDER + 1:
            shoulder = np.asarray(j[_SMPL_L_SHOULDER]) - np.asarray(j[_SMPL_R_SHOULDER])
            torso = np.asarray(j[_SMPL_NECK]) - np.asarray(j[_SMPL_PELVIS])
            forward = np.cross(shoulder[:3], torso[:3])
            fmag = float(np.linalg.norm(forward))
            if fmag > 1e-3:
                return float(forward[2]) / fmag < forward_z_threshold

    # -- Signal 2: stashed pre-FaRL 2D face mean conf ---------------------
    face_conf_mat = poses.get("_face_conf_2d")
    if (
        face_conf_mat is not None
        and 0 <= p_idx < len(face_conf_mat)
        and 0 <= t < len(face_conf_mat[p_idx])
    ):
        score = face_conf_mat[p_idx][t]
        if score is not None:
            return float(score) >= face_conf_threshold

    # -- Signal 3: 2D head-kpt geometry -----------------------------------
    kp_list = person.get("keypoints")
    if kp_list is not None and 0 <= t < len(kp_list):
        kp = kp_list[t]
        if kp is not None and getattr(kp, "shape", (0,))[0] >= 5:
            return _is_face_visible_2d(kp[:5])

    # Nothing to go on — permissive default.
    return True
