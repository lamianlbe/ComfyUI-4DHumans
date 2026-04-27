"""
Load BBoxMaskPose (BMP) — iterative detect → pose → SAM2-mask pipeline
tuned for occluded crowd scenes.

Paper: Purkrabek et al., ICCV 2025 (arxiv:2412.01562), SOTA on OCHuman
for 2D pose (49.2 AP) and human-instance segmentation (34 AP).

Unlike our other segmenters (SAM3 / YOLO-seg / CrowdSAM), BMP gives us
**three outputs from a single forward**: per-instance bboxes, masks,
AND 17-point COCO keypoints — because its internal SAM2 is prompted
by the pose keypoints (not bboxes), so pose is a load-bearing part
of the mask generation path, not an afterthought.

BMP's relative strengths based on our tests:
  * 2D keypoints:   strong (beats YOLO11m-Pose on occluded frames)
  * Masks:          good for typical separation, but merges people
                    when RTMDet can't draw two bboxes for tightly
                    overlapping bodies
  * Speed:          ~200-500 ms / frame on Blackwell class GPUs

This node loads the full three-model stack (RTMDet-ins-L detector,
SAM 2.1 Hiera base+ for pose-prompted segmentation, PMPose-b for
pose estimation) and returns a single dict downstream nodes consume.

Weight paths (ComfyUI convention — place under ``models/bmp/``):

    models/bmp/rtmdet-ins-l-mask.pth          ~200 MB
    models/bmp/SAM-pose2seg_hiera_b+.pt       ~300 MB
    models/bmp/PMPose-b-1.0.0.pth             ~350 MB

Download from HuggingFace ``vrg-prague/BBoxMaskPose``:
    https://huggingface.co/vrg-prague/BBoxMaskPose/tree/main

If a weight is present locally we use it; if missing, we fall back to
BMP's auto-download-from-HF behaviour (cached under ~/.cache/huggingface
/hub/). You can mix: e.g. put the big detector locally to avoid re-fetch
while letting the others auto-download.

Prerequisites (pip installed into your ComfyUI env):
    - mmcv >= 2.2.0
    - mmdet >= 3.3.0
    - mmpose >= 1.3.1
    - mmpretrain (transient dep of mmpose registry)
    - bboxmaskpose  (pip install from github.com/MiraPurkrabek/BBoxMaskPose)
    - sam2 (Meta's SAM 2.1, pulled by BMP)

License note: BMP is GPL-3.0. By loading this node, your ComfyUI
graph inherits that license for any derived works distributed with
BMP weights / inference results.
"""

import logging
import os
import tempfile

from folder_paths import models_dir

_logger = logging.getLogger(__name__)


# PMPose's pip package ships pmpose/api.py but NOT the mmpose config
# files that its DEFAULT_CONFIGS points at — those live in the BMP
# repo's mmpose fork under ``mmpose/configs/{ProbMaskPose,MaskPose}/``
# and aren't part of the ``pmpose`` wheel. So an out-of-the-box
# ``PMPose(variant=...)`` call crashes with FileNotFoundError trying
# to open ``.../site-packages/mmpose/configs/ProbMaskPose/PMPose-b-1.0.0.py``.
#
# We vendor the full config set (~160 KB: 13 variant .py files + the
# shared ``_base_/default_runtime.py``) into bmp_configs/ next to this
# package, and pass the resolved path explicitly via
# ``PMPose(config_path=...)`` to override the broken default lookup.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BMP_CONFIGS_ROOT = os.path.join(REPO_ROOT, "bmp_configs")

# Variant → relative path under {mmpose,bmp_configs}/configs/. Mirrors
# pmpose.api's DEFAULT_CONFIGS table. PMPose lives under
# ProbMaskPose/, MaskPose lives under MaskPose/.
_PMPOSE_CONFIG_RELPATH = {
    # PMPose (full w/ presence + visibility heads, v1.0.0)
    "PMPose-s":       "ProbMaskPose/PMPose-s-1.0.0.py",
    "PMPose-b":       "ProbMaskPose/PMPose-b-1.0.0.py",
    "PMPose-l":       "ProbMaskPose/PMPose-l-1.0.0.py",
    "PMPose-h":       "ProbMaskPose/PMPose-h-1.0.0.py",
    # MaskPose (predecessor, v1.1.0 for size variants)
    "MaskPose-s":     "MaskPose/MaskPose-s-1.1.0.py",
    "MaskPose-b":     "MaskPose/MaskPose-b-1.1.0.py",
    "MaskPose-l":     "MaskPose/MaskPose-l-1.1.0.py",
    "MaskPose-h":     "MaskPose/MaskPose-h-1.1.0.py",
}


def _resolve_pmpose_config(rel_path: str) -> str:
    """Find a PMPose / MaskPose mmpose config file. Prefer the installed
    mmpose package's bundled configs (so a wheel-only install works on
    any host), fall back to vendored paths if the installed mmpose
    lacks them.
    """
    try:
        import mmpose
        installed_root = os.path.dirname(os.path.abspath(mmpose.__file__))
        candidate = os.path.join(installed_root, "configs", rel_path)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    return os.path.join(BMP_CONFIGS_ROOT, rel_path)


def _resolve_bmp_yaml(rel_path: str) -> str:
    """Find a bboxmaskpose config YAML. Prefer the installed
    bboxmaskpose package's bundled configs, fall back to vendored.

    rel_path examples:
      "configs/bmp_v2.yaml"
      "sam2/configs/sam-pose2seg/sam-pose2seg_hiera_b+.yaml"
    """
    try:
        import bboxmaskpose
        installed_root = os.path.dirname(os.path.abspath(bboxmaskpose.__file__))
        candidate = os.path.join(installed_root, rel_path)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    # Vendored layout was bmp_configs/{bmp,sam2}/* — adapt rel_path to
    # match: "configs/bmp_v2.yaml" → "bmp/bmp_v2.yaml", etc.
    if rel_path.startswith("configs/"):
        return os.path.join(
            BMP_CONFIGS_ROOT, "bmp", rel_path[len("configs/"):],
        )
    if rel_path.startswith("sam2/configs/"):
        return os.path.join(
            BMP_CONFIGS_ROOT, "sam2", rel_path[len("sam2/configs/"):],
        )
    return os.path.join(BMP_CONFIGS_ROOT, rel_path)


def _resolve_mmdet_config(rel_path: str) -> str:
    """Find an mmdet config file. Prefer installed mmdet, fall back to
    bmp_configs/mmdet/."""
    try:
        import mmdet
        installed_root = os.path.dirname(os.path.abspath(mmdet.__file__))
        candidate = os.path.join(installed_root, ".mim", "configs", rel_path)
        if os.path.isfile(candidate):
            return candidate
        # Some mmdet wheels also ship configs alongside the package
        # (not in .mim/), try that too.
        candidate2 = os.path.join(installed_root, "configs", rel_path)
        if os.path.isfile(candidate2):
            return candidate2
    except ImportError:
        pass
    # The vendored detector tree lives under bmp_configs/mmdet/...
    # without a "mmdet/" prefix (we cherry-picked just rtmdet/ + base).
    if rel_path.startswith("mmdet/"):
        return os.path.join(BMP_CONFIGS_ROOT, rel_path)
    return os.path.join(BMP_CONFIGS_ROOT, "mmdet", rel_path)


# BMP config aliases shipped in bboxmaskpose/configs/. Each trades off
# how aggressively SAM2 is prompted (num_pos_keypoints) vs recall.
_BMP_CONFIGS = ["bmp_v2", "bmp_D3", "bmp_J1"]

# PMPose variants shipped in HF. 'b' is the default (best recall/speed
# balance per the paper); 's'/'l'/'h' trade accuracy for speed.
_PMPOSE_VARIANTS = [
    "PMPose-s", "PMPose-b", "PMPose-l", "PMPose-h",
    "MaskPose-s", "MaskPose-b", "MaskPose-l", "MaskPose-h",
]

# ComfyUI models/ subdir where we expect BMP weights to live.
BMP_MODELS_DIR = os.path.join(models_dir, "bmp")

# Fixed weight filenames — these match the URL basenames from HF so
# users can ``wget -O <dir>/<filename> <url>`` verbatim.
RTMDET_FILENAME = "rtmdet-ins-l-mask.pth"
SAM2_POSE2SEG_FILENAME = "SAM-pose2seg_hiera_b+.pt"

# URLs used when the local file is absent. Mirror whatever BMP ships
# in its configs/*.yaml.
RTMDET_URL = (
    "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/"
    + RTMDET_FILENAME
)
SAM2_POSE2SEG_URL = (
    # '+' must be URL-escaped as %2B when fetching over HTTP.
    "https://huggingface.co/vrg-prague/BBoxMaskPose/resolve/main/"
    "SAM-pose2seg_hiera_b%2B.pt"
)


def _pose_filename_from_variant(variant: str) -> str:
    """Upstream version strings differ between PMPose (1.0.0) and
    MaskPose (1.1.0). Keep them in sync with PRETRAINED_URLS in
    pmpose/api.py.
    """
    if variant.startswith("PMPose-"):
        return f"{variant}-1.0.0.pth"
    if variant.startswith("MaskPose-"):
        return f"{variant}-1.1.0.pth"
    raise ValueError(f"Unknown pose variant: {variant}")


def _resolve_local_or_url(local_path: str, fallback_url: str, what: str) -> str:
    """Prefer ``local_path`` if it exists; otherwise return ``fallback_url``
    so BMP / mmengine auto-downloads it. Log which one we pick."""
    if os.path.isfile(local_path):
        _logger.info("  %s: local  %s", what, local_path)
        return local_path
    _logger.info(
        "  %s: missing locally (%s) → falling back to HF download  %s",
        what, local_path, fallback_url,
    )
    return fallback_url


def _write_patched_config(src_yaml_path: str,
                          det_config_abs: str,
                          det_checkpoint: str,
                          sam2_checkpoint: str,
                          sam2_config_abs: str) -> str:
    """Read BMP's packaged config YAML, swap the detector config +
    detector checkpoint + SAM2 config + SAM2 checkpoint paths for the
    ones we picked, write the result to a temp YAML file, return its
    path.

    We don't touch the ``pose_estimator`` section because our load flow
    constructs PMPose ourselves and injects it via ``pose_model=``, so
    BBoxMaskPose.__init__ skips its own pose_checkpoint lookup entirely.

    The ``*_abs`` swap-ins use absolute paths. BBoxMaskPose passes
    ``det_config`` STRAIGHT to ``init_detector`` (so an absolute path
    just works) and resolves ``sam2_config`` via
    ``os.path.join(BMP_ROOT, "bboxmaskpose", "sam2", sam2_config)`` —
    Python's os.path.join discards the prefix when the second arg is
    absolute, so feeding absolute paths bypasses the broken lookups
    into site-packages (which is missing the YAML/PY data files in
    the pip install).
    """
    import yaml

    with open(src_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["detector"]["det_config"] = det_config_abs
    cfg["detector"]["det_checkpoint"] = det_checkpoint
    cfg["sam2"]["sam2_checkpoint"] = sam2_checkpoint
    cfg["sam2"]["sam2_config"] = sam2_config_abs

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="bmp_cfg_", delete=False,
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


# Mapping: BMP config name → (our vendored BMP yaml, SAM2 config path
# the yaml references). Pre-computing here keeps the load path simple.
_BMP_TO_SAM2_CONFIG = {
    "bmp_v2": "sam-pose2seg/sam-pose2seg_hiera_b+.yaml",
    "bmp_D3": "samurai/sam2.1_hiera_b+.yaml",
    "bmp_J1": "samurai/sam2.1_hiera_b+.yaml",
}

# All three BMP configs point at the same RTMDet-ins-L detector config.
_BMP_DET_CONFIG = "mmdet/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py"


class LoadBMPNode:
    """Load BBoxMaskPose full pipeline (RTMDet + SAM2.1 + PMPose)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "Inference device. BMP is heavy (~4 GiB VRAM "
                            "peak for the three-model stack); CPU works "
                            "but is effectively unusable for video."
                        ),
                    },
                ),
                "config": (
                    _BMP_CONFIGS,
                    {
                        "default": "bmp_v2",
                        "tooltip": (
                            "BMP iteration config. bmp_v2 (default) is "
                            "2 iterations with 3 positive keypoint "
                            "prompts per SAM2 call — the paper's "
                            "recommended OCHuman setting. bmp_D3 and "
                            "bmp_J1 are legacy variants with slightly "
                            "different prompt counts."
                        ),
                    },
                ),
                "pose_variant": (
                    _PMPOSE_VARIANTS,
                    {
                        "default": "PMPose-b",
                        "tooltip": (
                            "Pose model weight variant. 'b' is the "
                            "balanced default. 'PMPose-*' gives full "
                            "presence/visibility/keypoint outputs; "
                            "'MaskPose-*' is the predecessor (no "
                            "presence/visibility probabilities but "
                            "faster). Small→huge: s/b/l/h."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("BMP",)
    RETURN_NAMES = ("bmp",)
    FUNCTION = "load"
    CATEGORY = "4dhumans"

    def load(self, device, config, pose_variant):
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch required") from e

        # Lazy-import the heavy packages so node registration stays cheap
        # even if BMP isn't installed yet — user sees the error only when
        # they try to load.
        try:
            from bboxmaskpose import BBoxMaskPose
            from pmpose import PMPose
            from pmpose.api import PRETRAINED_URLS as _PMPOSE_URLS
        except ImportError as e:
            raise ImportError(
                "BBoxMaskPose is not installed in this Python env. "
                "Install with:\n"
                "  pip install \"git+https://github.com/MiraPurkrabek/BBoxMaskPose.git\"\n"
                "BMP also requires mmcv/mmdet/mmpose/mmpretrain installed "
                "and compatible with your torch + CUDA. See project README."
            ) from e

        device_str = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "cpu"
        )
        if device == "cuda" and device_str == "cpu":
            _logger.warning(
                "LoadBMP: requested CUDA but torch.cuda.is_available() is "
                "False — falling back to CPU (BMP will be unusably slow)."
            )

        # ----------------------------------------------------------------
        # Resolve weights: prefer ``models/bmp/*`` locally, else HF URL.
        # ----------------------------------------------------------------
        _logger.info(
            "Loading BMP stack: config=%s pose_variant=%s device=%s",
            config, pose_variant, device_str,
        )
        _logger.info("Resolving BMP weights (local > HF URL):")

        det_ckpt = _resolve_local_or_url(
            os.path.join(BMP_MODELS_DIR, RTMDET_FILENAME),
            RTMDET_URL,
            "RTMDet-ins-L     ",
        )
        sam2_ckpt = _resolve_local_or_url(
            os.path.join(BMP_MODELS_DIR, SAM2_POSE2SEG_FILENAME),
            SAM2_POSE2SEG_URL,
            "SAM-pose2seg     ",
        )
        pose_filename = _pose_filename_from_variant(pose_variant)
        pose_ckpt = _resolve_local_or_url(
            os.path.join(BMP_MODELS_DIR, pose_filename),
            _PMPOSE_URLS[pose_variant],
            f"{pose_variant:17s}",
        )

        # ----------------------------------------------------------------
        # Step 1: PMPose. Two overrides layered on top of the stock
        # constructor, both needed because the pip package is shipped
        # without some of the assets upstream's default paths reference.
        #
        #   a) config_path=<our vendored .py> — the pmpose wheel ships
        #      api.py but not the mmpose config files; api's
        #      DEFAULT_CONFIGS still points at a nonexistent path under
        #      site-packages/mmpose/configs/. We vendor the 13 variant
        #      configs + _base_/default_runtime.py ourselves and feed
        #      the resolved path in directly.
        #
        #   b) PRETRAINED_URLS[variant] = <local .pth or HF url> —
        #      flows into init_pose_estimator as the checkpoint path.
        #      We restore the registry afterwards so a second LoadBMP
        #      with different resolution sees clean state.
        # ----------------------------------------------------------------
        variant_relpath = _PMPOSE_CONFIG_RELPATH.get(pose_variant)
        if variant_relpath is None:
            raise ValueError(
                f"Unknown pose variant '{pose_variant}'. "
                f"Known: {list(_PMPOSE_CONFIG_RELPATH)}"
            )
        pose_config_path = _resolve_pmpose_config(variant_relpath)
        if not os.path.isfile(pose_config_path):
            raise FileNotFoundError(
                f"PMPose config not found:\n  {pose_config_path}\n\n"
                f"Tried installed mmpose package + vendored fallback. "
                f"Make sure mmpose (or BMP's mmpose fork) is installed "
                f"with config data files included."
            )
        _logger.info("  PMPose config   : %s", pose_config_path)

        _original_pmpose_url = _PMPOSE_URLS.get(pose_variant)
        _PMPOSE_URLS[pose_variant] = pose_ckpt
        try:
            pose_model = PMPose(
                device=device_str,
                variant=pose_variant,
                from_pretrained=True,
                config_path=pose_config_path,   # override broken default
            )
        finally:
            # Restore the registry so a second LoadBMP call with a
            # different resolution doesn't see a stale override.
            if _original_pmpose_url is not None:
                _PMPOSE_URLS[pose_variant] = _original_pmpose_url

        # ----------------------------------------------------------------
        # Step 2: Patch BMP's YAML config with our resolved detector +
        # SAM2 paths, then build BBoxMaskPose off the patched file.
        #
        # Resolution order for each config: installed package first,
        # vendored bmp_configs/ as fallback. The fallback is kept around
        # for cases where the installed wheel was built without data
        # files — re-installing with our patched MANIFEST.in makes the
        # fallback unnecessary, but it doesn't hurt.
        # ----------------------------------------------------------------
        src_yaml = _resolve_bmp_yaml(f"configs/{config}.yaml")
        if not os.path.isfile(src_yaml):
            raise FileNotFoundError(
                f"BMP config not found:\n  {src_yaml}\n\n"
                f"Tried installed bboxmaskpose package + vendored "
                f"fallback. Make sure bboxmaskpose is installed with "
                f"config data files (re-build with the MANIFEST.in "
                f"patch if needed)."
            )

        sam2_config_rel = _BMP_TO_SAM2_CONFIG.get(config)
        if sam2_config_rel is None:
            raise ValueError(
                f"No SAM2 config mapping for BMP config '{config}'. "
                f"Known: {list(_BMP_TO_SAM2_CONFIG)}"
            )
        sam2_config_abs = _resolve_bmp_yaml(f"sam2/configs/{sam2_config_rel}")
        if not os.path.isfile(sam2_config_abs):
            raise FileNotFoundError(
                f"SAM2 config not found:\n  {sam2_config_abs}\n\n"
                f"Tried installed bboxmaskpose package + vendored fallback."
            )
        _logger.info("  SAM2 config     : %s", sam2_config_abs)

        # Resolve RTMDet detector config. Its ``_base_`` chain
        # (rtmdet_l_8xb32-300e_coco.py, rtmdet_tta.py, plus 3 _base_/
        # entries) is shipped alongside — mmengine resolves _base_
        # relative to the config file's dir, so as long as the layout
        # is preserved the chain walks naturally.
        det_config_abs = _resolve_mmdet_config(_BMP_DET_CONFIG)
        if not os.path.isfile(det_config_abs):
            raise FileNotFoundError(
                f"RTMDet config not found:\n  {det_config_abs}\n\n"
                f"Tried installed mmdet (.mim/configs and configs paths) "
                f"+ vendored bmp_configs/mmdet/ fallback."
            )
        _logger.info("  RTMDet config   : %s", det_config_abs)

        patched_yaml = _write_patched_config(
            src_yaml_path=src_yaml,
            det_config_abs=det_config_abs,
            det_checkpoint=det_ckpt,
            sam2_checkpoint=sam2_ckpt,
            sam2_config_abs=sam2_config_abs,
        )
        _logger.info("BMP config patched → %s", patched_yaml)

        try:
            bmp_model = BBoxMaskPose(
                config=config,                 # informational (logged)
                config_path=patched_yaml,      # what actually drives load
                device=device_str,
                pose_model=pose_model,         # our already-loaded PMPose
            )
        finally:
            # We can delete the temp config now — BBoxMaskPose has
            # finished parsing it.
            try:
                os.unlink(patched_yaml)
            except OSError:
                pass

        _logger.info(
            "BMP ready. Run BMPInstanceSegmentation downstream to do "
            "per-frame inference + cross-frame tracking."
        )

        return ({
            "bmp":           bmp_model,
            "pose_model":    pose_model,
            "device":        device_str,
            "config":        config,
            "pose_variant":  pose_variant,
            "det_ckpt":      det_ckpt,
            "sam2_ckpt":     sam2_ckpt,
            "pose_ckpt":     pose_ckpt,
        },)
