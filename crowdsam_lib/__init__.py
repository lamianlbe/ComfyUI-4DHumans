"""
Vendored CrowdSAM runtime (Cai et al., ECCV 2024; MIT licence).

Source: https://github.com/FelixCaae/CrowdSAM

Three sibling packages are shipped here so the stock CrowdSAM code
paths import cleanly without any pip install:

  crowdsam_lib/crowdsam/            — CrowdSAM wrapper (model.py,
                                       utils.py, data.py). A single
                                       top-level `from loguru import
                                       logger` in model.py was swapped
                                       for stdlib logging so we don't
                                       need loguru at runtime.
  crowdsam_lib/segment_anything_cs/ — Modified SAM fork with DINOv2
                                       prompt conditioning + the
                                       trained adapter head.
  crowdsam_lib/dinov2/              — DINOv2 repo (hubconf.py plus the
                                       `dinov2/` package). We load it
                                       via ``torch.hub.load(source=
                                       'local')`` so the original
                                       CrowdSAM config path works
                                       unchanged. Trimmed to
                                       hubconf.py + LICENSE + the
                                       dinov2/ package (~1.2 MB).

``ensure_lib_importable()`` prepends this directory to ``sys.path`` so
``from crowdsam.model import CrowdSAM`` and ``from segment_anything_cs
import ...`` resolve. DINOv2 is loaded separately via torch.hub so it
doesn't need to be on the Python path.
"""

import os
import sys

LIB_PATH = os.path.dirname(os.path.abspath(__file__))
DINOV2_REPO_PATH = os.path.join(LIB_PATH, "dinov2")


def ensure_lib_importable():
    """Prepend our vendored packages to ``sys.path`` if not already there.

    Safe to call repeatedly; idempotent.
    """
    if LIB_PATH not in sys.path:
        sys.path.insert(0, LIB_PATH)
