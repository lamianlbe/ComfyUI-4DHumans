"""
Vendored WiLoR runtime — copied from github.com/rolpotamias/WiLoR.

WiLoR's upstream repo isn't pip-installable (no setup.py / pyproject.toml).
We vendor the core ``wilor/`` package here so users don't have to clone
the upstream repo and configure PYTHONPATH manually.

Layout under wilor_lib/:
    wilor/
    ├── configs/
    ├── datasets/
    ├── models/       (WiLoR class, MANO wrapper, ViT backbone, head)
    └── utils/        (renderer, geometry, etc.)

Total size: ~250 KB. No weights, no demo images, no training scripts —
only the inference code path.

Required runtime dependencies (must be installed in the ComfyUI venv):
    torch, torchvision     ← already required by ComfyUI
    pytorch-lightning      ← WiLoR.load_from_checkpoint uses Lightning
    smplx                  ← MANO model
    timm                   ← ViT backbone layers
    yacs                   ← config nodes
    omegaconf / hydra      ← used by some utils (transient)
    scikit-image           ← image transforms
    rich                   ← logging utilities
    ultralytics            ← YOLO hand detector

The user is also expected to drop:
    models/wilor/wilor_final.ckpt           (~2 GB)
    models/wilor/detector.pt                (~50 MB)
    models/wilor/mano_data/MANO_RIGHT.pkl   (registration-gated)

Call ``ensure_lib_importable()`` once before importing ``wilor.*`` —
the LoadWiLoR node does this for you.
"""

import os
import sys

LIB_PATH = os.path.dirname(os.path.abspath(__file__))


def ensure_lib_importable() -> None:
    """Prepend ``wilor_lib/`` to ``sys.path`` so ``import wilor.models``
    resolves to the vendored package. Idempotent.
    """
    if LIB_PATH not in sys.path:
        sys.path.insert(0, LIB_PATH)
