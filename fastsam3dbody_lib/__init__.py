"""
Fast SAM 3D Body vendored library.

Contains vendored copies of:

- ``sam_3d_body/`` — Meta's SAM 3D Body + USC's Fast SAM 3D Body
  acceleration framework (https://github.com/facebookresearch/sam-3d-body,
  https://yangtiming.github.io/Fast-SAM-3D-Body-Page/).
- ``mhr2smpl/`` — direct feed-forward mapper from MHR mesh to SMPL
  parameters.  We vendor only the minimum Python modules needed for
  inference: ``mhr2smpl/multi_view/{infer_multiview.py, multiview_net.py}``
  and ``mhr2smpl/smooth/smoother_net.py``.  No experiments, checkpoints,
  or demo GIFs.
- ``tools/`` — detector / segmentor / FOV estimator wrappers.

Upstream uses absolute imports like ``from sam_3d_body.models.x`` and
``from tools.vis_utils``, plus ``mhr2smpl`` further adds its own
``multi_view/`` and ``smooth/`` folders to ``sys.path`` at runtime (see
the top of ``mhr2smpl/multi_view/infer_multiview.py``).  We mirror the
same setup so vendored imports keep working without code changes.

Usage inside our nodes::

    from fastsam3dbody_lib import ensure_lib_importable
    ensure_lib_importable()
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from infer_multiview import MHR2SMPLMultiView

Licence: the vendored code is released by Meta / USC under a
**non-commercial research licence**.  Consult each upstream repository
for the full text before deploying commercially.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MHR2SMPL_MULTIVIEW = os.path.join(_THIS_DIR, "mhr2smpl", "multi_view")
_MHR2SMPL_SMOOTH = os.path.join(_THIS_DIR, "mhr2smpl", "smooth")


def ensure_lib_importable():
    """Prepend vendored paths so ``import sam_3d_body`` etc. work.

    Safe to call multiple times; each path is only added once.
    """
    for p in (_THIS_DIR, _MHR2SMPL_MULTIVIEW, _MHR2SMPL_SMOOTH):
        if p not in sys.path:
            sys.path.insert(0, p)
