# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Trimmed hubconf: CrowdSAM only needs the vanilla DINOv2 backbones
# (dinov2_vit{s,b,l,g}14). The upstream hubconf also imported:
#
#   - classifier heads (depend on extra pretrained LC weights)
#   - depther heads (depend on NYU / KITTI checkpoints)
#   - cell_dino / xray_dino variants (extra checkpoints, missing
#     __init__.py in the subpackage dirs)
#   - dinotxt (depends on open_clip)
#
# None of those are needed for our inference path, and importing them
# would fail without the extra checkpoint files / optional deps. Keep
# just the backbone factories.

from dinov2.hub.backbones import (
    dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14,
    dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg,
)

dependencies = ["torch"]
