# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model for NerfSTs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type
from nerfstudio.cameras.rays import RayBundle

import torch
from torch.nn import Parameter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.tensorf import TensoRFModel, TensoRFModelConfig

@dataclass
class NerfSTTensoRFModelConfig(TensoRFModelConfig):
    """Configuration for the NerfSTModel."""

    _target: Type = field(default_factory=lambda: NerfSTTensoRFModel)
    use_l1: bool = False
    """Whether to use L1 loss"""
    fix_density: bool = False
    """ARF need fix density"""
    no_viewdep: bool = False
    """ARF discard view dependent modeling"""


class NerfSTTensoRFModel(TensoRFModel):
    """Model for NerfST."""

    config: NerfSTTensoRFModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        
        # self.lpips = LearnedPerceptualImagePatchSimilarity() # 自带
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.config.fix_density:
            density_mlp_paramas = list(x.requires_grad_(False) for x in self.field.mlp_head.parameters())
        else:
            density_mlp_paramas = list(self.field.mlp_head.parameters())
        param_groups["fields"] = (
            density_mlp_paramas
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
            self.field.density_encoding.parameters()
        )

        return param_groups