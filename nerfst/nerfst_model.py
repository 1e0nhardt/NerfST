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
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfst.nerfst_arf_field import ArfTcnnField

@dataclass
class NerfSTModelConfig(NerfactoModelConfig):
    """Configuration for the NerfSTModel."""

    _target: Type = field(default_factory=lambda: NerfSTModel)
    use_l1: bool = False
    """Whether to use L1 loss"""
    use_lpips: bool = False
    """Whether to use LPIPS loss"""
    fix_density: bool = False
    """ARF need fix density"""
    no_viewdep: bool = False
    """ARF discard view dependent modeling"""


class NerfSTModel(NerfactoModel):
    """Model for NerfST."""

    config: NerfSTModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = ArfTcnnField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            no_viewdep=self.config.no_viewdep
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        if self.config.fix_density:
            fields_params = []
            for name, params in self.field.named_parameters():
                if name == 'mlp_base.params':
                    params.requires_grad_(False)
                fields_params.append(params)
            param_groups["fields"] = fields_params
        else:
            param_groups["fields"] = list(self.field.parameters())
        
        return param_groups

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.config.use_lpips:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict
