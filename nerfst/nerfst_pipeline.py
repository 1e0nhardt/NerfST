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

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

from nerfst.nerfst_datamanager import (
    NerfSTDataManagerConfig,
)
from nerfst.style_transfer.pama import pama_infer_one_image
from rich.progress import Console, track

CONSOLE = Console(width=120)

@dataclass
class PamaConfig:
    """Pama Model Arguments"""

    checkpoints: str = "pretrained/pama/original"
    """pretrained model path. Include encoder.pth, decoder.pth, and 3 PAMA*.pth"""
    pretrained: bool = True
    """use pretrained model"""
    requires_grad: bool = False
    """whether to finetune"""
    training: bool = False
    """we only need infer, so always set this to False"""


@dataclass
class NerfSTPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NerfSTPipeline)
    """target class to instantiate"""
    datamanager: NerfSTDataManagerConfig = NerfSTDataManagerConfig()
    """specifies the datamanager config"""
    edit_rate: int = 10
    """how many NeRF steps before image edit"""
    edit_count: int = 1
    """how many images to edit per NeRF step"""
    pama_config: PamaConfig = field(default_factory=PamaConfig)


class NerfSTPipeline(VanillaPipeline):
    """NerfST pipeline"""

    config: NerfSTPipelineConfig

    def __init__(
        self,
        config: NerfSTPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # keep track of spot in dataset
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))
        
        # style_transfered_batch = []
        # original_images = self.datamanager.original_image_batch["image"]
        # style_image = self.datamanager.style_image.unsqueeze(dim=0).permute(0, 3, 1, 2)

        # CONSOLE.print(self.datamanager.image_batch['image'].shape)
        # CONSOLE.print(self.datamanager.image_batch['image'].device)
        # for i in track(range(original_images.shape[0])):
        #     style_transfered_image = pama_infer_one_image(original_images[i].unsqueeze(0).permute(0, 3, 1, 2), style_image, self.config.pama_config)
        #     style_transfered_batch.append(style_transfered_image.permute(0, 2, 3, 1).contiguous().cpu())
        # self.datamanager.original_image_batch['image'] = torch.concat(style_transfered_batch, dim=0).contiguous()
        # CONSOLE.print(self.datamanager.original_image_batch['image'].shape)
        # self.datamanager.image_batch['image'] = self.datamanager.original_image_batch['image']

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        ray_bundle, batch = self.datamanager.next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        # edit an image every ``edit_rate`` steps
        # if (step % self.config.edit_rate == 0):
        if (step == 30010):
            CONSOLE.print("Start Transfer")
            CONSOLE.print(len(self.datamanager.train_dataparser_outputs.image_filenames))

            # edit ``edit_count`` images in a row
            # for i in range(self.config.edit_count):
            for i in track(range(len(self.datamanager.train_dataparser_outputs.image_filenames))):

                # iterate through "spot in dataset"
                current_spot = next(self.train_indices_order)
                
                # get original image from dataset
                original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
                # generate current index in datamanger
                current_index = self.datamanager.image_batch["image_idx"][current_spot]

                # get current camera, include camera transforms from original optimizer
                camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
                current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
                current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

                # get current render of nerf
                original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                style_image = self.datamanager.style_image.unsqueeze(dim=0).permute(0, 3, 1, 2)


                # delete to free up memory
                del camera_outputs
                del current_camera
                del current_ray_bundle
                del camera_transforms
                torch.cuda.empty_cache()

                style_transferred_image = pama_infer_one_image(original_image, style_image, self.config.pama_config)

                # resize to original image size (often not necessary)
                if (style_transferred_image.size() != rendered_image.size()):
                    style_transferred_image = torch.nn.functional.interpolate(style_transferred_image, size=rendered_image.size()[2:], mode='bilinear')

                # write edited image to dataloader
                self.datamanager.image_batch["image"][current_spot] = style_transferred_image.squeeze().permute(1,2,0)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
