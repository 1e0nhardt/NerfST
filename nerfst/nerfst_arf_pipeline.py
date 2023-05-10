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
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

from nerfst.nerfst_datamanager import (
    NerfSTDataManagerConfig,
)
from nerfst.style_transfer.nn_loss import NNLoss
from rich.console import Console
from nerfst.util import iterate_eternally

CONSOLE = Console(width=120)

@dataclass
class NerfSTArfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NerfSTArfPipeline)
    """target class to instantiate"""
    datamanager: NerfSTDataManagerConfig = NerfSTDataManagerConfig()
    """specifies the datamanager config"""


class NerfSTArfPipeline(VanillaPipeline):
    """NerfSTArf pipeline"""

    config: NerfSTArfPipelineConfig

    def __init__(
        self,
        config: NerfSTArfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # keep track of spot in dataset
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            #! 顺序取改为随机取
            self.train_indices_order = iterate_eternally(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))
        self.nn_loss_fn = NNLoss(device=device)
        self.style_image = self.datamanager.style_image.unsqueeze(dim=0).permute(0, 3, 1, 2).to(device)

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
