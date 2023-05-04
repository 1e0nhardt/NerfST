from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import functools
import gc

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import track
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.utils import profiler
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


TRAIN_INTERATION_OUTPUT = Tuple[  # pylint: disable=invalid-name
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]
TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name
CONSOLE = Console(width=120)


@dataclass
class NeRFSTArfTrainerConfig(TrainerConfig):
    """Configuration for the NeRFSTTrainer."""

    _target: Type = field(default_factory=lambda: NeRFSTArfTrainer)
    ray_batch_size: int = 8192
    """梯度回传时每一块的大小"""   


class NeRFSTArfTrainer(Trainer):
    """Trainer for NeRFST"""

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """重置一下训练步数"""
        super().setup(test_mode)
        CONSOLE.print('*'*24)
        CONSOLE.print(f'Loaded model trained {self._start_step} steps')
        CONSOLE.print('*'*24)
        self._start_step = 0

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a one full image. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]

        # 随机选择一张训练图像对应的视角
        current_spot = next(self.pipeline.train_indices_order)
        # get original image from dataset
        original_image = self.pipeline.datamanager.original_image_batch["image"][current_spot].to(self.pipeline.device)
        # generate current index in datamanger
        current_index = self.pipeline.datamanager.image_batch["image_idx"][current_spot]
        # get current camera, include camera transforms from original optimizer
        camera_transforms = self.pipeline.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
        current_camera = self.pipeline.datamanager.train_dataparser_outputs.cameras[current_index].to(self.pipeline.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)
        
        # get current render of nerf
        rgb_gt = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)

        # delete to free up memory
        del current_camera
        del camera_transforms

        gc.collect()
        torch.cuda.empty_cache()

        def compute_image_loss():
            with torch.no_grad():
                camera_outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                rgb_pred = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)

            rgb_pred.requires_grad_(True)

            w_variance = torch.mean(torch.pow(rgb_pred[:, :, :, :-1] - rgb_pred[:, :, :, 1:], 2))
            h_variance = torch.mean(torch.pow(rgb_pred[:, :, :-1, :] - rgb_pred[:, :, 1:, :], 2))
            img_tv_loss = 1.0 * (h_variance + w_variance) / 2.0

            # downscale to decrease the demand of GPU memory
            nn_loss, _, content_loss = self.pipeline.nn_loss_fn(
                F.interpolate(
                    rgb_pred,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                self.pipeline.style_image,
                loss_names=["nn_loss", "content_loss"],
                contents=F.interpolate(
                    rgb_gt,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
            )

            content_loss = content_loss * 0.005  # was using 5e-3

            loss = nn_loss + content_loss + img_tv_loss
            loss.backward()

            rgb_pred_grad = rgb_pred.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
            rgb_pred = rgb_pred.squeeze(0).permute(1, 2, 0).contiguous().clone().detach()

            return rgb_pred_grad, loss, nn_loss, content_loss, img_tv_loss

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            rgb_pred_grad, loss, nn_loss, content_loss, img_tv_loss = compute_image_loss()

        loss_dict = {
            "nn_loss": nn_loss,
            "content_loss": content_loss,
            "img_tv_loss": img_tv_loss
        }

        num_rays_per_chunk = self.config.ray_batch_size
        num_rays = len(current_ray_bundle)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                ray_bundle = current_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                rgb_pred = self.pipeline.model.forward(ray_bundle=ray_bundle)["rgb"]
            # CONSOLE.print(rgb_pred.grad_fn)
            # CONSOLE.print(rgb_pred.shape)
            # CONSOLE.print(rgb_pred_grad.shape)
            # 相机参数优化的问题? Yes
            if self.pipeline.datamanager.config.camera_optimizer.mode == "off":
                self.grad_scaler.scale(rgb_pred).backward(rgb_pred_grad[start_idx:end_idx])
            else:
                # CONSOLE.print("using camera optimizer!!!")
                self.grad_scaler.scale(rgb_pred).backward(rgb_pred_grad[start_idx:end_idx], retain_graph=True)

        # self.grad_scaler.scale(rgb_pred).backward(rgb_pred_grad)  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        metrics_dict = {} 
        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad

            metrics_dict["Gradients/Total"] = total_grad

        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict
