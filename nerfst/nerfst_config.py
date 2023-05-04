from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)

from nerfst.nerfst_datamanager import NerfSTDataManagerConfig
from nerfst.nerfst_model import NerfSTModelConfig
from nerfst.nerfst_pipeline import NerfSTPipelineConfig, PamaConfig
from nerfst.nerfst_arf_pipeline import NerfSTArfPipelineConfig
from nerfst.nerfst_trainer import NeRFSTArfTrainerConfig

arf_method = MethodSpecification(
    config=NeRFSTArfTrainerConfig(
        method_name="arfst",  # ns-train method_name
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=50,
        save_only_latest_checkpoint=False,
        max_num_iterations=2000,
        pipeline=NerfSTArfPipelineConfig(
            datamanager=NerfSTDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(
                        lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(
                        lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=NerfSTModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                fix_density=True,
                no_viewdep=True
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=2000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=2000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="ARF Style Transfer"
)


nerfst_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfst",  # ns-train method_name
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=15000,
        pipeline=NerfSTPipelineConfig(
            datamanager=NerfSTDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(
                        lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(
                        lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=NerfSTModelConfig(eval_num_rays_per_chunk=1 << 15),
            pama_config=PamaConfig(
                checkpoints="pretrained/pama/original",
                pretrained=True,
                requires_grad=False, 
                training=False
            )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerf+Style Transfer"
)