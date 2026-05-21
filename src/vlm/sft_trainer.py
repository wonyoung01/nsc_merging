"""
Construct custom SFTTrainer for fine-tuning LLaVA with LoRA and tracks kernel projection rate.
"""

import logging
import math
import time

from models import HookManager, ProjectorCluster, get_projection_writer_hook
from torch.utils.data import Dataset
from transformers.trainer_utils import speed_metrics
from trl.trainer.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


class KernelSFTTrainer(SFTTrainer):
    def __init__(self, use_wandb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector_cluster = ProjectorCluster(self.model)
        self.hook_manager = HookManager()
        self.use_wandb = use_wandb
        self._device = next(self.model.parameters()).device

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # FIXME: Currently activating kernel projection tracking only during evaluation.
        self.projector_cluster.compute_kernel_projection(self.model.active_adapter)
        self.projector_cluster.send_to_device(self._device, self.model.active_adapter)
        # Register hook to log kernel projection
        for key, projector in self.projector_cluster.kernel_projectors[
            self.model.active_adapter
        ].items():
            try:
                module = self.model.get_submodule(key + ".lora_A." + self.model.active_adapter)
            except AttributeError as e:
                logger.error(f"Error accessing layer {key}: {e}")
                logger.error(f"Layer {key} not found in model.")
                continue
            self.hook_manager.add_hook(
                module,
                get_projection_writer_hook(
                    self.args.output_dir, projector, use_wandb=self.use_wandb
                ),
            )
        for key, projector in self.projector_cluster.image_projectors[
            self.model.active_adapter
        ].items():
            try:
                module = self.model.get_submodule(key + ".base_layer")
                # module = model.vit.get_submodule(key)
            except AttributeError as e:
                logger.error(f"Error accessing layer {key}: {e}")
                logger.error(f"Layer {key} not found in model.")
                continue
            self.hook_manager.add_hook(
                module,
                get_projection_writer_hook(
                    self.args.output_dir, projector, use_wandb=self.use_wandb
                ),
            )

        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset  # type: ignore
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,  # type: ignore
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:  # type: ignore
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]  # type: ignore
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:  # type: ignore
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]  # type: ignore
        output.metrics.update(  # type: ignore
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),  # type: ignore
            )
        )

        self.log(output.metrics)  # type: ignore

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        self.hook_manager.clear()

        return output.metrics  # type: ignore
