import os

import ipdb  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from peft.utils import ModulesToSaveWrapper, _get_submodules
from transformers import (
    AutoModel,
    CLIPVisionModelWithProjection,
    LlavaForConditionalGeneration,
    ViTModel,
)
from utils import get_base_model, get_criterion, load_lora_model

from src.vlm.utils import remove_modules_to_save

from .kernel_projection import HookManager, LossCluster, ProjectorCluster, get_projection_loss_hook
from .merging_utils import DiagonalLinear, LinearTrainable, LoraModelTrainableMerging

MERGE_TYPES = ["weight", "msu", "msu_neg", "svd"]


class SimpleMergeModel(nn.Module):
    def __init__(
        self,
        cfgs,
        model_dirs,
        lora_dicts,
        preference,
        merge_type="weight",
        coeff_type="prob",
        scaling_coeff=1.0,
        **kwargs,
    ):
        """
        SimpleMergeModel is a model that merges multiple LoRA adapters into a single model.
        Args:
            model1 (nn.Module): The first model to merge.
            model2 (nn.Module): The second model to merge.
            lora_dict (dict): A dictionary mapping task names to LoRA adapter paths.
            preference (torch.Tensor): A tensor representing the preference for each task.
            merge_type (str): The type of merging to perform. Options are:
                - "weight": Merge with a single coefficient per adapter.
                - "msu": Merge with multiple coefficients per adapter based on rank.
                - "svd": Merge with coefficients based on SVD decomposition.
        Raises:
            AssertionError: If the merge_type is not one of the supported types.
        """
        super().__init__()
        assert merge_type in MERGE_TYPES, f"Unknown merge type: {merge_type}"
        self.prefix = "lora_"
        self.backbone_name = "merged_adapter"
        self.scaled_name = "scaled_adapter"
        self.cfgs = cfgs
        if cfgs[0].model.id.lower().startswith("llava"):
            # Do not need to call lora_model at this stage for llava.
            _base_model = get_base_model(cfgs[0])
            _models = [_base_model for _ in cfgs]
        else:
            _models = [
                load_lora_model(
                    get_base_model(cfg),
                    lora_dicts[f"{cfg.data.name}-{cfg.task.name}"],
                    os.path.join(model_dir, cfg.get("decoder_save_name", "")),
                )
                for cfg, model_dir in zip(cfgs, model_dirs)
            ]
        self.lora_dicts = lora_dicts
        self.scaling_coeff = scaling_coeff
        self.preference = preference
        self.merge_type = merge_type
        self.coeff_type = coeff_type

        self.model_id = _model_id = _models[0].base_id
        self.isinstance_modules_to_save_mm_proj = "llava" in _model_id.lower()
        # TODO: Fix this restriction, we should allow merging models with different types.
        assert all(model.base_id == _model_id for model in _models), (
            "Models must have the same pretrained model ID for merging."
        )
        _vit = self._get_multi_lora_model(_model_id, lora_dicts)

        # Compute adapter-wise number of coefficients and ranks.
        self.adapter_wise_num_coeffs = self._compute_adapter_wise_num_coeffs(_vit.peft_config)
        self.adapter_wise_rank = self._compute_adapter_wise_rank(_vit.peft_config)

        _vit_wrapper = LoraModelTrainableMerging(_vit)
        _additional_mm_wrapper = []
        # Merge the LoRA adapters into a single model and share it with model1 and model2.
        self.vit = self.merge(_vit_wrapper)
        if self.isinstance_modules_to_save_mm_proj:
            for mod in [
                _vit.multi_modal_projector,
                _vit.multi_modal_projector.linear_1,
                _vit.multi_modal_projector.linear_2,
            ]:
                if isinstance(mod, PeftModel):
                    _additional_mm_wrapper.append(LoraModelTrainableMerging(mod))
        for _mm_wrapper in _additional_mm_wrapper:
            self.merge(_mm_wrapper, merge_type="weight")

        # Make scaled adapter trainable.
        for _mm_wrapper in _additional_mm_wrapper:
            _mm_wrapper.add_trainable_adapter(
                self.backbone_name,
                self.scaled_name,
                self._compute_adapter_wise_num_coeffs(
                    _mm_wrapper.lora_model.peft_config, merge_type="weight"
                ),
                self._compute_adapter_wise_rank(_mm_wrapper.lora_model.peft_config),
                self.scaling_coeff,
                "weight",
                self.coeff_type,
            )
        _vit_wrapper.add_trainable_adapter(
            self.backbone_name,
            self.scaled_name,
            self.adapter_wise_num_coeffs,
            self.adapter_wise_rank,
            self.scaling_coeff,
            self.merge_type,
            self.coeff_type,
        )

        for model in _models:
            model.vit = self.vit  # type: ignore
        self.models = nn.ModuleList(_models)

        # Merged model does not require gradient updates for the parameters except for the
        # coefficients.
        self.requires_grad_(False)
        # Set the merged model as the backbone for the merged model.
        # Before, set gradients to True for the coefficients.
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    sub_target.requires_grad_(True)  # type: ignore

    def _get_initialization(self):
        """
        Get the initialization method for the coefficients based on the merge type.
        """
        if self.merge_type == "msu_neg":
            return "stair"
        return "uniform"

    def _compute_adapter_wise_rank(self, lora_configs):
        """
        Compute the task rank based on the LoRA configurations and the merge type.
        This is used to determine how many coefficients are used to merge for each adapter.
        Assumptions:
            - The backbone self.vit has LoRA adapters loaded.
        Details:
            - For "weight", each adapter layer has 1 coefficient.
            - For "msu", each adapter layer has a number of coefficients equal to its rank.
            - For "msu_neg", each adapter layer has a number of coefficients equal to its rank.
            - For "svd", each adapter layer has a number of coefficients equal to \
            its rank * num_adapters.
        """
        # rank of each adapter
        task_rank = [lora_configs[key].r for key in self.lora_dicts.keys()]
        return task_rank

    def _compute_adapter_wise_num_coeffs(self, lora_configs, merge_type=None):
        """
        Compute the task rank based on the LoRA configurations and the merge type.
        This is used to determine how many coefficients are used to merge for each adapter.
        Assumptions:
            - The backbone self.vit has LoRA adapters loaded.
        Details:
            - For "weight", each adapter layer has 1 coefficient.
            - For "msu", each adapter layer has a number of coefficients equal to its rank.
            - For "msu_neg", each adapter layer has 2 coefficients \
            (one for positive and one for negative) with its MSUs.
            - For "svd", each adapter layer has a number of coefficients equal to \
            its rank * num_adapters.
        """
        if merge_type is None:
            merge_type = self.merge_type
        # rank of each adapter
        task_rank = self._compute_adapter_wise_rank(lora_configs)
        if merge_type == "weight":
            task_rank = [1 for _ in task_rank]
        elif merge_type == "svd":
            total_rank = sum(task_rank)
            task_rank = [total_rank for _ in task_rank]
        elif merge_type == "msu_neg":
            task_rank = [2 * r for r in task_rank]
        return task_rank

    def _get_multi_lora_model(self, model_type, lora_dict):
        base_model = get_vit_from_id(model_type)
        first_task = list(lora_dict.keys())[0]
        first_lora_path = lora_dict[first_task]
        if "unseen" in self.cfgs[0] and self.cfgs[0].unseen:
            raise ValueError("Please load the LoRA adapters directly for the first task.")
        model = PeftModel.from_pretrained(base_model, first_lora_path, first_task)
        for task_name, lora_path, cfg in zip(
            list(lora_dict.keys())[1:], list(lora_dict.values())[1:], self.cfgs[1:]
        ):
            if "unseen" in cfg and cfg.unseen:
                continue
            model.load_adapter(lora_path, adapter_name=task_name)

        if self.isinstance_modules_to_save_mm_proj:
            model = remove_modules_to_save(model)
        for lora_name in model.peft_config.keys():
            modules_to_save_status = model.peft_config[lora_name].modules_to_save
            if modules_to_save_status is not None:
                model.peft_config[lora_name].modules_to_save = None  # type: ignore
        return model

    def merge(self, _vit_wrapper, merge_type=None):
        """
        Merge the LoRA weights. It will be scalded with the coefficients.
        Raises:
            ValueError: If the merge type is unknown.
        """
        if merge_type is None:
            merge_type = self.merge_type
        merge_method = "concat_adapters_" + merge_type
        merge_fn = getattr(_vit_wrapper, merge_method, None)
        if merge_fn is None:
            raise ValueError(
                f"Merge type '{merge_type}' is not supported by LoraModelTrainableMerging."
            )
        merge_fn(
            list(self.lora_dicts.keys()),
            adapter_name=self.backbone_name,
        )
        _vit_wrapper.set_adapter(self.backbone_name)
        return _vit_wrapper.lora_model

    def _forward_helper(self, model, batch, result_accum, **kwargs):
        if isinstance(self.vit.base_model.model, LlavaForConditionalGeneration):
            # NOTE:Differntiable decode for Llava handles only single data point.
            out = model.differentiable_decode(
                batch,
                **kwargs,
            )
            if kwargs.get("backprop_entropy", False):
                out = out[1]
        elif isinstance(batch, dict):
            out = model(**batch)
        else:
            out = model(batch)
        result_accum.append(out)

    def forward(self, inputs_list, **kwargs):
        result = []
        for idx, model in enumerate(self.models):
            batch = inputs_list[idx]
            self._forward_helper(model, batch, result, **kwargs)

        return result

    def coeffs(self):
        """
        Get the coefficients for the merged model.
        Returns:
            dict: A dictionary mapping adapter names to their coefficients.
        """
        coeffs = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    coeffs[key] = sub_target._logit_to_coeff().detach().cpu().numpy()
        return coeffs

    def coeffs_without_detach(self):
        """
        Get the coefficients for the merged model.
        Returns:
            dict: A dictionary mapping adapter names to their coefficients.
        """
        coeffs = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    coeffs[key] = sub_target._logit_to_coeff()
        return coeffs

    def probs(self):
        """
        Get the coefficients for the merged model.
        Returns:
            dict: A dictionary mapping adapter names to their coefficients.
        """
        probs = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                sub_target = target.lora_A[self.scaled_name][1]  # type: ignore
                if isinstance(sub_target, DiagonalLinear):
                    probs[key] = (
                        sub_target._logit_to_prob()
                    )  # prob is used for the loss => no detach
        return probs

    def reg_loss(self, lambda_reg=10.0):
        """
        # Regularization loss for the coefficients
        Args:
            lambda_reg (float): Regularization strength.
        Returns:
            torch.Tensor: Regularization loss.
        """

        def compute_erank(matrix):
            """Compute the effective rank of a matrix."""
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            s = s / (s.sum() + 1e-10)
            entropy = -(s * torch.log(s + 1e-10)).sum()
            erank = torch.exp(entropy)
            return erank

        loss_list = []
        key_list = [key for key, _ in self.vit.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LinearTrainable):
                delta = target.get_delta_weight()
                rank = target.get_rank()
                erank = compute_erank(delta)
                loss_list.append(1.0 - (erank / rank))
        loss = torch.stack(loss_list).mean()
        return lambda_reg * loss


class KernelMergeModel(SimpleMergeModel):
    """
    KernelMergeModel is a model that merges multiple LoRA adapters into a single model
    using kernel-based merging.
    This model is used to merge the coefficients of the LoRA adapters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector_cluster = ProjectorCluster(
            self._get_multi_lora_model(self.model_id, self.lora_dicts)
        )
        self.hook_manager = HookManager()
        self.loss_cluster = LossCluster()

    def activate_projectors(
        self,
        curr_adapter,
        block_candidates="24,25,26,27,28,29,30,31",
        layer_candidates="v_proj",
    ):
        """
        Activate the projectors for the given adapter name.
        This method is used to activate the kernel projectors and image projectors
        for the given adapter name.
        """
        block_candidates = block_candidates.split(",")
        block_choices = [*block_candidates]
        layer_candidates = layer_candidates.split(",")
        layer_choices = [*layer_candidates]
        self.hook_manager.clear()
        # Activate the kernel projector for the current adapter
        if curr_adapter in self.projector_cluster.kernel_projectors:
            activate_list = [curr_adapter]
        else:
            activate_list = list(self.projector_cluster.kernel_projectors.keys())[:-2]
        for adapter_name in activate_list:
            for key, projector in self.projector_cluster.kernel_projectors[adapter_name].items():
                try:
                    module = self.vit.get_submodule(
                        key + ".lora_A." + self.vit.active_adapter + ".0"
                    )
                except AttributeError:
                    print(f"Module {key} not found in the model.")
                if any(choice in key for choice in block_choices) and any(
                    choice in key for choice in layer_choices
                ):
                    self.hook_manager.add_hook(
                        module,
                        get_projection_loss_hook(
                            projector,
                            key,
                            curr_adapter,
                            self.loss_cluster,
                            use_wandb=True,
                            target_adapter=adapter_name,
                        ),
                    )

    def forward(self, inputs_list, **kwargs):
        result = []
        loss_cluster = kwargs.get("loss_cluster", None)
        use_ada = kwargs.get("use_ada", False)
        for idx, (cfg, model) in enumerate(zip(self.cfgs, self.models)):
            adapter_name = f"{cfg.data.name}-{cfg.task.name}"
            # Activate the projectors for the current adapter
            batch = inputs_list[idx]
            self.activate_projectors(adapter_name)
            self._forward_helper(model, batch, result, **kwargs)

            if loss_cluster is not None:
                each_loss = loss_cluster.return_loss() / len(inputs_list)
                if hasattr(each_loss, "grad_fn") and each_loss.grad_fn is not None:
                    each_loss.backward()
                loss_cluster.clear()

        self.hook_manager.clear()
        return result


class LinearCombLoss(nn.Module):
    """Linear combination of per-task losses weighted by preference."""

    def __init__(self, criterions_cfgs, preference, epsilon=0):
        super().__init__()
        self.criterions = [get_criterion(cfg) for cfg in criterions_cfgs]
        self.preference = preference
        self.epsilon = epsilon

    def forward(self, pred_lists, target_lists=None):
        if not (len(pred_lists) == len(self.criterions)):
            raise ValueError("pred_lists, criterions, and must have the same length.")
        loss = torch.zeros(1, device=pred_lists[0].device)
        for idx, (preference, pred, target) in enumerate(
            zip(self.preference, pred_lists, target_lists)
        ):
            loss += preference * (self._get_score(pred, target, idx) + self.epsilon)
        return loss

    def _get_score(self, pred, target, idx):
        def entropy(pred):
            probs = F.softmax(pred, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy.mean()

        if target is None:
            return entropy(pred)
        else:
            return self.criterions[idx](pred, target)[0]


def get_vit_from_id(model_id):
    if "clip" in model_id.lower():
        return CLIPVisionModelWithProjection.from_pretrained(model_id, use_safetensors=True)
    elif "google/vit" in model_id.lower():
        return ViTModel.from_pretrained(model_id, use_safetensors=True)
    elif "llama" in model_id.lower():
        model = AutoModel.from_pretrained(
            model_id, return_dict=True, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        model.config.use_cache = False
        model.config.pretrained_tp = 1
        return model
    elif "llava" in model_id.lower():
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        return model
    else:
        raise ValueError(f"Unknown model id: {model_id}. Supported models are CLIP and ViT.")
