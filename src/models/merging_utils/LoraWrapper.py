from copy import deepcopy
from dataclasses import replace

import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import Linear, LoraLayer
from peft.utils import _freeze_adapter, _get_submodules


class LoraModelTrainableMerging:
    def __init__(self, lora_model):
        self.lora_model = lora_model
        self.cache_svd = {}

    def add_trainable_adapter(
        self,
        src_name,
        tgt_name,
        adapter_wise_num_coeffs,
        adapter_wise_rank,
        coeff_scaler,
        merge_type,
        coeff_type,
    ):
        """
        Adds a trainable LoRA adapter to the model.

        Args:
            src_name (str): Name of the source adapter to copy.
            tgt_name (str): Name of the target adapter to create.

        Returns:
            None
        """
        if src_name not in self.lora_model.peft_config:
            raise ValueError(f"Adapter {src_name} does not exist.")
        # If the target adapter already exists, delete it
        if tgt_name in self.lora_model.peft_config:
            raise ValueError(f"Adapter {tgt_name} already exists. Please delete it first.")
        assert self.active_adapter == src_name, (
            f"Active adapter {self.active_adapter} does not match source adapter {src_name}."
        )
        self.lora_model.peft_config[tgt_name] = deepcopy(self.lora_model.peft_config[src_name])

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            parent, target, target_name = _get_submodules(self.model, key)
            # If already trainable, skip
            if isinstance(target, LinearTrainable):
                continue
            # If not a LoraLayer, skip
            if not isinstance(target, LoraLayer):
                continue
            assert isinstance(target, Linear), f"Target {target} is not a Linear class."
            new_target = LinearTrainable(
                new_name=tgt_name,
                linear_layer=target,
                adapter_wise_num_coeffs=adapter_wise_num_coeffs,
                adapter_wise_rank=adapter_wise_rank,
                coeff_scaler=coeff_scaler,
                merge_type=merge_type,
                coeff_type=coeff_type,
            )
            setattr(parent, target_name, new_target)
        self.set_adapter(tgt_name)

    def concat_adapters_weight(self, adapters, adapter_name):
        """
        Concatenates multiple LoRA adapters into a single adapter with specified weights.
        Unlike default huggingface merging, I want the merged parameters to have gradients
        flow.

        Args:
            adapters (list): List of adapter names to concatenate.
            adapter_name (str): Name for the concatenated adapter.

        Returns:
            None
        """
        # Delete the adapter if it already exists
        if adapter_name in self.peft_config:
            self.delete_adapter(adapter_name)
        _, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type="cat",
            svd_rank=None,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if not isinstance(target, LoraLayer):
                continue
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            target_lora_A.data = target_lora_A.data * 0.0
            target_lora_B.data = target_lora_B.data * 0.0
            loras_A, loras_B = [], []
            for adapter in adapters:
                if adapter in target.lora_A:
                    current_adapter_lora_A = target.lora_A[adapter].weight
                    current_adapter_lora_B = target.lora_B[adapter].weight
                elif adapter in target.lora_embedding_A:
                    current_adapter_lora_A = target.lora_embedding_A[adapter]
                    current_adapter_lora_B = target.lora_embedding_B[adapter]
                else:
                    continue
                loras_A.append(current_adapter_lora_A.data * target.scaling[adapter])
                loras_B.append(current_adapter_lora_B.data)

            if len(loras_A) == 0:
                raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
            loras_A = torch.cat(loras_A, dim=0)
            loras_B = torch.cat(loras_B, dim=1)
            target_lora_A[: loras_A.shape[0], :] = loras_A
            target_lora_B[:, : loras_B.shape[1]] = loras_B

    def concat_adapters_msu(self, adapters, adapter_name):
        """
        Concatenates multiple LoRA adapters into a single adapter with specified weights.
        Unlike default huggingface merging, I want the merged parameters to have gradients
        flow.

        Args:
            adapters (list): List of adapter names to concatenate.
            adapter_name (str): Name for the concatenated adapter.

        Returns:
            None
        """
        # Delete the adapter if it already exists
        if adapter_name in self.peft_config:
            self.delete_adapter(adapter_name)
        _, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type="cat",
            svd_rank=None,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if not isinstance(target, LoraLayer):
                continue
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            target_lora_A.data = target_lora_A.data * 0.0
            target_lora_B.data = target_lora_B.data * 0.0
            loras_A, loras_B = [], []
            for adapter in adapters:
                if adapter in target.lora_A:
                    current_adapter_lora_A = target.lora_A[adapter].weight
                    current_adapter_lora_B = target.lora_B[adapter].weight
                elif adapter in target.lora_embedding_A:
                    current_adapter_lora_A = target.lora_embedding_A[adapter]
                    current_adapter_lora_B = target.lora_embedding_B[adapter]
                else:
                    continue
                loras_A.append(current_adapter_lora_A.data * target.scaling[adapter])
                loras_B.append(current_adapter_lora_B.data)

            if len(loras_A) == 0:
                raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
            loras_A = torch.cat(loras_A, dim=0)
            loras_B = torch.cat(loras_B, dim=1)
            target_lora_A[: loras_A.shape[0], :] = loras_A
            target_lora_B[:, : loras_B.shape[1]] = loras_B

    def concat_adapters_msu_neg(self, adapters, adapter_name):
        """
        Concatenates multiple LoRA adapters into a single adapter with specified weights.
        Unlike default huggingface merging, I want the merged parameters to have gradients
        flow. This method allows for merging with negative coefficients,

        Args:
            adapters (list): List of adapter names to concatenate.
            adapter_name (str): Name for the concatenated adapter.

        Returns:
            None
        """
        # Delete the adapter if it already exists
        if adapter_name in self.peft_config:
            self.delete_adapter(adapter_name)
        _, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type="cat",
            svd_rank=None,
        )

        # TODO: Check whether it is the best way to do it
        new_rank = new_rank * 2
        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if not isinstance(target, LoraLayer):
                continue
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            target_lora_A.data = target_lora_A.data * 0.0
            target_lora_B.data = target_lora_B.data * 0.0
            loras_A, loras_B = [], []
            for adapter in adapters:
                if adapter in target.lora_A:
                    current_adapter_lora_A = target.lora_A[adapter].weight
                    current_adapter_lora_B = target.lora_B[adapter].weight
                elif adapter in target.lora_embedding_A:
                    current_adapter_lora_A = target.lora_embedding_A[adapter]
                    current_adapter_lora_B = target.lora_embedding_B[adapter]
                else:
                    continue
                # Expand the weight
                current_adapter_lora_A = torch.cat(
                    [current_adapter_lora_A, -current_adapter_lora_A], dim=0
                )
                current_adapter_lora_B = torch.cat(
                    [current_adapter_lora_B, current_adapter_lora_B], dim=1
                )
                loras_A.append(current_adapter_lora_A.data * target.scaling[adapter])
                loras_B.append(current_adapter_lora_B.data)

            if len(loras_A) == 0:
                raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
            loras_A = torch.cat(loras_A, dim=0)
            loras_B = torch.cat(loras_B, dim=1)
            target_lora_A[: loras_A.shape[0], :] = loras_A
            target_lora_B[:, : loras_B.shape[1]] = loras_B

    def concat_adapters_svd(self, adapters, adapter_name):
        """
        Concatenates multiple LoRA adapters into a single adapter with specified coefficients.
        This method allows for trainable merging of LoRA layers.

        Args:
            adapters (list): List of adapter names to concatenate.
            adapter_name (str): Name for the concatenated adapter.

        Returns:
            None
        """
        # Delete the adapter if it already exists
        if adapter_name in self.peft_config:
            self.delete_adapter(adapter_name)
        _, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type="cat",
            svd_rank=None,
        )

        # TODO: Check whether it is the best way to do it
        new_rank = new_rank * len(adapters)

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if not isinstance(target, LoraLayer):
                continue
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            target_lora_A.data = target_lora_A.data * 0.0
            target_lora_B.data = target_lora_B.data * 0.0

            loras_A_concat, us = self._get_svd(
                adapters, key, target, rank=(new_rank // len(adapters))
            )
            target_lora_B[:, :] = torch.cat([us for _ in adapters], dim=1)

            # Merge LoRAs with coefficients
            step = loras_A_concat.shape[1] // len(adapters)
            stack_list = [
                loras_A_concat[:, start : start + step]
                for start in range(0, loras_A_concat.shape[1], step)
            ]
            target_lora_A[:, :] = torch.cat(stack_list, dim=0)

    @torch.no_grad()
    def _get_svd(self, adapters, key, target, rank):
        """
        Computes the SVD of the concatenated LoRA adapters.

        Args:
            adapters (list): List of adapter names to concatenate.
            key (str): The key in the model where the LoRA layer is located.
            target (LoraLayer): The target LoRA layer to apply SVD on.
            rank (int): The rank for the SVD.

        Returns:
            torch.Tensor: The singular values from the SVD.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        original_dtype = next(target.parameters()).dtype
        if key in self.cache_svd:
            return self.cache_svd[key]["lora_A"], self.cache_svd[key]["lora_B"]
        w_concat = torch.cat([target.get_delta_weight(adapter) for adapter in adapters], dim=1)
        w_concat = w_concat.to(torch.float32).to(device)
        u, s, vt = torch.linalg.svd(w_concat, full_matrices=False)
        s = s[:rank]
        u = u[:, :rank]
        vt = vt[:rank, :]
        lora_B = torch.mm(u, torch.diag(s)).to(original_dtype)
        lora_A = vt.to(original_dtype)
        self.cache_svd[key] = {
            "lora_A": lora_A.cpu(),
            "lora_B": lora_B.cpu(),
        }
        return lora_A.cpu(), lora_B.cpu()

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying LoRA model.
        This allows the wrapper to behave like the original model.
        """
        return getattr(self.lora_model, name)


class LinearTrainable(Linear):
    """
    A wrapper for the Linear layer to make it trainable.
    This is used to replace the Linear layer in the LoRA model.
    """

    def __init__(
        self,
        new_name: str,
        linear_layer: Linear,
        adapter_wise_num_coeffs: list,
        adapter_wise_rank: list,
        coeff_scaler: float = 1.0,
        temperature: float = 1.0,
        merge_type: str = "weight",
        coeff_type: str = "prob",
        **kwargs,
    ):
        adapter_name = linear_layer._active_adapter
        if isinstance(adapter_name, list):
            adapter_name = adapter_name[0]

        super().__init__(
            base_layer=linear_layer.base_layer,
            adapter_name=new_name,
            r=linear_layer.r[adapter_name],
            lora_alpha=linear_layer.lora_alpha[adapter_name],
            lora_dropout=0.0,  # Not learning real parameters, so no dropout
            fan_in_fan_out=linear_layer.fan_in_fan_out,
            is_target_conv_1d_layer=linear_layer.is_target_conv_1d_layer,
            init_lora_weights=False,
            use_rslora=False,
            use_dora=linear_layer.use_dora[adapter_name],
            lora_bias=linear_layer.lora_bias[adapter_name],
            **kwargs,
        )
        # super().__init__() creates the adapter of new_name. Check if it exists
        # Copy the adapter attributes from the original layer
        if new_name in self.lora_A:
            original_lora_A = linear_layer.lora_A[adapter_name]
            original_lora_B = linear_layer.lora_B[adapter_name]
            self.lora_A[new_name] = nn.Sequential(
                deepcopy(original_lora_A),
                DiagonalLinear(
                    n_features=original_lora_A.weight.shape[0],
                    temperature=temperature,
                    adapter_wise_num_coeffs=adapter_wise_num_coeffs,
                    adapter_wise_rank=adapter_wise_rank,
                    merge_type=merge_type,
                    coeff_type=coeff_type,
                    scaler=coeff_scaler,
                ),
            )
            self.lora_B[new_name] = deepcopy(original_lora_B)
        else:
            raise ValueError(
                f"Adapter {new_name} not found in lora_A or lora_embedding_A. "
                "Please check the adapter name."
            )

    def get_rank(self, adapter_name=None):
        if adapter_name is None:
            adapter_name = self._active_adapter
            if isinstance(adapter_name, list):
                adapter_name = adapter_name[0]
        coeff = self.lora_A[adapter_name][1]._logit_to_coeff()
        return coeff.shape[0]

    def get_delta_weight(self, adapter_name=None):
        if adapter_name is None:
            adapter_name = self._active_adapter
            if isinstance(adapter_name, list):
                adapter_name = adapter_name[0]
        lora_A = self.lora_A[adapter_name][0].weight.data
        lora_B = self.lora_B[adapter_name].weight.data
        coeff = self.lora_A[adapter_name][1]._logit_to_coeff().to(lora_A.device)
        if coeff.dim() == 1:
            return lora_B @ (lora_A * coeff.unsqueeze(1))
        elif coeff.dim() == 2:
            return lora_B @ (coeff @ lora_A)
        else:
            raise ValueError(f"Unsupported logit dimension: {self.logit.dim()}")

    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                scaling = self.scaling[active_adapter]
                result = result + lora_B(lora_A(x)) * scaling

            result = result.to(torch_result_dtype)

        return result


class DiagonalLinear(nn.Module):
    def __init__(
        self,
        n_features,
        adapter_wise_num_coeffs,
        adapter_wise_rank,
        merge_type,
        coeff_type,
        scaler,
        temperature=1.0,
    ):
        """
        A trainable diagonal linear layer that applies a softmax scaling to the input features.
        This layer is used to scale the coefficients of the LoRA adapters.
        Args:
            n_features (int): Number of features in the input.
            adapter_wise_num_coeffs (list): List of coefficients for each adapter.
            temperature (float): Temperature for the softmax scaling.
            scaler (float): Scaling factor for the coefficients.
            init (str): Initialization method for the coefficients. Default is "uniform".
            coeff_type (str): Type of coefficients to use.
                            Default is "linear". Can be "prob" or "linear" or "exp".
        """
        super().__init__()
        self.merge_type = merge_type
        self.coeff_type = coeff_type
        if coeff_type in ["sigmoid"]:
            assert merge_type == "weight" or merge_type == "msu", (
                "Coefficient type 'sigmoid' is only compatible with merge_type 'weight'. "
                "Please use a different coefficient type."
            )
            n = len(adapter_wise_num_coeffs)
            if merge_type == "weight":
                self.logit = nn.Parameter(-np.log(n / scaler - 1) * torch.ones(n))
            elif merge_type == "msu":
                self.logit = nn.Parameter(-np.log(n / scaler - 1) * torch.ones(n_features))
        elif coeff_type in ["general"]:
            assert merge_type != "weight", (
                "Coefficient type 'general' is not compatible with merge_type 'weight'. "
                "Please use a different coefficient type."
            )
            self.logit = nn.Parameter(
                scaler * torch.eye(n_features) / float(len(adapter_wise_num_coeffs))
            )
        elif merge_type in ["svd", "msu"]:
            self.logit = nn.Parameter(
                scaler * torch.ones(n_features) / float(len(adapter_wise_num_coeffs))
            )
        elif merge_type in ["weight"]:
            self.logit = nn.Parameter(
                scaler
                * torch.ones(len(adapter_wise_num_coeffs))
                / float(len(adapter_wise_num_coeffs))
            )
        else:
            raise ValueError(f"Unknown initialization method for merging: {merge_type}")
        self.temperature = temperature
        self.scaler = scaler
        self.scaling_factor = torch.tensor(
            [num_coeff for num_coeff in adapter_wise_num_coeffs for _ in range(num_coeff)]
        )
        self.adapter_wise_num_coeffs = adapter_wise_num_coeffs
        self.adapter_wise_rank = adapter_wise_rank
        self.num_adapters = len(adapter_wise_num_coeffs)
        self.r = sum(adapter_wise_rank) // self.num_adapters

    def forward(self, x):
        torch_result_dtype = x.dtype
        if self.merge_type == "weight":
            # HACK: Not supporting LoRA with different ranks for now
            weights = self.logit.unsqueeze(-1).tile(self.r).flatten()
            result = x * weights
        elif self.logit.dim() == 1:
            weights = self._logit_to_coeff()  # (1, 197, 512)
            result = x * weights  # broadcasted over batch and token dimensions
        elif self.logit.dim() == 2:
            result = F.linear(x, self.logit, bias=None)
        else:
            raise ValueError(f"Unsupported logit dimension: {self.logit.dim()}")
        return result.to(torch_result_dtype)

    def _logit_to_coeff(self):
        """
        Convert the logit to coefficients.
        Returns:
            torch.Tensor: The coefficients tensor.
        """
        if self.coeff_type == "prob":
            # Convert logits to probabilities
            _scaling_factor = self.scaling_factor.to(self.logit.device)
            coeffs = self._logit_to_prob() * _scaling_factor * self.scaler
        elif self.coeff_type == "linear":  # Linear scaling
            coeffs = self.logit
        elif self.coeff_type == "exp":  # Exponential scaling
            coeffs = torch.exp(self.logit)
        elif self.coeff_type == "softplus":  # Softmax scaling
            coeffs = F.softplus(self.logit)
        elif self.coeff_type == "general":  # General case
            coeffs = self.logit.reshape(-1)
        elif self.coeff_type == "sigmoid":  # Sigmoid scaling
            coeffs = torch.sigmoid(self.logit)
        else:
            raise ValueError(f"Unknown coefficient type: {self.coeff_type}")

        if self.merge_type in ["weight"]:
            new_coeffs = []
            for c, rank in zip(coeffs, self.adapter_wise_rank):
                new_coeffs.append(c.repeat(rank))
            return torch.cat(new_coeffs, dim=0)
        return coeffs

    def _logit_to_prob(self):
        """
        Convert the logit to probabilities.
        Returns:
            torch.Tensor: The probabilities tensor.
        """
        assert self.coeff_type in ["prob", "linear", "exp", "softplus", "sigmoid"], (
            f"Unsupported coefficient type: {self.coeff_type}. "
            "Supported types are 'prob', 'linear', 'exp', 'sigmoid' and 'softplus'."
        )
        return F.softmax(self.logit / self.temperature, dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_features={self.logit.numel()}, scaler={self.scaler}, adapeter_wise_num_coeffs={self.adapter_wise_num_coeffs})"  # noqa: E501
