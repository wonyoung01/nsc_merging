import os
from abc import ABC, abstractmethod
from collections import defaultdict

import ipdb  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from peft.tuners.lora import LoraLayer
from peft.utils import _get_submodules

NSC_LORA_RANK = 16


class HookManager:
    def __init__(self):
        self.handles = []

    def add_hook(self, module, hook_fn):
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def is_empty(self):
        return len(self.handles) == 0


class ProjectorCluster:
    """
    Wrapper for a nn.Modeul that applies a kernel projection
    to the input before passing it through the model.
    Args:
        model (nn.Module): The underlying model to wrap peft model.
    """

    def __init__(self, model, use_kernel=True, use_image=False):
        self.use_kernel = use_kernel
        self.use_image = use_image
        if not (use_kernel or use_image):
            raise ValueError("At least one of use_kernel or use_image must be True.")
        self.vit = model
        self.active_adapter = self.vit.active_adapter
        self.kernel_projectors = {}
        self.image_projectors = {}
        for adapter in self.vit.peft_config.keys():
            self.compute_kernel_projection(adapter)

    def compute_kernel_projection(self, adapter_name):
        """
        Compute the kernel projection matrix based on the reference layer's weight.
        This is a placeholder for the actual implementation.
        """
        if adapter_name not in self.vit.peft_config.keys():
            raise ValueError(f"Adapter {adapter_name} not found in the model's peft_config.")
        self.kernel_projectors[adapter_name] = {}
        self.image_projectors[adapter_name] = {}
        key_list = [key for key, _ in self.vit.model.named_modules() if self.vit.prefix not in key]
        for key in key_list:
            if "multi_modal_projector" in key:
                continue
            _, target, _ = _get_submodules(self.vit.model, key)
            if isinstance(target, LoraLayer):
                if not hasattr(target, "lora_A"):
                    raise ValueError(f"LoraLayer {key} does not have lora_A attribute")
                if not hasattr(target, "lora_B"):
                    raise ValueError(f"LoraLayer {key} does not have lora_B attribute")
                lora_A = getattr(target.lora_A, adapter_name, None)
                lora_B = getattr(target.lora_B, adapter_name, None)
                if lora_A is None:
                    raise ValueError(f"Adapter {adapter_name} not found in {target.lora_A}")
                if lora_B is None:
                    raise ValueError(f"Adapter {adapter_name} not found in {target.lora_B}")
                if self.use_kernel:
                    self.kernel_projectors[adapter_name][key] = KernelProjector(
                        adapter_name + "-" + key + ".lora_A", lora_A
                    )
                if self.use_image:
                    self.image_projectors[adapter_name][key] = ImageProjector(
                        adapter_name + "-" + key + ".lora_B", lora_B
                    )

    def send_to_device(self, device, adapter_name=None):
        if adapter_name is None:
            if self.use_kernel:
                for name in self.kernel_projectors.keys():
                    for projector in self.kernel_projectors[name].values():
                        projector.to(device)
            if self.use_image:
                for name in self.image_projectors.keys():
                    for projector in self.image_projectors[name].values():
                        projector.to(device)
            return
        if adapter_name not in self.vit.peft_config.keys():
            raise ValueError(f"Adapter {adapter_name} not found in the model's peft_config.")
        if self.use_kernel:
            for projector in self.kernel_projectors[adapter_name].values():
                projector.to(device)
        if self.use_image:
            for image_projector in self.image_projectors[adapter_name].values():
                image_projector.to(device)


class Projector(nn.Module, ABC):
    def __init__(self, name, reference_layer):
        super().__init__()
        self.name = name
        self.original_weight = reference_layer.weight.data.clone().to(torch.float64)

    @abstractmethod
    def get_percent(self, x, y):
        pass

    @abstractmethod
    def get_ratio(self, x, y):
        pass

    @abstractmethod
    def compute_proj(self, weight: torch.Tensor) -> torch.Tensor:
        pass


class KernelProjector(Projector):
    def __init__(self, name, reference_layer):
        super().__init__(name, reference_layer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _weight = self.original_weight.to(device)
        aat = _weight @ _weight.T + 1e-8 * torch.eye(_weight.shape[0], device=_weight.device)
        L = torch.linalg.cholesky(aat)
        aat_inv = torch.cholesky_inverse(L).to(torch.float32)
        aat_inv = aat_inv.to(self.original_weight.device)
        self.register_buffer("inv", aat_inv)

    def get_percent(self, x, y=None):
        numerator = (y * F.linear(y, self.inv)).sum(dim=-1)
        denominator = (x * x).sum(dim=-1) + 1e-8
        return torch.sqrt(
            1 - (numerator / denominator)
        ).mean().item() * 100, denominator.sqrt().mean().item()

    def get_ratio(self, x, y):
        numerator = (y * F.linear(y, self.inv)).sum(dim=-1)
        denominator = (x * x).sum(dim=-1) + 1e-8
        return torch.sqrt(1 - (numerator / denominator)).mean()

    def compute_proj(self, weight):
        with torch.no_grad():
            null_dim = weight.shape[1] - weight.shape[0]
            if null_dim <= 0:
                raise ValueError(
                    f"Kernel projection requires more columns than rows. Got {weight.shape}"
                )
            _, _, Vh = torch.linalg.svd(weight, full_matrices=True)
            N = Vh.T[:, -null_dim:]
            return (N @ N.T).to(torch.float32)


class ImageProjector(Projector):
    def __init__(self, name, reference_layer):
        super().__init__(name, reference_layer)
        self.register_buffer("proj", self.compute_proj(self.original_weight))

    def forward(self, x):
        return F.linear(x, self.proj)

    def get_percent(self, x, y=None):
        x_proj = self.forward(x)
        kernel_space = torch.norm(x_proj, dim=-1)
        total_space = torch.norm(x, dim=-1)
        ratio = kernel_space / (total_space + 1e-8)
        return ratio.mean().item() * 100, total_space.mean().item()

    def get_ratio(self, x, y=None):
        x_proj = self.forward(x)
        kernel_space = torch.norm(x_proj, dim=-1)
        total_space = torch.norm(x, dim=-1)
        return (kernel_space / (total_space + 1e-8)).mean()

    def compute_proj(self, weight):
        with torch.no_grad():
            NtN = weight.T @ weight + 1e-8 * torch.eye(weight.shape[1], device=weight.device)
            return (weight @ torch.linalg.solve(NtN, weight.T)).to(torch.float32)


class LossCluster:
    """
    A class to manage loss values for different kernel projectors.
    It can be used to store and compute losses based on the kernel projections.
    """

    def __init__(self):
        self.kernel_losses = defaultdict(lambda: {})
        self.image_losses = defaultdict(lambda: {})

    def add_kernel_loss(self, key, adapter_name, value):
        if key in self.kernel_losses[adapter_name]:
            self.kernel_losses[adapter_name][key].append(value)
        else:
            self.kernel_losses[adapter_name][key] = [value]

    def add_image_loss(self, key, adapter_name, value):
        if key in self.image_losses[adapter_name]:
            self.image_losses[adapter_name][key].append(value)
        else:
            self.image_losses[adapter_name][key] = [value]

    def get_kernel_loss(self, key, adapter_name):
        if adapter_name not in self.kernel_losses:
            raise ValueError(f"Adapter {adapter_name} not found in losses.")
        val = self.kernel_losses[adapter_name].get(key, None)
        if val is None:
            raise ValueError(f"Key {key} not found in adapter {adapter_name} losses.")
        return sum(val) / len(val)

    def get_image_loss(self, key, adapter_name):
        if adapter_name not in self.image_losses:
            raise ValueError(f"Adapter {adapter_name} not found in losses.")
        val = self.image_losses[adapter_name].get(key, None)
        if val is None:
            raise ValueError(f"Key {key} not found in adapter {adapter_name} losses.")
        return sum(val) / len(val)

    def return_kernel_loss(self, reduction="mean"):
        """
        Returen a whole leaf tensor of losses according to the reduction type
        """
        _kernel_loss_list = [
            self.get_kernel_loss(key, adapter_name)
            for adapter_name in self.kernel_losses.keys()
            for key in self.kernel_losses[adapter_name].keys()
        ]
        if _kernel_loss_list == []:
            return torch.tensor(0.0)
        kernel_loss = torch.stack(_kernel_loss_list).sum()
        kernel_cnt = len(_kernel_loss_list)
        if reduction == "sum":
            kernel_loss = kernel_loss / len(self.kernel_losses.keys())
        elif reduction == "mean":
            if kernel_cnt > 0:
                kernel_loss = kernel_loss / kernel_cnt
            else:
                kernel_loss = torch.tensor(0.0, device=kernel_loss.device)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}. Use 'mean' or 'sum'.")
        return kernel_loss

    def return_image_loss(self, reduction="mean"):
        """
        Return a whole leaf tensor of image losses according to the reduction type
        """
        _image_loss_list = [
            self.get_image_loss(key, adapter_name)
            for adapter_name in self.image_losses.keys()
            for key in self.image_losses[adapter_name].keys()
        ]
        if _image_loss_list == []:
            return torch.tensor(0.0)
        image_loss = torch.stack(_image_loss_list).sum()
        image_cnt = len(_image_loss_list)
        if reduction == "sum":
            image_loss = image_loss / len(self.image_losses.keys())
        elif reduction == "mean":
            if image_cnt > 0:
                image_loss = image_loss / image_cnt
            else:
                image_loss = torch.tensor(0.0, device=image_loss.device)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}. Use 'mean' or 'sum'.")
        return image_loss

    def return_loss(self, reduction="mean"):
        """
        Return a whole leaf tensor of losses according to the reduction type
        """
        kernel_loss = self.return_kernel_loss(reduction)
        image_loss = self.return_image_loss(reduction)
        return kernel_loss + image_loss

    def clear(self):
        self.kernel_losses.clear()
        self.image_losses.clear()

    def is_empty(self):
        return len(self.kernel_losses) == 0


def get_projection_writer_hook(output_dir, projector, **kwargs):
    use_wandb = kwargs.get("use_wandb", False)

    def hook(module, input, output):
        filename = os.path.join(
            output_dir, kwargs.get("filename", f"{projector.name}_projection.json")
        )
        projector.to(output.device)
        with open(filename, "a") as f:
            if isinstance(projector, KernelProjector):
                percent, scale = projector.get_percent(input[0], output)
                if use_wandb:
                    wandb.log(
                        {
                            f"kernel/{projector.name}": percent,
                            f"kernel_scale/{projector.name}": scale,
                        }
                    )
                f.write(f'{{"kernel_percent": {percent:.2f}, "scale": {scale:.4f}}}\n')
            elif isinstance(projector, ImageProjector):
                percent, scale = projector.get_percent(output)
                if use_wandb:
                    wandb.log(
                        {
                            f"image/{projector.name}": percent,
                            f"image_scale/{projector.name}": scale,
                        }
                    )
                f.write(f'{{"image_percent": {percent:.2f}, "scale": {scale:.4f}}}\n')
            else:
                raise ValueError(f"Unsupported projector type: {type(projector)}")

    return hook


def get_projection_loss_hook(
    projector: Projector, key: str, curr_adapter: str, loss_class: LossCluster, **kwargs
):
    use_wandb = kwargs.get("use_wandb", False)
    target_adapter = kwargs.get("target_adapter", None)
    if target_adapter is None:
        raise ValueError("target_adapter must be provided in kwargs.")

    def hook(module, input, output):
        projector.to(output.device)
        if isinstance(projector, KernelProjector):
            start, end = _get_start_end(target_adapter)
            loss = projector.get_ratio(input[0], output[..., start:end])
            loss_class.add_kernel_loss(key, curr_adapter, loss)
            if use_wandb:
                wandb.log({f"kernel_loss/{projector.name}": loss.item()})
        elif isinstance(projector, ImageProjector):
            ratio = projector.get_ratio(output)
            loss = 1.0 - ratio
            loss_class.add_image_loss(key, curr_adapter, loss)
            if use_wandb:
                wandb.log({f"image_loss/{projector.name}": loss.item()})
        else:
            raise ValueError(f"Unsupported projector type: {type(projector)}")

    return hook


def _get_start_end(curr_adapter):
    i = j = 0

    # CLIP Image Classification
    if "stanford_cars" in curr_adapter:
        i, j = 0, 1
    elif "dtd" in curr_adapter:
        i, j = 1, 2
    elif "eurosat" in curr_adapter:
        i, j = 2, 3
    elif "gtsrb" in curr_adapter:
        i, j = 3, 4
    elif "mnist" in curr_adapter:
        i, j = 4, 5
    elif "resisc45" in curr_adapter:
        i, j = 5, 6
    elif "sun397" in curr_adapter:
        i, j = 6, 7
    elif "svhn" in curr_adapter:
        i, j = 7, 8
    # NLI
    elif "mnli" in curr_adapter:
        i, j = 0, 1
    elif "qnli" in curr_adapter:
        i, j = 1, 2
    elif "snli" in curr_adapter:
        i, j = 2, 3
    elif "rte" in curr_adapter:
        i, j = 3, 4
    elif "sick" in curr_adapter:
        i, j = 4, 5
    elif "scitail" in curr_adapter:
        i, j = 5, 6
    # LLaVA
    elif "VizWiz_VQA" in curr_adapter:
        i, j = 0, 1
    elif "IconQA" in curr_adapter:
        i, j = 1, 2
    elif "ChartQA" in curr_adapter:
        i, j = 2, 3
    elif "DocVQA" in curr_adapter:
        i, j = 3, 4
    elif "COCO" in curr_adapter:
        i, j = 4, 5
    elif "Flickr" in curr_adapter:
        i, j = 5, 6
    else:
        raise ValueError(f"Unknown adapter name: {curr_adapter}")

    return i * NSC_LORA_RANK, j * NSC_LORA_RANK
