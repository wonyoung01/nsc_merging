from collections.abc import MutableMapping
from copy import deepcopy

import torch
from merge_strategy import TA, KnOTS, LoRA_LEGO, Ties
from peft import PeftModel
from transformers import LlavaForConditionalGeneration

from src.vlm.utils import remove_modules_to_save


class TVDict(MutableMapping):
    def __init__(self):
        self.tv = {}
        self.merged_tv = {}
        self.dtype = torch.float32

    def add_task(self, model, task_name):
        if task_name not in self.tv:
            self.tv[task_name] = {}
        for key, param in model.base_model.named_parameters():
            if "lora_" in key and task_name in key:
                new_key = key.replace(task_name + ".", "")
                self.tv[task_name][new_key] = param.detach().clone().to(self.dtype)

    def add_task_full(self, model, task_name):
        # Use this when LoRA A is full delta already
        if task_name not in self.tv:
            self.tv[task_name] = {}
        for key, param in model.base_model.named_parameters():
            if task_name in key and "lora_A" in key:
                new_key = key.replace("lora_A" + ".", "")
                new_key = new_key.replace(task_name + ".", "")
                self.tv[task_name][new_key] = param.detach().clone().to(self.dtype)

    def get_merged_update(self, merge_kwargs):
        try:
            method = merge_kwargs["method"]
        except KeyError as e:
            raise ValueError(
                "Merging method must be specified in merge_kwargs with key 'method'"
            ) from e
        if method == "ta":
            merged_tv = TA(self.tv, scaling_factor=merge_kwargs.get("scaling_factor", 0.3))
        elif method == "ties":
            merged_tv = Ties(
                self.tv,
                topK=merge_kwargs.get("topK", 0.9),
                scaling_factor=merge_kwargs.get("scaling_factor", 0.3),
            )
        elif method == "dare_ties":
            merged_tv = Ties(
                self.tv,
                topK=merge_kwargs.get("topK", 0.9),
                drop_p=merge_kwargs.get("drop_p", 0.9),
                scaling_factor=merge_kwargs.get("scaling_factor", 0.3),
                cache_path=merge_kwargs.get("cache_path", None),
                profile=merge_kwargs.get("profile", False),
            )
        elif method == "knots_ties":
            merged_tv = KnOTS(
                self.tv,
                topK=merge_kwargs.get("topK", 0.9),
                scaling_factor=merge_kwargs.get("scaling_factor", 0.3),
                cache_path=merge_kwargs.get("cache_path", None),
                profile=merge_kwargs.get("profile", False),
            )
        elif method == "knots_dare_ties":
            merged_tv = KnOTS(
                self.tv,
                topK=merge_kwargs.get("topK", 0.9),
                drop_p=merge_kwargs.get("drop_p", 0.9),
                scaling_factor=merge_kwargs.get("scaling_factor", 0.3),
                cache_path=merge_kwargs.get("cache_path", None),
                profile=merge_kwargs.get("profile", False),
            )
        elif method == "lora_lego":
            merged_tv = LoRA_LEGO(
                self.tv,
                cache_path=merge_kwargs.get("cache_path", None),
                profile=merge_kwargs.get("profile", False),
            )
        else:
            raise ValueError(f"Unknown merging method: {method}")
        self.merged_tv = merged_tv

    def merge_all_tasks(self, model):
        model_dtype = next(model.parameters()).dtype
        if self.merged_tv == {}:
            raise ValueError("Merged task vectors not found. Please run get_merged_update first.")
        for k, tv in self.merged_tv.items():
            new_k = k.replace("model.", "", 1)
            model.get_parameter(new_k).data += tv.to(model_dtype)

    # Required methods
    def __getitem__(self, key):
        return self.tv[key]

    def __setitem__(self, key, value):
        self.tv[key] = value

    def __delitem__(self, key):
        del self.tv[key]

    def __iter__(self):
        return iter(self.tv)

    def __len__(self):
        return len(self.tv)

    # Optional: pretty printing
    def __repr__(self):
        return f"{self.__class__.__name__}({self.tv!r})"


def merge_loras(base_model, lora_dict, weights, combination_type=None, **kwargs):
    """
    Merges LoRA layers from multiple tasks into a single model using linear weighting.

    Args:
        base_model (PretrainedModel): The base model to which LoRA layers will be added.
        lora_dict (dict): A dictionary where keys are task names and values are dictionaries
                          containing 'path' (to the LoRA checkpoint) and
                          'weight' (the merging weight).

    Returns:
        torch.nn.Module: The merged model.
    """
    tv = TVDict()
    first_task = list(lora_dict.keys())[0]
    first_lora_path = lora_dict[first_task]
    return_model = deepcopy(base_model)
    model = PeftModel.from_pretrained(base_model, first_lora_path, first_task)

    # Load the remaining LoRA layers
    for task_name, lora_path in list(lora_dict.items())[1:]:
        model.load_adapter(lora_path, adapter_name=task_name)

    # Extract task vectors for each task
    for task_name in lora_dict.keys():
        tv.add_task(model, task_name)
    # Compute the merged task vector
    tv.get_merged_update(kwargs)

    if isinstance(model.base_model.model, LlavaForConditionalGeneration):
        model = remove_modules_to_save(model)

    # Merge the LoRA layers with the specified weights
    if combination_type is None:
        combination_type = "ta"

    for lora_name in model.peft_config.keys():
        modules_to_save_status = model.peft_config[lora_name].modules_to_save  # type: ignore
        if modules_to_save_status is not None:
            model.peft_config[lora_name].modules_to_save = None  # type: ignore

    if isinstance(model.base_model.model, LlavaForConditionalGeneration):
        for peft_mod, base_mod in zip(
            [
                model.multi_modal_projector,
                model.multi_modal_projector.linear_1,
                model.multi_modal_projector.linear_2,
            ],
            [
                return_model.multi_modal_projector,
                return_model.multi_modal_projector.linear_1,
                return_model.multi_modal_projector.linear_2,
            ],
        ):
            if isinstance(peft_mod, PeftModel):
                mod_tv = TVDict()
                mod_kwargs = kwargs.copy()
                mod_kwargs["method"] = combination_type
                mod_kwargs["cache_path"] = None
                if combination_type in ["ta", "ties"] and kwargs["method"] != "combination_type":
                    mod_kwargs["scaling_factor"] = 0.4
                for task_name in lora_dict.keys():
                    mod_tv.add_task_full(peft_mod, task_name)
                mod_tv.get_merged_update(mod_kwargs)
                for k, tv_param in mod_tv.merged_tv.items():
                    new_k = k.replace("model.", "", 1)
                    new_k = new_k.replace("linear.", "", 1)
                    base_mod.get_parameter(new_k).data += tv_param
    tv.merge_all_tasks(return_model)
    return return_model
