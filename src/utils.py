import argparse
import importlib
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from transformers.optimization import get_linear_schedule_with_warmup

NEED_RESOLVING = ["model", "data", "task", "dataloader", "train", "optimizer", "lora", "processor"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name, e.g., 'clip_vit_b_16'",
    )
    parser.add_argument(
        "--lora",
        type=str,
        help="LoRA configuration, e.g., 'lora16'",
    )

    # Catch the rest of the args as dotlist-style overrides
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Overrides like key=value")

    args = parser.parse_args()
    return args


def set_seed(seed, deterministic=False):
    """Set the random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}, deterministic={deterministic}")


def get_config(main_cfg, args):
    cfg_path = args.config
    overrides = args.overrides
    sub_cfg = {}
    cfg_dir = os.path.dirname(cfg_path)
    for name in NEED_RESOLVING:
        # if name not in main_cfg, skip
        if name not in main_cfg:
            continue
        if name in args.__dict__ and args.__dict__[name] is not None:
            main_cfg[name] = args.__dict__[name]
        subpath = os.path.join(cfg_dir, f"{name}/{main_cfg[name]}.yaml")
        sub_cfg[name] = OmegaConf.load(subpath)

    # Merge the main config with the sub-configs
    cfg = OmegaConf.merge(main_cfg, sub_cfg)

    # Override with command line arguments
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def get_device(device: str):
    """Get the device to use for training/testing"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    elif device not in ["cuda", "cpu"]:
        return "cpu"
    return device


def get_lora_config(cfg):
    """Get LoRA configuration from the config"""
    lora_params = {
        "r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha_multiplier * cfg.lora.r,
        "lora_dropout": cfg.lora.dropout,
        "bias": cfg.lora.bias,
    }
    if hasattr(cfg.lora, "target_modules") and cfg.lora.target_modules is not None:
        lora_params["target_modules"] = [module_name for module_name in cfg.lora.target_modules]
    if hasattr(cfg.lora, "modules_to_save") and cfg.lora.modules_to_save is not None:
        lora_params["modules_to_save"] = [module_name for module_name in cfg.lora.modules_to_save]
    if hasattr(cfg.lora, "task_type") and cfg.lora.task_type is not None:
        lora_params["task_type"] = cfg.lora.task_type
    if hasattr(cfg.lora, "inference_mode") and cfg.lora.inference_mode is not None:
        lora_params["inference_mode"] = cfg.lora.inference_mode

    return LoraConfig(**lora_params)


def get_criterion(cfg):
    """Get the loss criterion based on the task"""
    if cfg.task.name == "classification":
        from models import ClassificationLoss

        criterion = ClassificationLoss()
    elif cfg.task.name == "sequence_classification":
        from models import SequenceClassificationLoss

        criterion = SequenceClassificationLoss()
    elif cfg.task.name in ["vqa", "captioning"]:
        # FIXME: Currently, nothing to measure here
        criterion = None
    else:
        raise NotImplementedError(f"Criterion loading for {cfg.task.name} is not implemented yet.")
    return criterion


def get_optimizer(cfg, model, **kwargs):
    """Get the optimizer from the config"""
    if cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    if cfg.task.name == "sequence_classification":
        num_warmup_steps = kwargs.get("num_warmup_steps", None)
        num_training_steps = kwargs.get("num_training_steps", None)
        if num_warmup_steps is None or num_training_steps is None:
            raise ValueError(
                "num_warmup_steps and num_training_steps "
                "must be provided for sequence classification."
            )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)
    return optimizer, scheduler


def get_cliphead(path):
    """Load the CLIP head from a given path"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLIP head file not found at {path}")
    class_vector = torch.load(path)
    if class_vector.dim() != 2:
        raise ValueError(f"Expected class_vector to be 2D, got {class_vector.dim()}D")
    return class_vector


def get_dataset_class(cfg):
    kwargs = {
        "root": cfg.data.data_path,
    }
    if cfg.data.name in [
        "stanford_cars",
        "dtd",
        "eurosat",
        "gtsrb",
        "mnist",
        "resisc45",
        "sun397",
        "svhn",
    ]:
        assert cfg.task.name == "classification", (
            f"{cfg.data.name} dataset is for classification task only."
        )
        mapping = {
            "stanford_cars": "StanfordCars",
            "dtd": "DTD",
            "eurosat": "EuroSAT",
            "gtsrb": "GTSRB",
            "mnist": "MNIST",
            "resisc45": "RESISC45",
            "sun397": "SUN397",
            "svhn": "SVHN",
        }
        data_module = importlib.import_module("src.data")
        dataset = getattr(data_module, mapping[cfg.data.name])
    else:
        raise NotImplementedError(f"Dataset loading for {cfg.data.name} is not implemented yet.")
    return dataset, kwargs


def get_tokenizer(cfg):
    def grab_nli_loader_fns(name):
        """Returns the dataset loader functions for the specified NLI dataset"""
        if name == "snli":
            from data.snli import prepare_train_loaders
        elif name == "mnli":
            from data.mnli import prepare_train_loaders
        elif name == "sick":
            from data.sick import prepare_train_loaders
        elif name == "qnli":
            from data.qnli import prepare_train_loaders
        elif name == "rte":
            from data.rte import prepare_train_loaders
        elif name == "scitail":
            from data.scitail import prepare_train_loaders
        else:
            raise NotImplementedError(name)

        return prepare_train_loaders

    prepare_train_loaders = grab_nli_loader_fns(cfg.data.name)
    tokenizer = prepare_train_loaders(cfg)["tokenizer"]

    return tokenizer


def _get_dataloader_nli(cfg, val_shuffle=False):
    def grab_nli_loader_fns(name):
        """Returns the dataset loader functions for the specified NLI dataset"""
        if name == "snli":
            from data.snli import prepare_test_loaders, prepare_train_loaders
        elif name == "mnli":
            from data.mnli import prepare_test_loaders, prepare_train_loaders
        elif name == "sick":
            from data.sick import prepare_test_loaders, prepare_train_loaders
        elif name == "qnli":
            from data.qnli import prepare_test_loaders, prepare_train_loaders
        elif name == "rte":
            from data.rte import prepare_test_loaders, prepare_train_loaders
        elif name == "scitail":
            from data.scitail import prepare_test_loaders, prepare_train_loaders
        else:
            raise NotImplementedError(name)

        return prepare_train_loaders, prepare_test_loaders

    prepare_train_loaders, prepare_test_loaders = grab_nli_loader_fns(cfg.data.name)
    train_loader = prepare_train_loaders(cfg)["full"]
    test_loaders = prepare_test_loaders(cfg, val_shuffle=val_shuffle)
    val_loader = test_loaders["val"]
    test_loader = test_loaders["test"]

    return train_loader, val_loader, test_loader


def _get_dataloader_vlm(cfg, val_shuffle=False):
    from datasets import load_dataset

    if cfg.data.name == "ChartQA":
        train_dataset, val_dataset, test_dataset = load_dataset(
            "HuggingFaceM4/ChartQA", split=["train", "val", "test"]
        )
    elif cfg.data.name == "DocVQA":
        train_dataset, val_dataset = load_dataset(
            "pixparse/docvqa-single-page-questions", split=["train", "validation"]
        )
        _split_ds = val_dataset.train_test_split(test_size=0.8, seed=42)  # type: ignore
        val_dataset = _split_ds["train"]
        test_dataset = _split_ds["test"]
    elif cfg.data.name == "VizWiz_VQA":
        train_dataset, val_dataset = load_dataset(
            "HuggingFaceM4/VizWiz", split=["train", "validation"]
        )
        _split_ds = val_dataset.train_test_split(test_size=0.8, seed=42)  # type: ignore
        val_dataset = _split_ds["train"]
        test_dataset = _split_ds["test"]
    elif cfg.data.name == "Flickr":
        # Flickr30k dataset has 5 captions per image, we need to explode them for training
        def explode_captions(batch):
            out = {
                "image": [],
                "caption": [],
                "sentid": [],
                "split": [],
                "img_id": [],
                "filename": [],
            }
            for image, captions, sentids, split, img_id, filename in zip(
                batch["image"],
                batch["caption"],
                batch["sentids"],
                batch["split"],
                batch["img_id"],
                batch["filename"],
            ):
                # safety: align lengths
                n = min(len(captions), len(sentids))
                for i in range(n):
                    out["image"].append(image)  # duplicate the PIL image (pointer/feature)
                    out["caption"].append(captions[i])
                    out["sentid"].append(sentids[i])
                    out["split"].append(split)
                    out["img_id"].append(img_id)
                    out["filename"].append(filename)
            return out

        _dataset = load_dataset("nlphuji/flickr30k", split="test")
        _train_dataset = _dataset.filter(lambda example: example["split"] == "train")
        train_dataset = _train_dataset.map(
            explode_captions,
            batched=True,
            remove_columns=_dataset.column_names,  # type: ignore
        )
        val_dataset = _dataset.filter(lambda example: example["split"] == "val")
        test_dataset = _dataset.filter(lambda example: example["split"] == "test")
    elif cfg.data.name == "COCO":
        from collections import defaultdict

        from datasets import Dataset, load_from_disk

        HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        VAL_CACHE = os.path.join(HF_HOME, "datasets", "coco_karpathy_val_ds")
        TEST_CACHE = os.path.join(HF_HOME, "datasets", "coco_karpathy_test_ds")

        def aggregate_sentences_by_image(dset, key_col="imgid", sentence_col="sentences"):
            groups = defaultdict(list)
            for i, k in enumerate(dset[key_col]):
                groups[k].append(i)

            new_data = {
                key_col: [],
                "image": [],
                "filepath": [],
                "filename": [],
                "split": [],
                "sentences": [],
                "sentids": [],
                "cocoid": [],
            }

            for k, idxs in groups.items():
                first = dset[idxs[0]]
                new_data[key_col].append(k)
                new_data["image"].append(first["image"])
                new_data["filepath"].append(first["filepath"])
                new_data["filename"].append(first["filename"])
                new_data["split"].append(first["split"])
                new_data["cocoid"].append(first["cocoid"])
                new_data["sentids"].append([dset[i]["sentences"]["sentid"] for i in idxs])
                new_data["sentences"].append([dset[i]["sentences"] for i in idxs])

            return Dataset.from_dict(new_data)

        # Using Karpathy's COCO dataset on HuggingFace
        train_dataset, val_dataset_flat, test_dataset_flat = load_dataset(
            "HuggingFaceM4/COCO", split=["train", "validation", "test"]
        )
        if os.path.exists(VAL_CACHE) and os.path.exists(TEST_CACHE):
            print("✅ Using cached val/test datasets")
            val_dataset = load_from_disk(VAL_CACHE)
            test_dataset = load_from_disk(TEST_CACHE)
        else:
            val_dataset = aggregate_sentences_by_image(val_dataset_flat)
            test_dataset = aggregate_sentences_by_image(test_dataset_flat)
            val_dataset.save_to_disk(VAL_CACHE)
            test_dataset.save_to_disk(TEST_CACHE)
            breakpoint()
    elif cfg.data.name == "IconQA":
        from datasets import load_from_disk

        HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        TRAIN_CACHE = os.path.join(HF_HOME, "datasets", "iconqa_train_ds")
        VAL_CACHE = os.path.join(HF_HOME, "datasets", "iconqa_val_ds")

        if os.path.exists(TRAIN_CACHE) and os.path.exists(VAL_CACHE):
            print("✅ Using cached train/val datasets")
            train_dataset = load_from_disk(TRAIN_CACHE)
            val_dataset = load_from_disk(VAL_CACHE)
        else:
            raise ValueError("IconQA dataset cache not found. Please create the cache first.")
        _split_ds = val_dataset.train_test_split(test_size=0.8, seed=42)  # type: ignore
        val_dataset = _split_ds["train"]
        test_dataset = _split_ds["test"]
    else:
        raise NotImplementedError(
            f"Dataset loading for {cfg.data.name} is not implemented yet for VLM."
        )

    # When only we cares evaluation, train_dataset can be None
    train_loader = None
    val_loader = None
    test_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,  # type: ignore
            batch_size=cfg.dataloader.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.train_num_workers,
            pin_memory=True,
        )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,  # type: ignore
            batch_size=cfg.dataloader.val_batch_size,
            shuffle=val_shuffle,
            num_workers=cfg.dataloader.val_num_workers,
            pin_memory=True,
        )
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,  # type: ignore
            batch_size=cfg.dataloader.get("test_batch_size", cfg.dataloader.val_batch_size),
            shuffle=False,
            num_workers=cfg.dataloader.get("test_num_workers", cfg.dataloader.val_num_workers),
            pin_memory=True,
        )
    return train_loader, val_loader, test_loader


def _get_dataloader_classification(
    cfg, train_transform=None, val_transform=None, val_shuffle=False
):
    if train_transform is None and val_transform is None:
        from data import Transforms

        transformation = Transforms(cfg)
        train_transform, val_transform = (
            transformation.get(split="train"),
            transformation.get(split="val"),
        )
    # Get the dataset class based on the task type
    dataset_class, kwargs = get_dataset_class(cfg)
    # Get the dataset and dataloader
    kwargs["split"] = "train"
    kwargs["transform"] = train_transform
    train_dataset = dataset_class(**kwargs)
    kwargs["split"] = "val"
    kwargs["transform"] = val_transform
    val_dataset = dataset_class(**kwargs)
    if cfg.data.get("has_test", False):
        test_kwargs = kwargs.copy()
        test_kwargs["split"] = "test"
        test_kwargs["transform"] = val_transform
        test_dataset = dataset_class(**test_kwargs)
    elif cfg.data.get("val_fraction", 0) > 0.0:
        val_fraction = cfg.data.val_fraction
        test_dataset = val_dataset
        train_dataset, val_dataset = split_train_into_train_val(
            train_dataset,
            val_fraction,
            max_val_samples=cfg.data.get("max_val_samples", 5000),
            seed=cfg.data.get("split_seed", 0),
        )
    else:
        raise ValueError("Either 'has_test' or 'val_fraction' must be specified in cfg.data")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.train_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=val_shuffle,
        num_workers=cfg.dataloader.val_num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.get("test_batch_size", cfg.dataloader.val_batch_size),
        shuffle=False,
        num_workers=cfg.dataloader.get("test_num_workers", cfg.dataloader.val_num_workers),
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def split_train_into_train_val(dataset, val_fraction, max_val_samples=5000, seed=0):
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))

    return trainset, valset


def get_dataloader(cfg, train_transform=None, val_transform=None, val_shuffle=False):
    """Get the dataloader for the given dataset and task"""
    # For nli task, we get the dataloaders from the config directly.
    if cfg.task.name == "sequence_classification":
        return _get_dataloader_nli(cfg, val_shuffle)
    # For VLM tasks, we have a separate dataloader function
    elif cfg.task.name in ["vqa", "captioning"]:
        return _get_dataloader_vlm(cfg, val_shuffle)
    # For image classification task
    elif cfg.task.name == "classification":
        return _get_dataloader_classification(cfg, train_transform, val_transform, val_shuffle)
    else:
        raise NotImplementedError(f"Dataloader loading for {cfg.task.name} is not implemented yet.")


def get_base_model(cfg):
    """Get the base model based on the configuration"""
    if cfg.task.name == "classification" and "clip" in cfg.model.name:
        from models import ViTClipClassifier

        # Load the pre-trained model
        model = ViTClipClassifier(
            cliphead_path=os.path.join(cfg.model.cliphead_dir, cfg.data.cliphead_name),
            model_id=cfg.model.id,
        )
    elif cfg.task.name == "sequence_classification":
        from models import SequenceClassifier

        # Load the pre-trained model
        model = SequenceClassifier(model_id=cfg.model.id, num_classes=cfg.data.num_classes)
        if hasattr(cfg.data, "mask_class"):
            model.mask_class = cfg.data.mask_class
        model.vit.config.pad_token_id = model.pad_token_id = get_tokenizer(cfg).pad_token_id

    elif cfg.task.name in ["vqa", "captioning"]:
        from models import LlavaWrapper

        model = LlavaWrapper(
            model_id=cfg.model.id,
            dtype=torch.bfloat16 if cfg.model.bf16 else torch.float16,
            attn_implementation="flash_attention_2" if cfg.model.flash_attention else "sdpa",
            size=cfg.processor.size,
            use_fast=cfg.processor.use_fast,
        )

    else:
        raise NotImplementedError(f"Base model loading for {cfg.task.name} is not implemented yet.")

    return model


def init_lora_model(cfg):
    """Get the LoRA model based on the configuration"""
    model = get_base_model(cfg)  # Has vit backbone and decoder
    lora_config = get_lora_config(cfg)
    model.vit = get_peft_model(model.vit, lora_config)  # type: ignore
    model.vit.print_trainable_parameters()

    return model


def load_lora_model(model, adapter_path, decoder_path=None):
    """Load a pre-trained LoRA model"""
    model.vit = PeftModel.from_pretrained(model.vit, adapter_path)
    # if the decoder does not exist, just skip loading it
    if decoder_path is not None and hasattr(model, "decoder") and model.decoder is not None:
        model.decoder.load_state_dict(torch.load(decoder_path))
    return model


def save_lora_model(model, adapter_path, decoder_path):
    """Save the LoRA model"""
    model.vit.save_pretrained(adapter_path)
    # if the decoder does not exist, just skip saving it
    if hasattr(model, "decoder") and model.decoder is not None:
        torch.save(model.decoder.state_dict(), decoder_path)
