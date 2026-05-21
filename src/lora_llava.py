from omegaconf import OmegaConf
from utils import get_args, get_config, get_dataloader
from vlm.utils import evaluate_llava, train_llava


def get_task(cfg):
    dataset_name = cfg.data.lower() if isinstance(cfg.data, str) else cfg.data.name.lower()
    if dataset_name in [
        "chartqa",
        "docvqa",
        "vizwiz_vqa",
        "iconqa",
    ]:
        return "vqa"
    elif dataset_name in ["flickr", "coco"]:
        return "captioning"
    else:
        raise ValueError(f"Unsupported task for dataset: {dataset_name}")


if __name__ == "__main__":
    args = get_args()
    OmegaConf.register_new_resolver("not", lambda x: not x)
    main_cfg = OmegaConf.load(args.config)
    # Expand full config if it is not complete
    if "complete" in main_cfg and main_cfg.complete:
        cfg = main_cfg
    else:
        main_cfg.task = get_task(main_cfg)
        cfg = get_config(main_cfg, args)
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    if cfg.run.mode == "train":
        assert train_loader is not None and val_loader is not None, (
            "Train and validation dataloaders must be provided for training."
        )
        train_llava(
            cfg=cfg,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
        )
    elif cfg.run.mode == "test":
        assert test_loader is not None, "Test dataloader must be provided for testing."
        metric = evaluate_llava(cfg=cfg, model=None, test_dataset=test_loader.dataset)
        print(metric)
    else:
        raise ValueError(f"Unsupported run mode: {cfg.run.mode}")
