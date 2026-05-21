from .huggingface_datasets import HuggingFaceDataset


def prepare_train_loaders(cfg):
    dataset_class = HuggingFaceDataset(
        location=cfg.data.data_path,
        task=cfg.data.name,
        model_name_or_path=cfg.model.id,
        batch_size=cfg.dataloader.train_batch_size,
        num_workers=cfg.dataloader.train_num_workers,
    )
    loaders = {"full": dataset_class.train_loader, "tokenizer": dataset_class.tokenizer}
    return loaders


def prepare_test_loaders(cfg, val_shuffle):
    dataset_class = HuggingFaceDataset(
        location=cfg.data.data_path,
        task=cfg.data.name,
        model_name_or_path=cfg.model.id,
        batch_size=cfg.dataloader.val_batch_size,
        test_batch_size=cfg.dataloader.get("test_batch_size", cfg.dataloader.val_batch_size),
        num_workers=cfg.dataloader.val_num_workers,
        test_num_workers=cfg.dataloader.get("test_num_workers", cfg.dataloader.val_num_workers),
        val_shuffle=val_shuffle,
    )

    loaders = {"test": dataset_class.test_loader, "val": dataset_class.val_loader}

    return loaders
