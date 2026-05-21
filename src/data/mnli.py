from torch.utils.data import DataLoader

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
    )

    loaders = {
        "test": dataset_class.val_loader,
    }
    if cfg.data.get("val_fraction", 0) > 0.0:
        dataset = dataset_class.tokenized_datasets
        val_test = dataset["validation"].train_test_split(
            test_size=cfg.data.val_fraction, shuffle=False, seed=42
        )
        test_loader = DataLoader(
            val_test["train"],
            shuffle=False,
            collate_fn=dataset_class.collate_fn,
            batch_size=dataset_class.test_batch_size,
            num_workers=dataset_class.test_num_workers,
        )
        val_loader = DataLoader(
            val_test["test"],
            shuffle=val_shuffle,
            collate_fn=dataset_class.collate_fn,
            batch_size=dataset_class.batch_size,
            num_workers=dataset_class.num_workers,
        )
        loaders["test"] = test_loader
        loaders["val"] = val_loader
    return loaders
