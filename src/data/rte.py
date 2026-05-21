from torch.utils.data import DataLoader

from .huggingface_datasets import HuggingFaceDataset


class RTE(HuggingFaceDataset):
    def __init__(
        self,
        location=None,
        task=None,
        model_name_or_path=None,
        batch_size=32,
        test_batch_size=64,
        num_workers=16,
        test_num_workers=16,
    ):
        super().__init__(
            location,
            task,
            model_name_or_path,
            batch_size,
            test_batch_size,
            num_workers,
            test_num_workers,
        )

        self.tokenized_datasets = self.tokenized_datasets.map(
            lambda example: {"labels": 2 if example["labels"] == 1 else example["labels"]}
        )  # 0 is entailment, 2 is non-entailment
        self.train_loader = DataLoader(
            self.tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            self.tokenized_datasets["validation"],
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.tokenized_datasets["test"],
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=test_batch_size,
            num_workers=test_num_workers,
        )


def prepare_train_loaders(cfg):
    dataset_class = RTE(
        location=cfg.data.data_path,
        task=cfg.data.name,
        model_name_or_path=cfg.model.id,
        batch_size=cfg.dataloader.train_batch_size,
        num_workers=cfg.dataloader.train_num_workers,
    )
    loaders = {"full": dataset_class.train_loader, "tokenizer": dataset_class.tokenizer}
    return loaders


def prepare_test_loaders(cfg, val_shuffle=False):
    dataset_class = RTE(
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
            test_size=cfg.data.val_fraction, shuffle=True, seed=42
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
