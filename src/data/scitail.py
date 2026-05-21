import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from .huggingface_datasets import task_ids, task_to_keys

ROOT = "./data"  # Path to the root directory of the dataset


class SCITAIL:
    def __init__(
        self,
        location=None,
        task=None,
        model_name_or_path=None,
        batch_size=32,
        test_batch_size=64,
        num_workers=16,
        test_num_workers=16,
        val_shuffle=False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        data_format = "tsv_format"
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side=padding_side
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dataset = datasets.load_dataset(task_ids[task], data_format, cache_dir=ROOT)
        sentence1_key, sentence2_key = task_to_keys[task]

        self.collate_fn = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            return_tensors="pt",
        )

        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            outputs = self.tokenizer(*args, truncation=True, max_length=1000)
            return outputs

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[t for t in dataset["train"].column_names if t != "label"],
        )

        self.tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        self.tokenized_datasets = self.tokenized_datasets.filter(
            lambda example: example["labels"] == "neutral" or example["labels"] == "entails"
        )
        self.tokenized_datasets = self.tokenized_datasets.map(
            lambda example: {
                "labels": 0
                if example["labels"] == "entails"
                else 1
                if example["labels"] == "neutral"
                else example["labels"]
            }
        )

        if "train" in self.tokenized_datasets:
            self.train_loader = DataLoader(
                self.tokenized_datasets["train"],
                shuffle=True,
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        if "test" in self.tokenized_datasets:
            self.test_loader = DataLoader(
                self.tokenized_datasets["test"],
                shuffle=False,
                collate_fn=self.collate_fn,
                batch_size=test_batch_size,
                num_workers=test_num_workers,
            )
        if "validation" in self.tokenized_datasets:
            self.val_loader = DataLoader(
                self.tokenized_datasets["validation"],
                shuffle=val_shuffle,
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                num_workers=num_workers,
            )


def prepare_train_loaders(cfg):
    dataset_class = SCITAIL(
        location=cfg.data.data_path,
        task=cfg.data.name,
        model_name_or_path=cfg.model.id,
        batch_size=cfg.dataloader.train_batch_size,
        num_workers=cfg.dataloader.train_num_workers,
    )
    loaders = {"full": dataset_class.train_loader, "tokenizer": dataset_class.tokenizer}
    return loaders


def prepare_test_loaders(cfg, val_shuffle=False):
    dataset_class = SCITAIL(
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
