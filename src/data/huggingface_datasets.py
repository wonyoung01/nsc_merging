import os

import datasets
import ipdb  # noqa: F401
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

torch.multiprocessing.set_sharing_strategy("file_system")  # Set multiprocessing sharing strategy
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings

ROOT = "./data"  # Path to the root directory of the dataset

task_to_keys = {
    "snli": ("premise", "hypothesis"),
    "mnli": ("text1", "text2"),
    "sick": ("text_a", "text_b"),
    "qnli": ("text1", "text2"),
    "rte": ("text1", "text2"),
    "scitail": ("premise", "hypothesis"),
}

task_ids = {
    "snli": "stanfordnlp/snli",
    "mnli": "SetFit/mnli",
    "sick": "DefenceLab/sick",
    "qnli": "SetFit/qnli",
    "rte": "SetFit/rte",
    "scitail": "allenai/scitail",
}


class HuggingFaceDataset:
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
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.test_num_workers = test_num_workers
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side=padding_side
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.collate_fn = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            return_tensors="pt",
        )

        task_cfg = task_ids[task]  # type: ignore
        if isinstance(task_cfg, dict):
            dataset = datasets.load_dataset(**task_cfg, cache_dir=ROOT)  # type: ignore
        else:
            dataset = datasets.load_dataset(task_cfg, cache_dir=ROOT)

        sentence1_key, sentence2_key = task_to_keys[task]  # type: ignore

        def tokenize_function(examples):
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            outputs = self.tokenizer(*args, truncation=True, max_length=2000)
            return outputs

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[t for t in dataset["train"].column_names if t != "label"],  # type: ignore
        )

        self.tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        self.tokenized_datasets = self.tokenized_datasets.filter(
            lambda example: (
                example["labels"] == 0 or example["labels"] == 1 or example["labels"] == 2
            )
        )

        if "train" in self.tokenized_datasets:
            self.train_loader = DataLoader(
                self.tokenized_datasets["train"],  # type: ignore
                shuffle=True,
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        if "test" in self.tokenized_datasets:
            self.test_loader = DataLoader(
                self.tokenized_datasets["test"],  # type: ignore
                shuffle=False,
                collate_fn=self.collate_fn,
                batch_size=test_batch_size,
                num_workers=test_num_workers,
            )
        if "validation" in self.tokenized_datasets:
            self.val_loader = DataLoader(
                self.tokenized_datasets["validation"],  # type: ignore
                shuffle=val_shuffle,
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                num_workers=num_workers,
            )
