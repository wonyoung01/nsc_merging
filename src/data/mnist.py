import pathlib

import torchvision.datasets as datasets
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        retname=True,
    ):
        assert split in ["train", "test", "val"], "Split must be either 'train', 'test', 'val'."
        self.dataset = datasets.MNIST(
            root=pathlib.Path(root),
            train=(split == "train"),
            download=True,
            transform=None,
        )

        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.retname = retname
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {}
        image, target = self.dataset[idx]
        # CLIP processor defaultly expects a PIL image and changes the image to RGB mode
        sample["image"] = image
        sample["label"] = target
        if self.retname:
            sample["meta"] = {
                "img_size": (image.size[1], image.size[0]),
            }

        if self.transform is not None:
            for key in list(sample.keys()):
                if key != "meta" and key in self.transform:
                    sample[key] = self.transform[key](sample[key])
        return sample
