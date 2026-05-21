import pathlib

import ipdb  # noqa: F401
from torch.utils.data import Dataset
from torchvision.datasets import SVHN as PyTorchSVHN  # noqa: N811


class SVHN(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        retname=True,
    ):
        assert split in ["train", "test", "val"], "Split must be either 'train', 'test', 'val'."
        self.dataset = PyTorchSVHN(
            root=pathlib.Path(root),
            download=True,
            split="train" if split == "train" else "test",
            transform=None,  # No transforms applied here, will be handled later
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
        # Add the metadata if required
        if self.retname:
            sample["meta"] = {
                "img_size": (image.size[1], image.size[0]),
            }

        # Assuming the transform is a dictionary of transforms
        if self.transform is not None:
            for key in sample.keys():
                if key != "meta" and key in self.transform:
                    sample[key] = self.transform[key](sample[key])
        return sample


if __name__ == "__main__":
    # Example usage
    ROOT = "./data/8vision/svhn/"  # Set the root directory to the dataset here
    train_dataset = SVHN(root=ROOT, split="train")
    test_dataset = SVHN(root=ROOT, split="test")
    ipdb.set_trace()
    print(f"Number of samples in the dataset: {len(train_dataset)}")
    print(f"Class names: {train_dataset.classes}")
