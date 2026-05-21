import os

import ipdb  # noqa: F401
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class DTD(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        retname=True,
    ):
        assert split in ["train", "test", "val"], "Split must be either 'train', 'test', 'val'."
        _datadir = os.path.join(root, split)
        self.dataset = ImageFolder(_datadir)

        idx_to_class = dict((v, k) for k, v in self.dataset.class_to_idx.items())
        self.classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
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
    ROOT = "./data/8vision/dtd/"  # Set the root directory to the dataset here
    train_dataset = DTD(root=ROOT, split="train")
    ipdb.set_trace()
    print(f"Number of samples in the dataset: {len(train_dataset)}")
    print(f"Class names: {train_dataset.classes}")
