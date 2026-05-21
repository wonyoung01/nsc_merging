import os

import ipdb  # noqa: F401
from PIL import Image
from torch.utils.data import Dataset


class SUN397(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        retname=True,
    ):
        assert split in ["train", "val", "test"], "Split must be either 'train',  'val', 'test'"
        self.data_dir = os.path.join(root, "train" if split == "train" else "test")

        self.raw_classes = sorted(os.listdir(self.data_dir))
        self.classes = [
            cls[2:].replace("_", " ")
            for cls in self.raw_classes
            if os.path.isdir(os.path.join(self.data_dir, cls))
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.raw_classes)}
        self.retname = retname
        self.transform = transform
        self.images = self._load_images()

    def _load_images(self):
        """Load images from the dataset directory."""
        images = []
        for cls in self.raw_classes:
            class_dir = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):  # Make sure it's a file, not a directory
                    images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        sample = {}
        img_path, label = self.images[idx]
        image = Image.open(img_path)
        # CLIP processor defaultly expects a PIL image and changes the image to RGB mode
        sample["image"] = image
        sample["label"] = label
        # Add the metadata if required
        if self.retname:
            sample["meta"] = {
                "img_name": os.path.basename(img_path),
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
    ROOT = "./data/8vision/sun397/"
    train_dataset = SUN397(root=ROOT, split="train")
    val_dataset = SUN397(root=ROOT, split="val")
    test_dataset = SUN397(root=ROOT, split="test")
    ipdb.set_trace()
    print(f"Number of samples in the dataset: {len(train_dataset)}")
    print(f"Class names: {train_dataset.raw_classes}")
