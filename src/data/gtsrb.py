import csv
import os
import pathlib

import ipdb  # noqa: F401
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset


class GTSRB(Dataset):
    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        retname=True,
    ):
        assert split in ["train", "test", "val"], "Split must be either 'train', 'test', 'val'."
        self.retname = retname
        self.transform = transform
        self._split = split

        self._base_folder = pathlib.Path(root)
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if split == "train" else "Final_Test/Images")
        )

        if self._split == "train":
            _, class_to_idx = find_classes(str(self._target_folder))
            samples = make_dataset(
                str(self._target_folder),
                extensions=(".ppm",),
                class_to_idx=class_to_idx,
            )
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (
                        str(self._target_folder / row["Filename"]),
                        int(row["ClassId"]),
                    )
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.classes = [
            "red and white circle 20 kph speed limit",
            "red and white circle 30 kph speed limit",
            "red and white circle 50 kph speed limit",
            "red and white circle 60 kph speed limit",
            "red and white circle 70 kph speed limit",
            "red and white circle 80 kph speed limit",
            "end / de-restriction of 80 kph speed limit",
            "red and white circle 100 kph speed limit",
            "red and white circle 120 kph speed limit",
            "red and white circle red car and black car no passing",
            "red and white circle red truck and black car no passing",
            "red and white triangle road intersection warning",
            "white and yellow diamond priority road",
            "red and white upside down triangle yield right-of-way",
            "stop",
            "empty red and white circle",
            "red and white circle no truck entry",
            "red circle with white horizonal stripe no entry",
            "red and white triangle with exclamation mark warning",
            "red and white triangle with black left curve approaching warning",
            "red and white triangle with black right curve approaching warning",
            "red and white triangle with black double curve approaching warning",
            "red and white triangle rough / bumpy road warning",
            "red and white triangle car skidding / slipping warning",
            "red and white triangle with merging / narrow lanes warning",
            "red and white triangle with person digging / construction / road work warning",
            "red and white triangle with traffic light approaching warning",
            "red and white triangle with person walking warning",
            "red and white triangle with child and person walking warning",
            "red and white triangle with bicyle warning",
            "red and white triangle with snowflake / ice warning",
            "red and white triangle with deer warning",
            "white circle with gray strike bar no speed limit",
            "blue circle with white right turn arrow mandatory",
            "blue circle with white left turn arrow mandatory",
            "blue circle with white forward arrow mandatory",
            "blue circle with white forward or right turn arrow mandatory",
            "blue circle with white forward or left turn arrow mandatory",
            "blue circle with white keep right arrow mandatory",
            "blue circle with white keep left arrow mandatory",
            "blue circle with white arrows indicating a traffic circle",
            "white circle with gray strike bar indicating no passing for cars has ended",
            "white circle with gray strike bar indicating no passing for trucks has ended",
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        sample = {}
        img_path, label = self._samples[idx]
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
    ROOT = "./data/8vision/gtsrb/"  # Set the root directory to the dataset here
    train_dataset = GTSRB(root=ROOT, split="train")
    val_dataset = GTSRB(root=ROOT, split="val")
    test_dataset = GTSRB(root=ROOT, split="test")
    ipdb.set_trace()
    print(f"Number of samples in the dataset: {len(train_dataset)}")
    print(f"Class names: {train_dataset.classes}")
