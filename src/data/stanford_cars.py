import pathlib

import ipdb  # noqa: F401
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset


class StanfordCars(Dataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split
    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
        retname=True,
    ):
        self.root = root

        self._split = split
        self._base_folder = pathlib.Path(root)
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)[
                "annotations"
            ]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)[
            "class_names"
        ].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.retname = retname
        self.transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        sample = {}
        image_path, target = self._samples[idx]
        # CLIP processor defaultly expects a PIL image and changes the image to RGB mode
        _img = Image.open(image_path)
        sample["image"] = _img
        sample["label"] = target
        # Add the metadata if required
        if self.retname:
            sample["meta"] = {
                "img_name": str(image_path),
                "img_size": (_img.size[1], _img.size[0]),
            }

        # Assuming the transform is a dictionary of transforms
        if self.transform is not None:
            for key in sample.keys():
                if key != "meta" and key in self.transform:
                    sample[key] = self.transform[key](sample[key])
        return sample


if __name__ == "__main__":
    # Example usage
    ROOT = "./data/8vision/stanford_cars/"  # Set the root directory to the dataset here
    train_dataset = StanfordCars(root=ROOT, split="train")
    print(f"Number of samples in the dataset: {len(train_dataset)}")
    print(f"Class names: {train_dataset.classes}")
