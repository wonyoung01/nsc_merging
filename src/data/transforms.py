import torchvision.transforms as transforms
from transformers import CLIPImageProcessor


def _image_transform(split, input_size):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


class Transforms:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_size = cfg.model.input_size

    def get(self, split):
        """
        Returns a transform function based on the configuration and split.

        Args:
            split: The dataset split (e.g., 'train', 'val').

        Returns:
            A transform function that applies the necessary transformations.
        """
        cfg = self.cfg
        # Check validity of the split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")
        # Put the image transform first
        sample = {}
        if "clip" in cfg.model.name:
            # Use CLIP image processor for CLIP models
            sample["image"] = ClipImageTransform()
        else:
            sample["image"] = _image_transform(split, cfg.model.input_size)
        task_list = getattr(cfg, "tasks", [(cfg.data.name, cfg.task.name, cfg.task.tgt)])
        for dataset, task, tgt in task_list:
            fn = getattr(self, task, None)
            if fn is None:
                # Treat as a no-op if the task is not defined
                continue
            sample[tgt] = lambda x: fn(x, dataset, split)  # noqa: B023
        return sample


class ClipImageTransform:
    def __init__(self):
        self.processor = CLIPImageProcessor()

    def __call__(self, x):
        """
        Applies the CLIP image processor to the input image.

        Args:
            x: The input image data.

        Returns:
            Processed image tensor.
        """
        return self.processor(x, return_tensors="pt")["pixel_values"].squeeze(0)
