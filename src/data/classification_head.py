import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel

from src.utils import get_dataloader

DATASETS = [
    "stanford_cars",
    "dtd",
    "eurosat",
    "gtsrb",
    "mnist",
    "resisc45",
    "sun397",
    "svhn",
]

cars_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"i love my {c}!",
    lambda c: f"a photo of my dirty {c}.",
    lambda c: f"a photo of my clean {c}.",
    lambda c: f"a photo of my new {c}.",
    lambda c: f"a photo of my old {c}.",
]

dtd_template = [
    lambda c: f"a photo of a {c} texture.",
    lambda c: f"a photo of a {c} pattern.",
    lambda c: f"a photo of a {c} thing.",
    lambda c: f"a photo of a {c} object.",
    lambda c: f"a photo of the {c} texture.",
    lambda c: f"a photo of the {c} pattern.",
    lambda c: f"a photo of the {c} thing.",
    lambda c: f"a photo of the {c} object.",
]

eurosat_template = [
    lambda c: f"a centered satellite photo of {c}.",
    lambda c: f"a centered satellite photo of a {c}.",
    lambda c: f"a centered satellite photo of the {c}.",
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

resisc45_template = [
    lambda c: f"satellite imagery of {c}.",
    lambda c: f"aerial imagery of {c}.",
    lambda c: f"satellite photo of {c}.",
    lambda c: f"aerial photo of {c}.",
    lambda c: f"satellite view of {c}.",
    lambda c: f"aerial view of {c}.",
    lambda c: f"satellite imagery of a {c}.",
    lambda c: f"aerial imagery of a {c}.",
    lambda c: f"satellite photo of a {c}.",
    lambda c: f"aerial photo of a {c}.",
    lambda c: f"satellite view of a {c}.",
    lambda c: f"aerial view of a {c}.",
    lambda c: f"satellite imagery of the {c}.",
    lambda c: f"aerial imagery of the {c}.",
    lambda c: f"satellite photo of the {c}.",
    lambda c: f"aerial photo of the {c}.",
    lambda c: f"satellite view of the {c}.",
    lambda c: f"aerial view of the {c}.",
]

sun397_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]


dataset_to_template = {
    "stanford_cars": cars_template,
    "dtd": dtd_template,
    "eurosat": eurosat_template,
    "gtsrb": gtsrb_template,
    "mnist": mnist_template,
    "resisc45": resisc45_template,
    "sun397": sun397_template,
    "svhn": svhn_template,
}


def get_templates(dataset_name):
    assert dataset_name in dataset_to_template, f"Unsupported dataset: {dataset_name}"
    return dataset_to_template[dataset_name]


@torch.no_grad()
def build_classification_head(model, tokenizer, classnames, template, device):
    if not isinstance(model, CLIPModel):
        raise NotImplementedError("Only CLIP models are supported.")

    logit_scale = model.logit_scale

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            embeddings = []
            for t in template:
                tokenized_template = tokenizer(t(classname))
                tokenized_template = {
                    k: torch.tensor(v).to(device).reshape(1, -1)
                    for k, v in tokenized_template.items()
                }
                embedding = model.text_projection(model.text_model(**tokenized_template)[1])
                embeddings.append(embedding)
            embeddings = torch.concat(embeddings, dim=0)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    return zeroshot_weights


if __name__ == "__main__":
    vit_path = "openai/clip-vit-base-patch32"
    model_name = vit_path.split("/")[-1]
    classification_heads_dir = f"./new_cliphead/{model_name}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        vit_path, use_safetensors=True, dtype=torch.float32, device_map=None
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(vit_path)

    dataset_names = DATASETS

    os.makedirs(classification_heads_dir, exist_ok=True)

    language_encoder = model.text_model.eval().to(device)
    for dataset_name in tqdm(dataset_names):
        print(f"On {dataset_name}")
        template = get_templates(dataset_name)
        # Add data_cfg to cfg by cfg.data
        cfg_path = f"./experiments/lora_acquistion/configs/data/{dataset_name.lower()}.yaml"
        cfg = OmegaConf.create()
        cfg.data = OmegaConf.load(cfg_path)
        train_loader, _, _ = get_dataloader(cfg)
        classnames = train_loader.dataset.classes
        clip_encodings = build_classification_head(
            model, processor.tokenizer, classnames, template, device
        )
        torch.save(
            clip_encodings, os.path.join(classification_heads_dir, f"{dataset_name}_head.pt")
        )
