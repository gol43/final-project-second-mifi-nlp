import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultimodalDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dish_id = self.df.loc[idx, "dish_id"]
        text = self.df.loc[idx, "ingredients"]
        target = self.df.loc[idx, "total_calories"]

        img_path = os.path.join(
            "data",
            "images",
            str(dish_id),
            "rgb.png"
        )

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        return {
            "target": target,
            "image": image,
            "text": text
        }


def collate_fn(batch, tokenizer): # собирает несколько объектов в батч и токенизирует все тексты.
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    # используем именно FloatTensor потому-что тут у нас регрессия, а не классификация как в уроках
    targets = torch.FloatTensor([item["target"] for item in batch])

    tokenized_input = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    return {
        "target": targets,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(image_model_name, ds_type="train"):
    cfg = timm.get_pretrained_cfg(image_model_name)

    if ds_type == "train":
        transforms = A.Compose([
            A.SmallestMaxSize(
                max_size=max(cfg.input_size[1], cfg.input_size[2])
            ),
            A.RandomCrop(
                height=cfg.input_size[1],
                width=cfg.input_size[2]
            ),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.5
            ),
            A.ColorJitter(p=0.3),
            A.Normalize(
                mean=cfg.mean,
                std=cfg.std
            ),
            ToTensorV2()
        ])

    else:
        transforms = A.Compose([
            A.SmallestMaxSize(
                max_size=max(cfg.input_size[1], cfg.input_size[2])
            ),
            A.CenterCrop(
                height=cfg.input_size[1],
                width=cfg.input_size[2]
            ),
            A.Normalize(
                mean=cfg.mean,
                std=cfg.std
            ),
            ToTensorV2()
        ])

    return transforms