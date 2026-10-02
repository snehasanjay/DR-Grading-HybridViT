import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(img_size, train):
    if train:
        # Fundus images have no canonical orientation, so flips and full rotations are safe.
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].tolist()
        self.labels = df["diagnosis"].astype(int).tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, f"{self.ids[i]}.png")).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i], dtype=torch.long)


def make_balanced_sampler(labels, num_classes=5, power=0.5):
    """
    Oversamples minority grades (Severe/Proliferative). power=0.5 uses inverse
    square-root frequency: a middle ground between no balancing (0) and full
    balancing (1), which tends to over-fit the few minority images.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_w = 1.0 / np.power(np.maximum(counts, 1), power)
    sample_w = class_w[labels]
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)
