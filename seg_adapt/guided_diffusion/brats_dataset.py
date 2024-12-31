import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# BraTS数据集类
class BraTSDataSet(Dataset):
    def __init__(
        self,
        resolution,
        images_dir,
        annotations_dir,
        transform=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        self.annotation_paths = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir)]
        self.shard = shard
        self.num_shards = num_shards

        self.image_paths = self.image_paths[shard:][::num_shards]
        self.annotation_paths = self.annotation_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(image_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)
            annotation = self.transform(annotation)

        return image, annotation

# 加载BraTS数据集
def load_brats(
    *,
    images_dir,
    annotations_dir,
    batch_size,
    image_size,
    transform=None,
    validation=True,
    shard=0,
    num_shards=1,
    deterministic=False,
):
    if not images_dir or not annotations_dir:
        raise ValueError("unspecified dataset directories")

    image_dir = os.path.join(images_dir, 'training' if validation else 'training')
    annotation_dir = os.path.join(annotations_dir, 'training' if validation else 'training')

    dataset = BraTSDataSet(
        image_size,
        image_dir,
        annotation_dir,
        transform=transform,
        shard=shard,
        num_shards=num_shards,
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=not deterministic, 
        num_workers=1, 
        drop_last=True
    )

    while True:
        yield from loader

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # 根据你的需要调整归一化参数
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])