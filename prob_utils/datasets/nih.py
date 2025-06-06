import os
from glob import glob
from sklearn.model_selection import train_test_split

import torch_em

from prob_utils.my_datasets import DualImageCollectionDataset


def my_label_transform(x):
    return (x == 255).astype("float32")


def get_nih_loader(
    data_path,
    split,
    batch_size=1,
    patch_shape=(256, 256),
    val_fraction=0.1,
    augmentation1=None,
    augmentation2=None
):
    if split == "val":
        raw_paths = glob(os.path.join(data_path, "images", "train", "*.png"))
        label_paths = glob(os.path.join(data_path, "labels", "train", "*.png"))
    else:
        raw_paths = glob(os.path.join(data_path, "images", split, "*.png"))
        label_paths = glob(os.path.join(data_path, "labels", split, "*.png"))

    raw_paths.sort()
    label_paths.sort()

    if val_fraction > 0:
        if split in ("train", "val"):
            xt, xv, lt, lv = train_test_split(raw_paths, label_paths, random_state=42, test_size=val_fraction)
            raw_paths = xv if split == "val" else xt
            label_paths = lv if split == "val" else lt

    dataset = DualImageCollectionDataset(
        raw_paths, label_paths, patch_shape,
        raw_transform=torch_em.transform.get_raw_transform(),
        label_transform=my_label_transform,
        augmentation1=augmentation1, augmentation2=augmentation2
    )
    loader = torch_em.segmentation.get_data_loader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    return loader
