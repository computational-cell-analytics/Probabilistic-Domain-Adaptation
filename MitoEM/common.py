import os

from torchvision import transforms

from torch_em.data import MinForegroundSampler
from torch_em.data.datasets import get_mitoem_loader
from torch_em.transform.raw import get_raw_transform, AdditiveGaussianNoise, GaussianBlur, RandomContrast

from prob_utils.my_utils import my_standardize_torch
from prob_utils.my_datasets import get_vnc_mito_loader, get_lucchi_loader, get_uro_cell_loader


# Source Model Loaders
def get_mitoem_loaders(
    data_path,
    patch_shape=(1, 512, 512),
    my_sampler=MinForegroundSampler(min_fraction=0.05)
):
    train_loader = get_mitoem_loader(
        path=data_path,
        splits="train",
        patch_shape=patch_shape,
        batch_size=4,
        ndim=2,
        binary=True,
        sampler=my_sampler,
        download=True,
        num_workers=16,
        shuffle=True,
    )

    val_loader = get_mitoem_loader(
        path=data_path,
        splits="val",
        patch_shape=patch_shape,
        batch_size=1,
        ndim=2,
        binary=True,
        sampler=my_sampler,
        download=True,
        num_workers=16,
        shuffle=True,
        n_samples=100,
    )

    return train_loader, val_loader


# Augmentations for Target Domain
def my_weak_augmentations(p=0.25):
    norm = my_standardize_torch
    aug1 = transforms.Compose([
        my_standardize_torch,
        transforms.RandomApply([GaussianBlur()], p=p),
        transforms.RandomApply([AdditiveGaussianNoise(scale=(0, 0.15), clip_kwargs=False)], p=p),
    ])
    return get_raw_transform(normalizer=norm, augmentation1=aug1)


def my_strong_augmentations(p=0.5):
    norm = my_standardize_torch
    aug1 = transforms.Compose([
        my_standardize_torch,
        transforms.RandomApply([GaussianBlur(sigma=(0.6, 3.0))], p=p),
        transforms.RandomApply([AdditiveGaussianNoise(scale=(0.05, 0.25), clip_kwargs=False)], p=p/2),
        transforms.RandomApply([RandomContrast(mean=0.0, alpha=(0.33, 3.0), clip_kwargs=False)], p=p),
    ])
    return get_raw_transform(normalizer=norm, augmentation1=aug1)


# Domain Adaptation Loaders
def get_dual_loaders(
    em_data: str,
    root_input_dir: str,
    patch_shape=(1, 512, 512),
    weak_augs=my_weak_augmentations(),
    strong_augs=my_strong_augmentations(),
    my_sampler=MinForegroundSampler(min_fraction=0.05)
):
    path = os.path.join(root_input_dir, em_data)

    if em_data == "vnc":
        train_loader = get_vnc_mito_loader(
            path=path, partition="tr",
            batch_size=4,
            patch_shape=patch_shape,
            ndim=2,
            binary=True,
            sampler=my_sampler,
            augmentation1=weak_augs,
            augmentation2=strong_augs,
            download=True,
            num_workers=16,
            shuffle=True,
            n_samples=400,
        )

        val_loader = get_vnc_mito_loader(
            path=path,
            partition="ts",
            batch_size=1,
            patch_shape=patch_shape,
            ndim=2,
            binary=True,
            sampler=my_sampler,
            augmentation1=weak_augs,
            augmentation2=strong_augs,
            download=True,
            num_workers=16,
            shuffle=True,
            n_samples=400,
        )

    elif em_data == "lucchi":
        train_loader = get_lucchi_loader(
            path=path,
            split="train",
            batch_size=4,
            ndim=2,
            patch_shape=patch_shape,
            sampler=my_sampler,
            augmentation1=weak_augs,
            augmentation2=strong_augs,
            download=True,
            num_workers=16,
            shuffle=True,
        )

        val_loader = get_lucchi_loader(
            path=path,
            split="test",
            batch_size=1,
            ndim=2,
            patch_shape=patch_shape,
            sampler=my_sampler,
            augmentation1=weak_augs,
            augmentation2=strong_augs,
            download=True,
            num_workers=16,
            shuffle=True,
        )

    elif em_data == "urocell":
        sampler = MinForegroundSampler(min_fraction=0.01)
        train_loader = get_uro_cell_loader(
            path=path,
            split="train",
            patch_shape=patch_shape,
            batch_size=4,
            ndim=2,
            sampler=sampler,
            augmentation1=weak_augs,
            augmentation2=strong_augs,
            download=True,
            num_workers=16,
            shuffle=True,
            n_samples=400,
        )

        val_loader = get_uro_cell_loader(
            path=path,
            split="val",
            patch_shape=patch_shape,
            batch_size=1,
            ndim=2,
            sampler=sampler,
            augmentation1=weak_augs,
            augmentation2=strong_augs,
            download=True,
            num_workers=16,
            shuffle=True,
            n_samples=400,
        )

    return train_loader, val_loader
