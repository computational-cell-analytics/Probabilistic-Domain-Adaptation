from torch_em.data import MinForegroundSampler
from torch_em.data.datasets import get_mitoem_loader


def get_mitoem_loaders(
    data_path,
    patch_shape=(1, 512, 512),
    my_sampler=MinForegroundSampler(min_fraction=0.05)
):
    train_loader = get_mitoem_loader(
        path=data_path,
        splits="train",
        patch_shape=patch_shape,
        batch_size=16,
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
