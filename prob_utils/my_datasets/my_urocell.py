import os

from .my_segmentation_datasets import default_dual_segmentation_loader


def get_uro_cell_loader(path, split, download, ndim, **kwargs):
    # Download the UroCell data and prepare it in a specific way.
    from torch_em.data.datasets.electron_microscopy.uro_cell import get_uro_cell_paths
    paths = get_uro_cell_paths(path=os.path.join(path, "urocell"), target="mito", download=download)

    # NOTE: Use all volumes besides the last one, i.e. reserved for testing.
    paths = sorted(paths)
    paths = paths[:-1]

    # Create simple splits: use all crops besides second-last (reserved for val) for training.
    if split == "train":
        paths = paths[:-1]
    elif split == "val":
        paths = [paths[-1]]
    else:
        raise ValueError

    raw_key, label_key = "raw", "labels/mito"

    # Prepare the dataloader.
    return default_dual_segmentation_loader(paths, raw_key, paths, label_key, ndim=ndim, **kwargs)
