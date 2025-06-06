"""Dataloader wrapped to get the Pseudo Labels and Consensus Responses for Target Training
"""

import torch

import torch_em
from torch_em.data.datasets.util import update_kwargs
from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths

from . import DualImageCollectionDataset


def _livecell_segmentation_loader(
    image_paths, label_paths,
    batch_size, patch_shape,
    label_transform=None,
    label_transform2=None,
    raw_transform=None,
    transform=None,
    label_dtype=torch.float32,
    dtype=torch.float32,
    n_samples=None,
    augmentation1=None,
    augmentation2=None,
    **loader_kwargs
):

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = torch_em.transform.get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = torch_em.transform.get_augmentations(ndim=2)

    ds = DualImageCollectionDataset(
        image_paths, label_paths,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        label_transform2=label_transform2,
        label_dtype=label_dtype,
        augmentation1=augmentation1,
        augmentation2=augmentation2,
        transform=transform,
        n_samples=n_samples
    )

    return torch_em.segmentation.get_data_loader(ds, batch_size, **loader_kwargs)


def get_dual_livecell_loader(
    path,
    patch_shape,
    split,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    cell_types=None,
    label_path=None,
    label_dtype=torch.int64,
    **kwargs
):
    assert split in ("train", "val", "test")
    if cell_types is not None:
        assert isinstance(cell_types, (list, tuple)), \
            f"cell_types must be passed as a list or tuple instead of {cell_types}"

    image_paths, seg_paths = get_livecell_paths(
        path=path, split=split, cell_types=cell_types, label_path=label_path, download=download,
    )

    assert sum((offsets is not None, boundaries, binary)) <= 1
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, add_binary_target=True, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform2', label_transform, msg=msg)
        label_dtype = torch.float32
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)
        label_dtype = torch.float32
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)
        label_dtype = torch.float32

    kwargs.update({"patch_shape": patch_shape})
    return _livecell_segmentation_loader(image_paths, seg_paths, label_dtype=label_dtype, **kwargs)
