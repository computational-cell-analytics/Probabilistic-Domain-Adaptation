import os
from typing import Union, Tuple

import torch
from torch.utils.data import DataLoader

from torch_em.model import UNet2d
from torch_em.data.datasets import get_livecell_loader


CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


def get_model(dataset_name: str, device: Union[str, torch.device]) -> torch.nn.Module:
    """Get the semantic segmentation model.

    Args:
        dataset_name:


    Returns:
        The segmentation model.
    """

    if dataset_name == "livecell":
        model = UNet2d(
            in_channels=1,
            out_channels=1,
            final_activation="Sigmoid",
            depth=4,
            initial_features=64
        )

    else:
        raise NotImplementedError

    model.to(device)
    return model


def get_dataloaders(
    data_path: Union[os.PathLike, str],
    dataset_name: str,
    patch_shape: Tuple[int, ...],
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Get the dataloaders for different datasets.

    Args:
        data_path:
        dataset_name:
        patch_shape:
        kwargs: Additional keyword arguments compatible for the particular dataset's dataloader.

    Returns:
        The training set dataloader.
        The validation set dataloader.
    """

    if dataset_name == "livecell":
        cell_types = kwargs.get("cell_types", None)  # Get the cell-type from kwargs.
        train_loader = get_livecell_loader(
            path=data_path, split="train", binary=True, patch_shape=patch_shape, batch_size=4,
            cell_typs=cell_types, download=True, num_workers=16, shuffle=True,
        )
        val_loader = get_livecell_loader(
            path=data_path, split="val", binary=True, patch_shape=patch_shape, batch_size=1,
            cell_typs=cell_types, download=True, num_workers=16, shuffle=True,
        )

    else:
        raise ValueError(f"'{dataset_name}' is not a valid dataset choice.")

    return train_loader, val_loader
