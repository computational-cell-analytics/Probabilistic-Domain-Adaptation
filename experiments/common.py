import os
from typing import Union, Tuple, Literal

import torch
from torch.utils.data import DataLoader

from torch_em.model import UNet2d, UNet3d
from torch_em.data import datasets, MinForegroundSampler

from prob_utils.models.probabilistic_unet import ProbUNet3D


def get_model(
    dataset_name: str,
    device: Union[str, torch.device],
    backbone: Literal["unet", "punet"] = "unet",
    testing: bool = False,
) -> torch.nn.Module:
    """Get the semantic segmentation model.

    Args:
        dataset_name: Name of dataset chosen.
        device: The torch device.

    Returns:
        The segmentation model.
    """
    if backbone not in ["unet", "punet"]:
        raise ValueError(backbone)

    if dataset_name in ["livecell", "lung_xray"]:
        if backbone == "unet":
            model = UNet2d(
                in_channels=1,
                out_channels=1,
                final_activation="Sigmoid",
                depth=4,
                initial_features=64
            )
        else:
            ...

    elif dataset_name == "em":
        if backbone == "unet":
            model = UNet3d(
                in_channels=1,
                out_channels=1,
                final_activation="Sigmoid",
                depth=4,
                initial_features=64
            )
        else:
            model = ProbUNet3D(
                in_channels=1,
                out_channels=1,
                final_sigmoid=True,
                layer_order="crb",
                num_levels=4,
                f_maps=64,
                prior_layer_order="cr",
                posterior_layer_order="cr",
                encoders_f_maps=64,
                encoder_num_levels=4,
                testing=testing,
            )

    else:
        raise NotImplementedError

    model.to(device)
    return model


def get_dataloaders(
    data_path: Union[os.PathLike, str], dataset_name: str, patch_shape: Tuple[int, ...], **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Get the dataloaders for different datasets.

    Args:
        data_path: The filepath where data is stored.
        dataset_name: Name of dataset chosen.
        patch_shape: The patch shape for training.
        kwargs: Additional keyword arguments compatible for the particular dataset's dataloader.

    Returns:
        The training set dataloader.
        The validation set dataloader.
    """

    if dataset_name == "livecell":
        cell_types = kwargs.get("cell_types", None)  # Get the cell-type from kwargs.
        train_loader = datasets.get_livecell_loader(
            path=data_path, split="train", binary=True, patch_shape=patch_shape, batch_size=4,
            cell_typs=cell_types, download=True, num_workers=16, shuffle=True,
        )
        val_loader = datasets.get_livecell_loader(
            path=data_path, split="val", binary=True, patch_shape=patch_shape, batch_size=1,
            cell_typs=cell_types, download=True, num_workers=16, shuffle=True,
        )

    elif dataset_name == "em":
        sampler = MinForegroundSampler(min_fraction=0.05)
        subtypes = kwargs.get("subtypes", None)  # Get the EM data sub-types from kwargs.
        if subtypes == "mitoem":
            train_loader = datasets.get_mitoem_loader(
                path=data_path, splits="train", patch_shape=patch_shape, batch_size=2, ndim=3,
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
                # HACK
                n_samples=100,
            )
            val_loader = datasets.get_mitoem_loader(
                path=data_path, splits="val", patch_shape=patch_shape, batch_size=1, ndim=3,
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
        elif subtypes == "vnc":
            train_loader = datasets.get_vnc_mito_loader(
                path=data_path, patch_shape=patch_shape, batch_size=2, ndim=3,
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
            val_loader = datasets.get_vnc_mito_loader(
                path=data_path, patch_shape=patch_shape, batch_size=1, ndim=3,
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
        elif subtypes == "urocell":
            train_loader = datasets.get_uro_cell_loader(
                path=data_path, patch_shape=patch_shape, batch_size=2, ndim=3, target="mito",
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
            val_loader = datasets.get_uro_cell_loader(
                path=data_path, patch_shape=patch_shape, batch_size=1, ndim=3, target="mito",
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
        elif subtypes == "lucchi":
            train_loader = datasets.get_lucchi_loader(
                path=data_path, patch_shape=patch_shape, batch_size=2, ndim=3, split="train",
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
            val_loader = datasets.get_lucchi_loader(
                path=data_path, patch_shape=patch_shape, batch_size=1, ndim=3, split="test",
                binary=True, sampler=sampler, download=True, num_workers=16, shuffle=True,
            )
        else:
            raise RuntimeError

    elif dataset_name == "lung_xray":
        # TODO: Add nih, jsrt1, jsrt2, montgomery.
        ...

    else:
        raise ValueError(f"'{dataset_name}' is not a valid dataset choice.")

    return train_loader, val_loader
