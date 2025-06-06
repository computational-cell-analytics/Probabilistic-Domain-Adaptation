import torch

import torch_em
from torch_em.data.datasets.light_microscopy.livecell import CELL_TYPES

from common import get_dataloaders, get_model


def train_unet(data_path, dataset_name, **kwargs):
    """Train UNet model.
    """

    train_loader, val_loader = get_dataloaders(
        data_path=data_path, dataset_name=dataset_name, patch_shape=(512, 512), **kwargs
    )
    model = get_model()
    device = "cuda" if torch.cuda.is_available else "cpu"

    trainer = torch_em.default_segmentation_trainer(
        name=f"unet-source-{dataset_name}",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=5e-4,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
    )
    trainer.fit(int(1e5), overwrite_training=False)


def train_livecell():
    for cell_type in CELL_TYPES:
        train_unet(data_path=args.input_path, dataset_name="livecell")


def main(args):
    if args.dataset_name == "livecell":
        train_livecell()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/data")
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-s", "--save_root", type=str, default=None)
    args = parser.parse_args()
    main(args)
