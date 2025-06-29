import torch

from common import get_dataloaders, get_model

from prob_utils.trainer.punet_trainer import PUNet_Trainer
from prob_utils.models.utils import ELBO


def train_punet(data_path, dataset_name, patch_shape, save_root, **kwargs):
    """Train PUNet model.
    """

    device = "cuda" if torch.cuda.is_available else "cpu"
    train_loader, val_loader = get_dataloaders(
        data_path=data_path, dataset_name=dataset_name, patch_shape=patch_shape, **kwargs
    )
    model = get_model(dataset_name=dataset_name, device=device, backbone="punet")

    name = f"punet-source-{dataset_name}"
    if "cell_types" in kwargs:
        cell_type = kwargs["cell_types"][0]
        name += f"-{cell_type}"

    if "subtype" in kwargs:
        subtypes = kwargs["subtypes"]
        name += f"-{subtypes}"

    # Other stuff for training.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
    supervised_loss = ELBO()

    trainer = PUNet_Trainer(
        name=name,
        save_root=save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=None,
        device=device,
        lr_scheduler=scheduler,
        optimizer=optimizer,
        mixed_precision=True,
        compile_model=False,
        log_image_interval=100,
        loss=supervised_loss,
        metric=supervised_loss,
    )
    trainer.fit(int(1e5), overwrite_training=False)


def train_em(args):
    # Train source models per mitochondria EM data choices.
    for subtype in ["mitoem", "vnc", "lucchi", "urocell"]:
        train_punet(
            data_path=args.input_path,
            dataset_name="em",
            patch_shape=(16, 512, 512),
            subtypes=subtype,
            save_root=args.save_root,
        )


def main(args):
    if args.dataset_name == "em":
        train_em(args)
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
